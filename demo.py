import argparse
import os
import clip
import tqdm
import glob
import csv

import torch
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from PIL import Image

import model.yolov3_GPD
import utils.datasets
import utils.utils
from utils.utils import bbox_iou


parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="data/input", help="path to input image folder")
parser.add_argument("--save_folder", type=str, default='data/result', help="path to saving result folder")
parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--pretrained_weights", type=str, default="weights/multi_task.pt",
                    help="path to pretrained weights file")
parser.add_argument("--image_size", type=int, default=608, help="size of each image dimension")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
parser.add_argument("--g_conf_thres", type=float, default=0.5, help="grasp confidence threshold")
parser.add_argument("--g_nms_thres", type=float, default=0.25, help="iou threshold for non-maximum suppression")
parser.add_argument("--text_query", type=str, default='Can you give me something to write?', help="language commands to be used in clip models")
parser.add_argument("--max_per_image", type=int, default=5, help="max of the number of detected boxes per image")
args = parser.parse_args()

def xywh2xyxy(x): # xywh format -> xyxy format 
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def detection_filter(prediction, max_per_image):  # 최대 허용 갯수 'max_per_image'에 따라 탐지된 바운딩 박스를 필터링합니다.
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # 바운딩 박스 좌표를 (x, y, width, height)에서 (xmin, ymin, xmax, ymax)로 변환
    output = [None]
    
    for image_i, image_pred in enumerate(prediction):  # 각 이미지에 대하여
        
        image_pred = image_pred[image_pred[:, 4] >= 0.1]  # 신뢰도 점수가 0.1 이상인 탐지만을 선택
        score = image_pred[:, 4]  # 신뢰도 점수 추출
        
        image_pred = image_pred[(-score).argsort()]  # 신뢰도 점수가 높은 순으로 정렬
        detections = torch.cat((image_pred[:, :5],), 1).cuda()  # 바운딩 박스와 신뢰도 점수를 GPU에 전송
        
        keep_boxes = []
        while detections.size(0):  # 탐지된 바운딩 박스가 남아 있는 동안
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > args.nms_thres  # IoU 임계값을 기준으로 중복 탐지 여부 결정
            invalid = large_overlap  # 중복된 바운딩 박스
            weights = detections[invalid, 4:5]  # 중복된 바운딩 박스들의 신뢰도를 가중치로 사용
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()  # 가중치 평균을 통해 바운딩 박스 업데이트
            keep_boxes += [detections[0]]  # 최적화된 바운딩 박스 저장
            detections = detections[~invalid]  # 처리된 바운딩 박스 제거
        
        if len(keep_boxes) <= max_per_image:  # 최대 허용 갯수 이하면 모든 바운딩 박스 저장
            output[image_i] = torch.stack(keep_boxes)
        else:  # 최대 허용 갯수 초과하면 가장 높은 신뢰도를 가진 바운딩 박스만 저장
            output[image_i] = torch.stack(keep_boxes[:max_per_image])
        return output  # 결과 반환

if __name__ == '__main__':
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Class agnostic model
    o_num_classes = 0
    g_num_classes = 0

    # Loading the model
    model = model.yolov3_GPD.YOLOv3(args.image_size, o_num_classes, g_num_classes).to(device)
    if args.pretrained_weights.endswith('.pt'):
        model.load_state_dict(torch.load(args.pretrained_weights))
    else:
        model.load_darknet_weights(args.pretrained_weights)

    # Create Dataloader
    dataset = utils.datasets.ImageFolder(args.image_folder, args.image_size) #608x608로 동일한 이미지 사용
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=args.num_workers)

    # Evaluation mode
    model.eval()  

    for path, image in tqdm.tqdm(dataloader, desc='Batch'): 
        with torch.no_grad():
            image = image.to(device)
            _, outputs, g_outputs = model(image) # Forward process
            outputs = detection_filter(outputs, args.max_per_image)

        image = Image.open(path[0]).convert('RGB')
        prediction = utils.utils.rescale_boxes_original(outputs[0], args.image_size, image.size)
        g_prediction = utils.utils.rescale_g_boxes_original(g_outputs[0], args.image_size, image.size)

        # 객체 박스 내부에서 grasp 박스 중 가장 높은 점수를 가진 박스를 선택합니다.
        g_box_list = []  
        for i in range(prediction.size(0)):  # 모든 예측된 객체 박스에 대해 반복
            obj_box = prediction[i] 
            g_prediction = g_prediction.cuda() 
            grasp = g_prediction[g_prediction[:, 5] >= 0.1]  

            # 객체 박스 내에 있는 gras[] 박스를 필터링하기 위한 조건
            condition_x = (grasp[:, 0] >= obj_box[0]) & (grasp[:, 0] <= obj_box[2])  # X 좌표가 객체 박스 내부에 있는지 확인
            condition_y = (grasp[:, 1] >= obj_box[1]) & (grasp[:, 1] <= obj_box[3])  # Y 좌표가 객체 박스 내부에 있는지 확인
            invalid = condition_x & condition_y  # X와 Y 조건을 모두 만족하는 그립 박스를 찾기

            g_cand = grasp[invalid]  # 조건에 맞는 grasp 박스 후보
            if g_cand.numel() == 0:
                continue  # 조건을 만족하는 그립 박스가 없으면 다음 객체 박스로 넘어감
            score = g_cand[:, 5]  # 조건을 만족하는 grasp 박스들의 점수
            g_cand = g_cand[(-score).argsort()]  # 점수가 높은 순으로 그립 박스를 정렬

            g_box_list.append(g_cand[0, :])  # 가장 점수가 높은 grasp 박스를 리스트에 추가
        g_box = torch.stack(g_box_list)  # 선택된 grasp 박스들을 하나의 텐서로 결합
  

        # Visualization
        fig = plt.figure(figsize=(15,7))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(image)

        # Draw object bounding boxes
        obj_output = prediction.cpu()
        for box in obj_output:
            x1, y1, x2, y2, _ = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor=(51/255, 255/255, 51/255), facecolor='none')
            ax.add_patch(rect)
        
        # Draw grasp bounding boxes
        grasp_output = g_box.cpu()
        for grasp in grasp_output:
            cx, cy, w, h, angle, _ = grasp
             # 회전되지 않은 상태의 사각형의 꼭지점 계산
            x1 = cx + w / 2
            y1 = cy - h / 2
            x2 = cx - w / 2
            y2 = cy - h / 2
            x3 = cx - w / 2
            y3 = cy + h / 2
            x4 = cx + w / 2
            y4 = cy + h / 2
            polygon = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
             # 회전 행렬 계산
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rotated_box = np.dot(polygon - (cx, cy), rotation_matrix.T) + (cx, cy) # 각도에 따라 꼭지점 회전

            # Draw lines with the desired color
            ax.plot([rotated_box[0, 0], rotated_box[1, 0]], [rotated_box[0, 1], rotated_box[1, 1]], color='r', linewidth=3)
            ax.plot([rotated_box[1, 0], rotated_box[2, 0]], [rotated_box[1, 1], rotated_box[2, 1]], color='r', linewidth=3)
            ax.plot([rotated_box[2, 0], rotated_box[3, 0]], [rotated_box[2, 1], rotated_box[3, 1]], color='r', linewidth=3)
            ax.plot([rotated_box[3, 0], rotated_box[0, 0]], [rotated_box[3, 1], rotated_box[0, 1]], color='r', linewidth=3)

            ax.plot([rotated_box[0, 0], rotated_box[3, 0]], [rotated_box[0, 1], rotated_box[3, 1]], color='b', linewidth=3)
            ax.plot([rotated_box[1, 0], rotated_box[2, 0]], [rotated_box[1, 1], rotated_box[2, 1]], color='b', linewidth=3)

        ax.axis('off')
        fig.canvas.draw()

        folder_name = path[0].split('\\')[-1].split('.')[0]
        os.makedirs(f'./{args.save_folder}/{folder_name}/cropped_images', exist_ok=True)
        plt.savefig(f'./{args.save_folder}/{folder_name}/detection_result.png', bbox_inches='tight', pad_inches=0)


        # Fine tuned text encoder CLIP model
        clip_model, preprocess = clip.load('ViT-B/16', device=device, jit=False)
        #saved_state_dict = torch.load('weights/clip_best.pt')
        # clip_model.load_state_dict(saved_state_dict)
        # clip_model.eval().float()

        # Crop an image as an object bounding box
        i = 0
        cropped_images = []
        np_image = np.array(image)
        for box in obj_output:
            i += 1
            x1, y1, x2, y2, _ = map(int, box)
            cropped_image = np_image[y1:y2, x1:x2]
            cropped_image = Image.fromarray(cropped_image).convert('RGB')
            file_name = f'./{args.save_folder}/{folder_name}/cropped_images/{i}.jpg'
            cropped_image.save(file_name)
            cropped_images.append(cropped_image)

        # Create image & text features
        image_input = torch.stack([preprocess(im) for im in cropped_images]).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_query = args.text_query
        text_input = clip.tokenize([text_query]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate the similarity score
        similarity = image_features @ text_features.T
        similarity_np = similarity.cpu().numpy()
        
        path = f'./{args.save_folder}/{folder_name}/cropped_images/*.jpg'
        image_files = sorted(glob.glob(path))
        csv_file_name = f'./{args.save_folder}/{folder_name}/similarity_score_output.csv'

        with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Score', 'Image File'])

            # Write scores and file names for each image file to CSV files
            for i in range(similarity_np.shape[0]):
                score = float(similarity_np[i, 0])
                image_file = image_files[i]
                writer.writerow([f'{score:.8f}', image_file])
            
            # Find the most similar images
            max_index = np.argmax(similarity_np)
            most_similar_image = image_files[max_index]

            # Adding information about the most similar images to a CSV file
            writer.writerow(['Most Similar Image', most_similar_image])

        fig, ax = plt.subplots(figsize=(15,7))
        ax.imshow(image)

        # Draw object bounding boxes
        x1, y1, x2, y2 = obj_output[max_index, :-1]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor=(51/255, 255/255, 51/255), facecolor='none')
        ax.add_patch(rect)

        # Draw grasp bounding boxes
        cx, cy, w, h, angle, _ = grasp_output[max_index, :]
        polygon = np.array([(cx + w / 2, cy - h / 2), (cx - w / 2, cy - h / 2), (cx - w / 2, cy + h / 2), (cx + w / 2, cy + h / 2)])
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_box = np.dot(polygon - (cx, cy), rotation_matrix.T) + (cx, cy)

        ax.plot([rotated_box[0, 0], rotated_box[1, 0]], [rotated_box[0, 1], rotated_box[1, 1]], color='r', linewidth=3)
        ax.plot([rotated_box[1, 0], rotated_box[2, 0]], [rotated_box[1, 1], rotated_box[2, 1]], color='r', linewidth=3)
        ax.plot([rotated_box[2, 0], rotated_box[3, 0]], [rotated_box[2, 1], rotated_box[3, 1]], color='r', linewidth=3)
        ax.plot([rotated_box[3, 0], rotated_box[0, 0]], [rotated_box[3, 1], rotated_box[0, 1]], color='r', linewidth=3)

        ax.plot([rotated_box[0, 0], rotated_box[3, 0]], [rotated_box[0, 1], rotated_box[3, 1]], color='b', linewidth=3)
        ax.plot([rotated_box[1, 0], rotated_box[2, 0]], [rotated_box[1, 1], rotated_box[2, 1]], color='b', linewidth=3)

        ax.axis('off')
        ax.set_title(f'{text_query}')

        plt.savefig(f'./{args.save_folder}/{folder_name}/clip_result.png', bbox_inches='tight', pad_inches=1)

