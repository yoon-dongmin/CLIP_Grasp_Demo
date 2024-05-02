import torch
import numpy as np
import tqdm
import math

from shapely.geometry import Polygon
from shapely.affinity import rotate

def parse_data_config(path: str):
    """데이터셋 설정 파일을 parse한다."""
    options = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def load_classes(path: str):
    """클래스 이름을 로드한다."""
    with open(path, "r") as f:
        names = f.readlines()
    for i, name in enumerate(names):
        names[i] = name.strip()
    return names


def init_weights_normal(m):
    """정규분포 형태로 가중치를 초기화한다."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, 0.1)

    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes_original(prediction, rescaled_size: int, original_size: tuple):
    """원본 이미지 크기에 맞게 바운딩 박스를 재조정합니다."""
    ow, oh = original_size  # 원본 이미지의 너비(ow)와 높이(oh)
    resize_ratio = rescaled_size / max(original_size)  # 재조정 비율 계산

    # 적용된 패딩 계산
    if ow > oh:
        resized_w = rescaled_size  # 재조정된 너비는 rescaled_size와 동일
        resized_h = round(min(original_size) * resize_ratio)  # 높이는 원본에서 비율에 맞게 조정
        pad_x = 0  # 너비에는 패딩이 없음
        pad_y = abs(resized_w - resized_h)  # 높이 패딩 계산
    else:
        resized_w = round(min(original_size) * resize_ratio)  # 너비는 원본에서 비율에 맞게 조정
        resized_h = rescaled_size  # 재조정된 높이는 rescaled_size와 동일
        pad_x = abs(resized_w - resized_h)  # 너비 패딩 계산
        pad_y = 0  # 높이에는 패딩이 없음

    # 바운딩 박스 재조정
    prediction[:, 0] = (prediction[:, 0] - pad_x // 2) / resize_ratio  # xmin 좌표 조정
    prediction[:, 1] = (prediction[:, 1] - pad_y // 2) / resize_ratio  # ymin 좌표 조정
    prediction[:, 2] = (prediction[:, 2] - pad_x // 2) / resize_ratio  # xmax 좌표 조정
    prediction[:, 3] = (prediction[:, 3] - pad_y // 2) / resize_ratio  # ymax 좌표 조정

    # 예측 결과가 원본 이미지의 좌표를 넘어가지 못하게 한다.
    for i in range(prediction.shape[0]):  # 각 바운딩 박스에 대하여
        for k in range(0, 3, 2):  # xmin, xmax 좌표 조정
            if prediction[i][k] < 0:
                prediction[i][k] = 0  # 최소값을 0으로 설정
            elif prediction[i][k] > ow:
                prediction[i][k] = ow  # 최대값을 이미지 너비로 설정

        for k in range(1, 4, 2):  # ymin, ymax 좌표 조정
            if prediction[i][k] < 0:
                prediction[i][k] = 0  # 최소값을 0으로 설정
            elif prediction[i][k] > oh:
                prediction[i][k] = oh  # 최대값을 이미지 높이로 설정

    return prediction  # 조정된 바운딩 박스 반환



def rescale_g_boxes_original(prediction, rescaled_size: int, original_size: tuple):
    """원본 이미지 크기에 맞게 바운딩 박스를 재조정합니다."""
    ow, oh = original_size  # 원본 이미지의 너비(ow)와 높이(oh)
    resize_ratio = rescaled_size / max(original_size)  # 재조정 비율 계산

    # 적용된 패딩 계산
    if ow > oh:
        resized_w = rescaled_size  # 재조정된 너비는 rescaled_size와 동일
        resized_h = round(min(original_size) * resize_ratio)  # 높이는 원본에서 비율에 맞게 조정
        pad_x = 0  # 너비에는 패딩이 없음
        pad_y = abs(resized_w - resized_h)  # 높이 패딩 계산
    else:
        resized_w = round(min(original_size) * resize_ratio)  # 너비는 원본에서 비율에 맞게 조정
        resized_h = rescaled_size  # 재조정된 높이는 rescaled_size와 동일
        pad_x = abs(resized_w - resized_h)  # 너비 패딩 계산
        pad_y = 0  # 높이에는 패딩이 없음

    # 바운딩 박스 재조정
    prediction[:, 0] = (prediction[:, 0] - pad_x // 2) / resize_ratio  # xmin 좌표 조정
    prediction[:, 1] = (prediction[:, 1] - pad_y // 2) / resize_ratio  # ymin 좌표 조정
    prediction[:, 2] = (prediction[:, 2]) / resize_ratio  # xmax 좌표 조정 (패딩 없음)
    prediction[:, 3] = (prediction[:, 3]) / resize_ratio  # ymax 좌표 조정 (패딩 없음)

    # 예측 결과가 원본 이미지의 좌표를 넘어가지 못하게 한다.
    for i in range(prediction.shape[0]):  # 각 바운딩 박스에 대하여
        for k in range(0, 3, 2):  # xmin, xmax 좌표 조정
            if prediction[i][k] < 0:
                prediction[i][k] = 0  # 최소값을 0으로 설정
            elif prediction[i][k] > ow:
                prediction[i][k] = ow  # 최대값을 이미지 너비로 설정

        for k in range(1, 4, 2):  # ymin, ymax 좌표 조정
            if prediction[i][k] < 0:
                prediction[i][k] = 0  # 최소값을 0으로 설정
            elif prediction[i][k] > oh:
                prediction[i][k] = oh  # 최대값을 이미지 높이로 설정

    return prediction  # 조정된 바운딩 박스 반환



def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls, class_agnostic):
    """
    Compute the average precision, given the Precision-Recall curve.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    conf_np = np.array([tensor.item() for tensor in conf])
    i = np.argsort(-conf_np)
    tp = np.array(tp)[i]
    conf = conf_np[i]
    if class_agnostic == False:

        pred_cls = np.array([cls.item() for cls in pred_cls])
        pred_cls = pred_cls[i]
        # Find unique classes
        unique_classes = np.unique(target_cls).astype("int32")
    else:
        unique_classes = None

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    if class_agnostic == True:
        if len(tp) == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp).cumsum()
            tpc = (tp).cumsum()

            # Recall
            recall_curve = tpc / len(tp)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(np.trapz(precision_curve, recall_curve))
    
    else:
        for c in unique_classes:
            i = pred_cls == c
            n_gt = (target_cls == c).sum()
            n_p = i.sum()
            
            if n_p == 0 and n_gt == 0:
                continue
            elif n_p == 0 or n_gt == 0:
                ap.append(0)
                r.append(0)
                p.append(0)
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum()
                tpc = (tp[i]).cumsum()

                # Recall
                recall_curve = tpc / (n_gt + 1e-16)
                r.append(recall_curve[-1])

                # Precision
                precision_curve = tpc / (tpc + fpc)
                p.append(precision_curve[-1])

                # AP from recall-precision curve
                ap.append(compute_ap(recall_curve, precision_curve))
                
    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes

def ap_for_g(tp, conf):
    """
    Compute the average precision, given the Precision-Recall curve.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    conf_np = np.array([tensor.item() for tensor in conf])
    i = np.argsort(-conf_np)
    tp = np.array(tp)[i]
    conf = conf_np[i]

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    if len(tp) == 0:
        ap.append(0)
        r.append(0)
        p.append(0)
    else:
        # Accumulate FPs and TPs
        fpc = (1 - tp).cumsum()
        tpc = (tp).cumsum()

        # Recall
        recall_curve = tpc / len(tp)
        r.append(recall_curve[-1])

        # Precision
        precision_curve = tpc / (tpc + fpc)
        p.append(precision_curve[-1])

        # AP from recall-precision curve
        ap.append(np.trapz(precision_curve, recall_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold, o_class_agnostic):
    """Compute true positives, predicted scores and predicted labels per batch."""
    batch_metrics = []
    for i, output in enumerate(outputs):

        if output is None:
            continue
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        if o_class_agnostic == False:
            pred_labels = output[:, -1]
        

        true_positives = np.zeros(pred_boxes.shape[0])
        annotations = targets[targets[:, 0] == i][:, 1:]
        if o_class_agnostic == False:
            target_labels = annotations[:, 0] if len(annotations) else []
        
        if len(annotations):
            detected_boxes = []
            if o_class_agnostic == False:
                target_boxes = annotations[:, 1:]
                for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                    
                    # If targets are found break
                    if len(detected_boxes) == len(annotations):
                        break

                    # Ignore if label is not one of the target labels
                    if pred_label not in target_labels:
                        continue

                    iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                    if iou >= iou_threshold and box_index not in detected_boxes:
                        true_positives[pred_i] = 1
                        detected_boxes += [box_index]
                batch_metrics.append([true_positives, pred_scores, pred_labels])              
            else:
                target_boxes = annotations[:, 0:]
                for pred_i, pred_box in enumerate(pred_boxes):

                    # If targets are found break
                    if len(detected_boxes) == len(annotations):
                        break

                    iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                    if iou >= iou_threshold and box_index not in detected_boxes:
                        true_positives[pred_i] = 1
                        detected_boxes += [box_index]
                batch_metrics.append([true_positives, pred_scores])
    return batch_metrics

def g_get_batch_statistics(g_outputs, g_targets, g_iou_threshold, g_class_agnostic):
    """Compute true positives, predicted scores and predicted labels per batch."""
    batch_metrics = []
    for i, output in enumerate(g_outputs):

        if output is None:
            continue

        pred_boxes = output[:, :5]
        pred_scores = output[:, 5]
        if g_class_agnostic == False:
            pred_labels = output[:, -1]
        
        true_positives = np.zeros(pred_boxes.shape[0])
        annotations = g_targets[g_targets[:, 0] == i][:, 1:]
        if g_class_agnostic == False:
            g_target_labels = annotations[:, 0] if len(annotations) else []

        if len(annotations):
            detected_boxes = []
            if g_class_agnostic == False:
                target_boxes = annotations[:, 1:]
                for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                    if len(detected_boxes) == len(annotations):
                        break

                    if pred_label not in g_target_labels:
                        continue
                    
                    iou, box_index = g_bbox_iou(pred_box.unsqueeze(0), target_boxes, device=torch.device("cuda"), nms_mode = True).max(0)
                    if iou >= g_iou_threshold and box_index not in detected_boxes:
                        true_positives[pred_i] = 1
                        detected_boxes += [box_index]
                batch_metrics.append([true_positives, pred_scores, pred_labels])
            else:
                target_boxes = annotations[:, 0:]
                for pred_i, pred_box in enumerate(pred_boxes):

                # If targets are found break
                    if len(detected_boxes) == len(annotations):
                        break

                    iou, box_index = g_bbox_iou(pred_box.unsqueeze(0), target_boxes, device=torch.device("cuda"), nms_mode = True).max(0)
                    if iou >= g_iou_threshold and box_index not in detected_boxes:
                        true_positives[pred_i] = 1
                        detected_boxes += [box_index]
                batch_metrics.append([true_positives, pred_scores])
    return batch_metrics

def g_get_statistics_VMRD(g_outputs, g_targets, g_iou_threshold):
    metrics = []

    pred_boxes = g_outputs[:, :5]
    pred_scores = g_outputs[:, 5]
    true_positives = np.zeros(pred_boxes.shape[0])

    detected_boxes = []
    for pred_i, pred_box in enumerate(pred_boxes):
        if len(detected_boxes) == len(pred_boxes):
            break

        ious = g_bbox_iou(pred_box.unsqueeze(0), g_targets, device=torch.device("cuda"), nms_mode = True)
        for box_index, iou in enumerate(ious):
            if iou >= g_iou_threshold and box_index not in detected_boxes:
                if g_bbox_angle_diff(pred_boxes[pred_i, 4], g_targets[box_index, 4]) <= math.radians(30) or g_bbox_angle_diff(pred_boxes[pred_i, 4], g_targets[box_index, 4] - math.pi):
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        metrics.append([true_positives, pred_scores])
    return metrics, np.count_nonzero(true_positives == 1)

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    box1 = box1.to(device)
    box2 = box2.to(device) 
    """Returns the IoU of two bounding boxes."""
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def non_max_suppression(prediction, conf_thres, nms_thres, o_class_agnostic):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold

        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        
        # Object confidence times class confidence
        if o_class_agnostic == False:
            score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        else:
            score = image_pred[:, 4]
        
        image_pred = image_pred[(-score).argsort()]

        if o_class_agnostic == False:
            class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
            detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1).cuda()
        else:
            detections = torch.cat((image_pred[:, :5],), 1).cuda()

        
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            if o_class_agnostic == False:
                label_match = detections[0, -1] == detections[:, -1]
                label_match = label_match.to(device=large_overlap.device)
                invalid = large_overlap & label_match
            elif o_class_agnostic == True:
                invalid = large_overlap       
            weights = detections[invalid, 4:5]
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def g_non_max_suppression(prediction, conf_thres, g_nms_thres, g_class_agnostic, device):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    # prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 5] >= conf_thres]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Grasp confidence
        if g_class_agnostic == False:
            score = image_pred[:, 5] * image_pred[:, 6:].max(1)[0]
        else:
            score = image_pred[:, 5]

        # Sort by it
        image_pred = image_pred[(-score).argsort()]

        if g_class_agnostic == False:
            class_confs, class_preds = image_pred[:, 6:].max(1, keepdim=True)
            detections = torch.cat((image_pred[:, :6], class_confs.float(), class_preds.float()), 1).cuda()
        else:
            detections = torch.cat((image_pred[:, :6],), 1).cuda()

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = g_bbox_iou(detections[0, :5].unsqueeze(0), detections[:, :5], device=device, nms_mode=True) > g_nms_thres
            if g_class_agnostic == False:
                label_match = detections[0, -1] == detections[:, -1]
                label_match = label_match.to(device=large_overlap.device)
                invalid = large_overlap & label_match
            else:   
                invalid = large_overlap
            weights = detections[invalid, 5:6]
            # Merge overlapping bboxes by order of confidence
            detections[0, :5] = (weights * detections[invalid, :5]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, target, anchors, ignore_thres, device, pred_cls=None):
    nB = pred_boxes.size(0) # num_batches
    nA = pred_boxes.size(1) # num_anchors
    nG = pred_boxes.size(2) # grid_size

    # Output tensors
    obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool, device=device)
    noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool, device=device)
    iou_scores = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tx = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    ty = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tw = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    th = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    
    if pred_cls is not None:
        nC = pred_cls.size(-1) # num_class
        class_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
        tcls = torch.zeros(nB, nA, nG, nG, nC, dtype=torch.float, device=device)
        target_boxes = target[:, 2:6] * nG
    else:
        target_boxes = target[:, 1:5] * nG
    
    # Convert to position relative to box
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    _, best_ious_idx = ious.max(0)
    
    # Separate target values
    if pred_cls is not None:
        b, target_labels = target[:, :2].long().t()
    else:
        b = target[:, 0].long().t()

    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    # Set masks
    obj_mask[b, best_ious_idx, gj, gi] = 1
    noobj_mask[b, best_ious_idx, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_ious_idx, gj, gi] = gx - gx.floor()
    ty[b, best_ious_idx, gj, gi] = gy - gy.floor()

    # Width and height
    tw[b, best_ious_idx, gj, gi] = torch.log(gw / anchors[best_ious_idx][:, 0] + 1e-16)
    th[b, best_ious_idx, gj, gi] = torch.log(gh / anchors[best_ious_idx][:, 1] + 1e-16)

    if pred_cls is not None:
        # One-hot encoding of label
        tcls[b, best_ious_idx, gj, gi, target_labels] = 1
        class_mask[b, best_ious_idx, gj, gi] = (pred_cls[b, best_ious_idx, gj, gi].argmax(-1) == target_labels).float()
    

    # Compute label correctness and iou at best anchor
    iou_scores[b, best_ious_idx, gj, gi] = bbox_iou(pred_boxes[b, best_ious_idx, gj, gi], target_boxes, x1y1x2y2=False)
    
    tconf = obj_mask.float()
    
    if pred_cls is not None:
        return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
    else:
        return iou_scores, obj_mask, noobj_mask, tx, ty, tw, th, tconf


def g_bbox_angle_diff(anchor_angle, target_angle):
    diff = abs(anchor_angle - target_angle)
    return diff

def g_bbox_iou(box1, box2, device, nms_mode=False):
    if nms_mode==True:
        box1 = box1.expand(box2.size(0), -1)

    iou_values = []
    for box1_info, box2_info in zip(box1, box2):
        box1_center = box1_info[:2].tolist()
        box1_size = box1_info[2:4].tolist()
        box1_angle = box1_info[4].item()

        box2_center = box2_info[:2].tolist()
        box2_size = box2_info[2:4].tolist()
        box2_angle = box2_info[4].item()
                
        # 회전된 사각형의 꼭짓점 계산
        box1_polygon = Polygon([(box1_center[0] - box1_size[0] / 2, box1_center[1] - box1_size[1] / 2),
                                (box1_center[0] - box1_size[0] / 2, box1_center[1] + box1_size[1] / 2),
                                (box1_center[0] + box1_size[0] / 2, box1_center[1] + box1_size[1] / 2),
                                (box1_center[0] + box1_size[0] / 2, box1_center[1] - box1_size[1] / 2)])
        
        coords = box1_polygon.exterior.coords[0]
        if box1_polygon.is_empty or any(math.isnan(float(coord)) or math.isinf(float(coord)) for coord in coords):
            iou = 0
            iou_values.append(iou)
        else:
            rotated_box1 = rotate(box1_polygon, math.degrees(box1_angle), origin=(box1_center[0], box1_center[1]))

            box2_polygon = Polygon([(box2_center[0] - box2_size[0] / 2, box2_center[1] - box2_size[1] / 2),
                                    (box2_center[0] - box2_size[0] / 2, box2_center[1] + box2_size[1] / 2),
                                    (box2_center[0] + box2_size[0] / 2, box2_center[1] + box2_size[1] / 2),
                                    (box2_center[0] + box2_size[0] / 2, box2_center[1] - box2_size[1] / 2)])
            rotated_box2 = rotate(box2_polygon, math.degrees(box2_angle), origin=(box2_center[0], box2_center[1]))

            if not rotated_box1.is_valid:
                iou = 0        
            elif rotated_box1.intersects(rotated_box2):
                iou = round(rotated_box1.intersection(rotated_box2).area / (rotated_box1.union(rotated_box2).area) + 1e-16, 2)
            else:
                iou = 0
            iou_values.append(iou)
    iou_values = torch.as_tensor(iou_values, dtype=torch.float, device=device)      
    return iou_values

def build_g_targets(pred_boxes, g_target, anchors, anchor_angles, g_ignore_thres_angle, device, pred_cls=None):
    nB = pred_boxes.size(0) # num_batches
    nA = pred_boxes.size(1) # num_anchors
    nG = pred_boxes.size(2) # grid_size

    # Output tensors
    obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool, device=device)
    noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool, device=device)
    iou_scores = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tx = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    ty = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tw = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    th = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tangle = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)

    if pred_cls is not None:
        nC = pred_cls.size(-1)
        class_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
        tcls = torch.zeros(nB, nA, nG, nG, nC, dtype=torch.float, device=device)
        g_target_boxes = g_target[:, 2:6] * nG
        g_target_boxes = torch.cat((g_target_boxes, g_target[:, 6:7]), dim=1)
    else:
        g_target_boxes = g_target[:, 1:5] * nG
        g_target_boxes = torch.cat((g_target_boxes, g_target[:, 5:6]), dim=1)

    # Convert to position relative to box
    gxy = g_target_boxes[:, :2]
    gwh = g_target_boxes[:, 2:4]
    g_angle = g_target[:, 5]

    # Get anchors with best iou
    angle_diff = torch.stack([g_bbox_angle_diff(anchor_angle, g_angle) for anchor_angle in anchor_angles])
    # _, best_ious_idx = angle_diff.min(0)
    best_ious_idx = torch.argmin(angle_diff, dim = 0)

    # Separate target values
    if pred_cls is not None:
        b, g_target_labels = g_target[:, :2].long().t()
    else:
        b = g_target[:, 0].long().t()
        
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    
    # Set masks
    obj_mask[b, best_ious_idx, gj, gi] = 1
    noobj_mask[b, best_ious_idx, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_angle_diff in enumerate(angle_diff.t()):
        noobj_mask[b[i], anchor_angle_diff < g_ignore_thres_angle, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_ious_idx, gj, gi] = gx - gx.floor()
    ty[b, best_ious_idx, gj, gi] = gy - gy.floor()

    # Width and height
    tw[b, best_ious_idx, gj, gi] = torch.log(gw / anchors[best_ious_idx][:, 0] + 1e-16)
    th[b, best_ious_idx, gj, gi] = torch.log(gh / anchors[best_ious_idx][:, 1] + 1e-16)

    # Angle
    tangle[b, best_ious_idx, gj, gi] = g_angle - anchor_angles[best_ious_idx]

    if pred_cls is not None:
        tcls[b, best_ious_idx, gj, gi, g_target_labels] = 1
        class_mask[b, best_ious_idx, gj, gi] = (pred_cls[b, best_ious_idx, gj, gi].argmax(-1) == g_target_labels).float()

    # Compute label correctness and iou at best anchor
    iou_scores[b, best_ious_idx, gj, gi] = g_bbox_iou(pred_boxes[b, best_ious_idx, gj, gi], g_target_boxes, device=device, nms_mode=False)

    tconf = obj_mask.float()

    if pred_cls is not None:
        return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tangle, tcls, tconf
    else:
        return iou_scores, obj_mask, noobj_mask, tx, ty, tw, th, tangle, tconf