import os
import glob
import random

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms
import numpy as np
import math
from PIL import Image


def horizontal_flip(images, targets, g_targets, o_class_agnostic, g_class_agnostic):
    images = torch.flip(images, [-1])
    if o_class_agnostic == True:
        targets[:, 1] = 1 - targets[:, 1]
    else:
        targets[:, 2] = 1 - targets[:, 2]

    if g_class_agnostic == True:
        g_targets[:, 1] = 1 - g_targets[:, 1]
        g_targets[:, 5] = math.pi - g_targets[:, 5]
    else:
        g_targets[:, 2] = 1 - g_targets[:, 2]
        g_targets[:, 6] = math.pi - g_targets[:, 6]
    return images, targets, g_targets

def rotation(images, targets, g_targets, o_class_agnostic, g_class_agnostic):
    angle = random.choice([math.pi/2, math.pi, math.pi+(math.pi/2)])
    rotation_count = int(angle // (math.pi/2))
    rotated_images = torch.rot90(images, rotation_count , [1, 2])
    
    def rotate_90():
        if o_class_agnostic == True:
            targets[:, 1], targets[:, 2] = targets[:, 2] , 1 - targets[:, 1]
            temp = targets[:, 3].clone()
            targets[:, 3] = targets[:, 4]
            targets[:, 4] = temp
        else:
            targets[:, 2], targets[:, 3] = targets[:, 3] , 1 - targets[:, 2]
            temp = targets[:, 4].clone()
            targets[:, 4] = targets[:, 5]
            targets[:, 5] = temp

        if g_class_agnostic == True:    
            g_targets[:, 1], g_targets[:, 2] = g_targets[:, 2], 1 - g_targets[:, 1]
            g_targets[:, 5] += math.pi/2
            mask = g_targets[:, 5] > math.pi
            g_targets[:, 5] -= mask * math.pi
        else:
            g_targets[:, 2], g_targets[:, 3] = g_targets[:, 3], 1 - g_targets[:, 2]
            g_targets[:, 6] += math.pi/2
            mask = g_targets[:, 6] > math.pi
            g_targets[:, 6] -= mask * math.pi
            
    
    def rotate_180():
        if o_class_agnostic == True:
            targets[:, 1], targets[:, 2] = 1 - targets[:, 1], 1 - targets[:, 2]
        else:
            targets[:, 2], targets[:, 3] = 1 - targets[:, 2], 1 - targets[:, 3]
        
        if g_class_agnostic == True:
            g_targets[:, 1], g_targets[:, 2] = 1 - g_targets[:, 1], 1 - g_targets[:, 2]
        else:
            g_targets[:, 2], g_targets[:, 3] = 1 - g_targets[:, 2], 1 - g_targets[:, 3]
                        
    if angle == math.pi/2:
        rotate_90()
    if angle == math.pi:
        rotate_180()
    if angle == math.pi+math.pi/2:
        rotate_90()
        rotate_180()
    
    return rotated_images, targets, g_targets

def pad_to_square(image, pad_value=0):
    _, h, w = image.shape

    difference = abs(h - w)

    # (top, bottom) padding or (left, right) padding
    if h <= w:
        top = difference // 2
        bottom = difference - difference // 2
        pad = [0, 0, top, bottom]
    else:
        left = difference // 2
        right = difference - difference // 2
        pad = [left, right, 0, 0]

    # Add padding
    image = F.pad(image, pad, mode='constant', value=pad_value)
    return image, pad


def resize(image, size):
    return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, folder_path, image_size):
        self.image_files = sorted(glob.glob("{}/*.*".format(folder_path)))
        self.image_size = image_size

    def __getitem__(self, index):
        image_path = self.image_files[index]

        # Extract image as PyTorch tensor
        image = torchvision.transforms.ToTensor()(Image.open(image_path).convert('RGB'))

        # Pad to square resolution
        image, _ = pad_to_square(image)

        # Resize
        image = resize(image, self.image_size)
        return image_path, image

    def __len__(self):
        return len(self.image_files)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, list_path: str, image_size: int, o_class_agnostic: bool, g_class_agnostic: bool, augment: bool, multiscale: bool, normalized_labels=True):
        with open(list_path, 'r') as file:
            self.trainval_index = [line.strip() for line in file.readlines()]
        self.image_files = ['VMRD/JPEGImages/' + ind + '.jpg' for ind in self.trainval_index]    
        self.od_label_files = ['VMRD/labels/od/'+ ind + '.txt' for ind in self.trainval_index]
        self.gd_label_files = ['VMRD/labels/gd/'+ ind + '.txt' for ind in self.trainval_index]

        self.image_size = image_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.o_class_agnostic = o_class_agnostic
        self.g_class_agnostic = g_class_agnostic
        # self.batch_count = 0

    def __getitem__(self, index):
        # 1. Image
        # -----------------------------------------------------------------------------------
        image_path = self.image_files[index].rstrip()

        # Apply augmentations
        if self.augment:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1, contrast=0.5), # contrast=0.5 추가
                torchvision.transforms.ToTensor()
            ])
        else:
            transforms = torchvision.transforms.ToTensor()

        # Extract image as PyTorch tensor
        image = transforms(Image.open(image_path).convert('RGB'))

        _, h, w = image.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        # Pad to square resolution
        image, pad = pad_to_square(image)
        _, p_h, p_w = image.shape
        padded_h, padded_w = (p_h, p_w)

        # 2. Label
        # -----------------------------------------------------------------------------------
        od_label_path = self.od_label_files[index].rstrip()
        gd_label_path = self.gd_label_files[index].rstrip()
        
        targets = None
        g_targets = None
        if os.path.exists(od_label_path) and os.path.exists(gd_label_path):
            boxes = torch.from_numpy(np.loadtxt(od_label_path).reshape(-1, 5))
            g_boxes = torch.from_numpy(np.loadtxt(gd_label_path).reshape(-1, 6))

            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            g_x1 = w_factor * (g_boxes[:, 1] - g_boxes[:, 3] / 2)
            g_y1 = h_factor * (g_boxes[:, 2] - g_boxes[:, 4] / 2)
            g_x2 = w_factor * (g_boxes[:, 1] + g_boxes[:, 3] / 2)
            g_y2 = h_factor * (g_boxes[:, 2] + g_boxes[:, 4] / 2)

            # Adjust for added padding
            x1 += pad[0]  # left
            y1 += pad[2]  # top
            x2 += pad[0] 
            y2 += pad[2] 
            g_x1 += pad[0]
            g_y1 += pad[2]
            g_x2 += pad[0]
            g_y2 += pad[2]

            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
            g_boxes[:, 1] = ((g_x1 + g_x2) / 2) / padded_w
            g_boxes[:, 2] = ((g_y1 + g_y2) / 2) / padded_h
            g_boxes[:, 3] *= w_factor / padded_w
            g_boxes[:, 4] *= h_factor / padded_h

            if self.o_class_agnostic == True:
                targets = torch.zeros((len(boxes), 5))
                targets[:, 1:] = boxes[:, 1:] # targets = (_, x_center, y_center, w, h)
            else:
                targets = torch.zeros((len(boxes), 6))
                targets[:, 1:] = boxes # targets = (_, class_ind, x_center, y_center, w, h)

            if self.g_class_agnostic == True:
                g_targets = torch.zeros((len(g_boxes), 6)) # g_targets = (_, x_center, y_center, w, h, angle)
                g_targets[:, 1:] = g_boxes[:, 1:]
            else:
                g_targets = torch.zeros((len(g_boxes), 7))
                g_targets[:, 1:] = g_boxes
            
                
        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                image, targets, g_targets = horizontal_flip(image, targets, g_targets, self.o_class_agnostic, self.g_class_agnostic)
            if np.random.random() < 0.5:
                image, targets, g_targets = rotation(image, targets, g_targets, self.o_class_agnostic, self.g_class_agnostic)

        return image_path, image, targets, g_targets

    def __len__(self):
        return len(self.image_files)

    def collate_fn(self, batch):
        paths, images, targets, g_targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        g_targets = [g_boxes for g_boxes in g_targets if g_boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        for i, g_boxes in enumerate(g_targets):
            g_boxes[:, 0] = i
        
        images = torch.stack([resize(image, self.image_size) for image in images])
            
        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None  # No boxes for an image
        
        try:
            g_targets = torch.cat(g_targets, 0)
        except RuntimeError:
            g_targets = None
        
        return paths, images, targets, g_targets
    
class VMRD_Dataset(torch.utils.data.Dataset):
    def __init__(self, list_path: str, image_size: int):
        with open(list_path, 'r') as file:
            self.test_index = [line.strip() for line in file.readlines()]
        self.image_files = ['VMRD/JPEGImages/' + ind + '.jpg' for ind in self.test_index]
        self.od_label_files = ['VMRD/labels/od/'+ ind + '.txt' for ind in self.test_index]
        self.gd_label_files = ['VMRD/labels/gd/'+ ind + '.txt' for ind in self.test_index]

        self.image_size = image_size

    def __getitem__(self, index):
        # 1. Image
        # -----------------------------------------------------------------------------------
        image_path = self.image_files[index].rstrip()

        # Extract image as PyTorch tensor
        image = torchvision.transforms.ToTensor()(Image.open(image_path).convert('RGB'))
        _, h, w = image.shape
        h_factor, w_factor = (h, w)

        # Pad to square resolution
        image, pad = pad_to_square(image)
        _, p_h, p_w = image.shape
        padded_h, padded_w = (p_h, p_w)

        # 2. Label
        # -----------------------------------------------------------------------------------
        od_label_path = self.od_label_files[index].rstrip()
        gd_label_path = self.gd_label_files[index].rstrip()

        targets = None
        g_targets = None
        if os.path.exists(od_label_path) and os.path.exists(gd_label_path):
            boxes = torch.from_numpy(np.loadtxt(od_label_path).reshape(-1, 5))
            g_boxes = torch.from_numpy(np.loadtxt(gd_label_path).reshape(-1, 6))

            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            g_x1 = w_factor * (g_boxes[:, 1] - g_boxes[:, 3] / 2)
            g_y1 = h_factor * (g_boxes[:, 2] - g_boxes[:, 4] / 2)
            g_x2 = w_factor * (g_boxes[:, 1] + g_boxes[:, 3] / 2)
            g_y2 = h_factor * (g_boxes[:, 2] + g_boxes[:, 4] / 2)

            # Adjust for added padding
            x1 += pad[0]  # left
            y1 += pad[2]  # top
            x2 += pad[0]
            y2 += pad[2]
            g_x1 += pad[0]
            g_y1 += pad[2]
            g_x2 += pad[0]
            g_y2 += pad[2]

            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
            g_boxes[:, 1] = ((g_x1 + g_x2) / 2) / padded_w
            g_boxes[:, 2] = ((g_y1 + g_y2) / 2) / padded_h
            g_boxes[:, 3] *= w_factor / padded_w
            g_boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 5))
            targets = boxes
            g_targets = torch.zeros((len(g_boxes), 6))
            g_targets = g_boxes
        
        return image_path, image, targets, g_targets
    
    def __len__(self):
        return len(self.image_files)
    
    def load_data(self, batch):
        paths, images, targets, g_targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        g_targets = [g_boxes for g_boxes in g_targets if g_boxes is not None]

        images = torch.stack([resize(image, self.image_size) for image in images])

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None  # No boxes for an image

        try:
            g_targets = torch.cat(g_targets, 0)
        except RuntimeError:
            g_targets = None

        return paths, images, targets, g_targets