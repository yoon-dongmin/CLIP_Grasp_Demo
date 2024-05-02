import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch.nn.functional as F
import numpy as np

import utils.utils
import math

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets.float(), reduction='none')

        if self.logits:
            pt = torch.sigmoid(bce_loss)
        else:
            pt = torch.exp(-bce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class YOLODetection(nn.Module):
    def __init__(self, anchors, image_size: int, num_classes: int, angle_anchors=None):
        super(YOLODetection, self).__init__()
        self.anchors = anchors
        self.angle_anchors = angle_anchors
        self.num_anchors = len(anchors)
        if angle_anchors is not None:
            self.num_angle_anchors = len(angle_anchors)
        self.num_classes = num_classes
        self.image_size = image_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.focal_loss = FocalLoss()
        self.ignore_thres = 0.5
        self.g_ignore_thres_angle = math.pi/6
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics = {}


    def forward(self, x, targets=None, g_targets=None):
        device = torch.device('cuda' if x.is_cuda else 'cpu')

        num_batches = x.size(0)  # x = (batch_size, od_final_out_channel, grid_size, grid_size)
        grid_size = x.size(2)

        # 출력값 형태 변환 (num_batches, self.num_anchors, grid_size, grid_size, self.num_classes + n)
        n = 5
        if self.angle_anchors is not None:
            n = 6
        prediction = (
            x.view(num_batches, self.num_anchors, self.num_classes + n, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )

        # Get outputs
        cx = torch.sigmoid(prediction[..., 0])  # Center x  #ex) size-(2, 3, 19, 19)
        cy = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)
        
        if self.angle_anchors is not None:
            angle = prediction[..., 4] # Grasp angle
            pred_conf = torch.sigmoid(prediction[..., 5]) # Grasp confidence
        if self.angle_anchors is not None and self.num_classes != 0:
            pred_cls = F.softmax(prediction[..., 6:], dim=-1) # Grasp class prediction

        elif self.angle_anchors is None and self.num_classes != 0:
            pred_cls = F.softmax(prediction[..., 5:], dim=-1) # Object Class prediction


        # Calculate offsets for each grid
        stride = self.image_size / grid_size
        grid_x = torch.arange(grid_size, dtype=torch.float, device=device).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size])
        grid_y = torch.arange(grid_size, dtype=torch.float, device=device).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])
        scaled_anchors = torch.as_tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors],
                                         dtype=torch.float, device=device)
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        if self.angle_anchors is not None:
            g_anchor_angle = torch.as_tensor([(theta) for theta in self.angle_anchors],
                                           dtype=torch.float, device=device)
            anchor_angle = g_anchor_angle.view((1, self.num_angle_anchors, 1, 1))

        # Add offset and scale with anchors
        if self.angle_anchors is None:
            pred_boxes = torch.zeros_like(prediction[..., :4], device=device)
        else:
            pred_boxes = torch.zeros_like(prediction[..., :5], device=device)
            pred_boxes[..., 4] = anchor_angle + angle
        pred_boxes[..., 0] = cx + grid_x
        pred_boxes[..., 1] = cy + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
 
        if self.angle_anchors is None and self.num_classes != 0:
            pred = (pred_boxes.view(num_batches, -1, 4) * stride,
                    pred_conf.view(num_batches, -1, 1),
                    pred_cls.view(num_batches, -1, self.num_classes))
        elif self.angle_anchors is None and self.num_classes == 0:
            pred = (pred_boxes.view(num_batches, -1, 4) * stride,
                    pred_conf.view(num_batches, -1, 1))
                        
        elif self.angle_anchors is not None and self.num_classes != 0:
            pred = (pred_boxes[..., :4].view(num_batches, -1, 4) * stride,
                    pred_boxes[..., 4].view(num_batches, -1 ,1),
                    pred_conf.view(num_batches, -1, 1),
                    pred_cls.view(num_batches, -1, self.num_classes))
        elif self.angle_anchors is not None and self.num_classes == 0:
            pred = (pred_boxes[..., :4].view(num_batches, -1, 4) * stride,
                    pred_boxes[..., 4].view(num_batches, -1, 1),
                    pred_conf.view(num_batches, -1, 1))

        output = torch.cat(pred, -1)

        if targets is None and g_targets is None:
            return output, 0

        if targets is not None and self.num_classes == 0: 
            iou_scores, obj_mask, no_obj_mask, tx, ty, tw, th, tconf = utils.utils.build_targets(
                pred_boxes=pred_boxes,
                target=targets,
                anchors=scaled_anchors,
                ignore_thres=self.ignore_thres,
                device=device,
                pred_cls=None
            )
            
        elif targets is not None and self.num_classes != 0:
            iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = utils.utils.build_targets(
                pred_boxes=pred_boxes,
                target=targets,
                anchors=scaled_anchors,
                ignore_thres=self.ignore_thres,
                device=device,
                pred_cls=pred_cls
            )
            
        elif g_targets is not None and self.num_classes == 0:
            iou_scores, obj_mask, no_obj_mask, tx, ty, tw, th, tangle, tconf = utils.utils.build_g_targets(
                pred_boxes=pred_boxes,
                g_target=g_targets,
                anchors=scaled_anchors,
                anchor_angles=g_anchor_angle,
                g_ignore_thres_angle=self.g_ignore_thres_angle,
                device=device,
                pred_cls=None
            )
        
        elif g_targets is not None and self.num_classes != 0:
            iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tangle, tcls, tconf = utils.utils.build_g_targets(
                pred_boxes=pred_boxes,
                g_target=g_targets,
                anchors=scaled_anchors,
                anchor_angles=g_anchor_angle,
                g_ignore_thres_angle=self.g_ignore_thres_angle,
                device=device,
                pred_cls=pred_cls
            )

        # Loss: Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(cx[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(cy[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_bbox = loss_x + loss_y + loss_w + loss_h

        if g_targets is not None:
            loss_angle = self.mse_loss(angle[obj_mask], tangle[obj_mask])
            loss_bbox = loss_x + loss_y + loss_w + loss_h + loss_angle
        
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], tconf[no_obj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj
        
        if self.num_classes != 0:
            loss_cls = self.focal_loss(pred_cls[obj_mask], tcls[obj_mask])
            loss_layer = 0.1*(loss_bbox + loss_conf) + loss_cls
        else:
            loss_layer = loss_bbox + loss_conf

        # Metrics
        conf50 = (pred_conf > 0.5).float()
        iou25 = (iou_scores > 0.25).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        if self.num_classes != 0:
            detected_mask = conf50 * class_mask * tconf
            cls_acc = 100 * class_mask[obj_mask].mean()
        else:
            detected_mask = conf50 *  tconf
        conf_obj = pred_conf[obj_mask].mean()
        conf_no_obj = pred_conf[no_obj_mask].mean()
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall25 = torch.sum(iou25 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        # Write loss and metrics
        if self.num_classes != 0:    
            self.metrics = {
                "loss_x": loss_x.detach().cpu().item(),
                "loss_y": loss_y.detach().cpu().item(),
                "loss_w": loss_w.detach().cpu().item(),
                "loss_h": loss_h.detach().cpu().item(),
                "loss_bbox": loss_bbox.detach().cpu().item(),
                "loss_conf": loss_conf.detach().cpu().item(),
                "loss_cls": loss_cls.detach().cpu().item(),
                "loss_layer": loss_layer.detach().cpu().item(),
                "cls_acc": cls_acc.detach().cpu().item(),
                "conf_obj": conf_obj.detach().cpu().item(),
                "conf_no_obj": conf_no_obj.detach().cpu().item(),
                "precision": precision.detach().cpu().item(),
                "recall25": recall25.detach().cpu().item(),
                "recall50": recall50.detach().cpu().item(),
                "recall75": recall75.detach().cpu().item()
            }
        else:
            self.metrics = {
                "loss_x": loss_x.detach().cpu().item(),
                "loss_y": loss_y.detach().cpu().item(),
                "loss_w": loss_w.detach().cpu().item(),
                "loss_h": loss_h.detach().cpu().item(),
                "loss_bbox": loss_bbox.detach().cpu().item(),
                "loss_conf": loss_conf.detach().cpu().item(),
                "loss_layer": loss_layer.detach().cpu().item(),
                "conf_obj": conf_obj.detach().cpu().item(),
                "conf_no_obj": conf_no_obj.detach().cpu().item(),
                "precision": precision.detach().cpu().item(),
                "recall25": recall25.detach().cpu().item(),
                "recall50": recall50.detach().cpu().item(),
                "recall75": recall75.detach().cpu().item()
            }            

        return output, loss_layer
    

class YOLOv3(nn.Module):
    def __init__(self, image_size: int, o_num_classes: int, g_num_classes: int):
        super(YOLOv3, self).__init__()
        od_anchors = {'scale1' : [(180, 180), (120, 120), (60, 60)],
                      'scale2' : [(360, 360), (300, 300), (240, 240)],
                      'scale3' : [(540, 540), (480, 480), (420, 420)]}
        gd_anchors = {'scale2' : [(100, 100), (100, 100), (100, 100), (100, 100)],
                      'scale3' : [(300, 300), (300, 300), (300, 300), (300, 300)]}
        gd_angle_anchors = {'scale2' : [(0), (math.pi/4), (math.pi/2), (math.pi*3/4)],
                            'scale3' : [(0), (math.pi/4), (math.pi/2), (math.pi*3/4)]}
        
        od_final_out_channel = 3 * (4 + 1 + o_num_classes)
        gd_final_out_channel = 4 * (5 + 1 + g_num_classes)

        self.darknet53 = self.make_darknet53()
        self.conv_block3 = self.make_conv_block(1024, 512)
        self.conv_final3 = self.make_conv_final(512, od_final_out_channel)
        self.yolo_layer3 = YOLODetection(od_anchors['scale3'], image_size, o_num_classes)
        self.gd_conv_block3 = self.make_conv_block(1024, 512)
        self.gd_conv_final3 = self.make_conv_final(512, gd_final_out_channel)
        self.gd_yolo_layer3 = YOLODetection(gd_anchors['scale3'], image_size, g_num_classes, angle_anchors = gd_angle_anchors['scale3'])
        
        self.upsample2 = self.make_upsample(512, 256, scale_factor=2)
        self.conv_block2 = self.make_conv_block(768, 256)
        self.conv_final2 = self.make_conv_final(256, od_final_out_channel)
        self.yolo_layer2 = YOLODetection(od_anchors['scale2'], image_size, o_num_classes)
        self.gd_conv_block2 = self.make_conv_block(768, 256)
        self.gd_conv_final2 = self.make_conv_final(256, gd_final_out_channel)
        self.gd_yolo_layer2 = YOLODetection(gd_anchors['scale2'], image_size, g_num_classes, angle_anchors = gd_angle_anchors['scale2'])

        self.upsample1 = self.make_upsample(256, 128, scale_factor=2)
        self.conv_block1 = self.make_conv_block(384, 128)
        self.conv_final1 = self.make_conv_final(128, od_final_out_channel)
        self.yolo_layer1 = YOLODetection(od_anchors['scale1'], image_size, o_num_classes)
        
        self.yolo_layers = [self.yolo_layer1, self.yolo_layer2, self.yolo_layer3]
        self.gd_yolo_layers = [self.gd_yolo_layer2, self.gd_yolo_layer3]

    def forward(self, x, targets=None, g_targets=None):
        loss = 0
        g_loss = 0
        residual_output = {}

        # Darknet-53 forward
        with torch.no_grad():
            for key, module in self.darknet53.items():
                module_type = key.split('_')[0]

                if module_type == 'conv':
                    x = module(x)
                elif module_type == 'residual':
                    out = module(x)
                    x += out
                    if key == 'residual_3_8' or key == 'residual_4_8' or key == 'residual_5_4':
                        residual_output[key] = x

        # Yolov3 layer forward
        conv_block3 = self.conv_block3(residual_output['residual_5_4'])
        scale3 = self.conv_final3(conv_block3)
        yolo_output3, layer_loss = self.yolo_layer3(scale3, targets=targets, g_targets=None)
        loss += layer_loss
        
        gd_conv_block3 = self.gd_conv_block3(residual_output['residual_5_4'])
        gd_scale3 = self.gd_conv_final3(gd_conv_block3)
        gd_yolo_output3, gd_layer_loss = self.gd_yolo_layer3(gd_scale3, targets=None, g_targets=g_targets)
        g_loss += gd_layer_loss

        scale2 = self.upsample2(conv_block3)
        scale2 = torch.cat((scale2, residual_output['residual_4_8']), dim=1)
        conv_block2 = self.conv_block2(scale2)
        scale2 = self.conv_final2(conv_block2)
        yolo_output2, layer_loss = self.yolo_layer2(scale2, targets=targets, g_targets=None)
        loss += layer_loss
        
        g_scale2 = self.upsample2(gd_conv_block3)
        g_scale2 = torch.cat((g_scale2, residual_output['residual_4_8']), dim=1)
        gd_conv_block2 = self.gd_conv_block2(g_scale2)
        gd_scale2 = self.gd_conv_final2(gd_conv_block2)
        gd_yolo_output2, gd_layer_loss = self.gd_yolo_layer2(gd_scale2, targets=None, g_targets=g_targets)
        g_loss += gd_layer_loss

        scale1 = self.upsample1(conv_block2)
        scale1 = torch.cat((scale1, residual_output['residual_3_8']), dim=1)
        conv_block1 = self.conv_block1(scale1)
        scale1 = self.conv_final1(conv_block1)
        yolo_output1, layer_loss = self.yolo_layer1(scale1, targets=targets, g_targets=None)
        loss += layer_loss
        
        total_loss = loss + g_loss

        yolo_outputs = [yolo_output1, yolo_output2, yolo_output3]
        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()
        
        gd_yolo_outputs = [gd_yolo_output2, gd_yolo_output3]
        gd_yolo_outputs = torch.cat(gd_yolo_outputs, 1).detach().cpu()
        
        return (total_loss, yolo_outputs, gd_yolo_outputs)

    def make_darknet53(self):
        modules = nn.ModuleDict()

        modules['conv_1'] = self.make_conv(3, 32, kernel_size=3, requires_grad=False)
        modules['conv_2'] = self.make_conv(32, 64, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_1_1'] = self.make_residual_block(in_channels=64)
        modules['conv_3'] = self.make_conv(64, 128, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_2_1'] = self.make_residual_block(in_channels=128)
        modules['residual_2_2'] = self.make_residual_block(in_channels=128)
        modules['conv_4'] = self.make_conv(128, 256, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_3_1'] = self.make_residual_block(in_channels=256)
        modules['residual_3_2'] = self.make_residual_block(in_channels=256)
        modules['residual_3_3'] = self.make_residual_block(in_channels=256)
        modules['residual_3_4'] = self.make_residual_block(in_channels=256)
        modules['residual_3_5'] = self.make_residual_block(in_channels=256)
        modules['residual_3_6'] = self.make_residual_block(in_channels=256)
        modules['residual_3_7'] = self.make_residual_block(in_channels=256)
        modules['residual_3_8'] = self.make_residual_block(in_channels=256)
        modules['conv_5'] = self.make_conv(256, 512, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_4_1'] = self.make_residual_block(in_channels=512)
        modules['residual_4_2'] = self.make_residual_block(in_channels=512)
        modules['residual_4_3'] = self.make_residual_block(in_channels=512)
        modules['residual_4_4'] = self.make_residual_block(in_channels=512)
        modules['residual_4_5'] = self.make_residual_block(in_channels=512)
        modules['residual_4_6'] = self.make_residual_block(in_channels=512)
        modules['residual_4_7'] = self.make_residual_block(in_channels=512)
        modules['residual_4_8'] = self.make_residual_block(in_channels=512)
        modules['conv_6'] = self.make_conv(512, 1024, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_5_1'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_2'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_3'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_4'] = self.make_residual_block(in_channels=1024)
        return modules

    def make_conv(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1, requires_grad=True):
        module1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        module2 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        if not requires_grad:
            for param in module1.parameters():
                param.requires_grad_(False)
            for param in module2.parameters():
                param.requires_grad_(False)

        modules = nn.Sequential(module1, module2, nn.LeakyReLU(negative_slope=0.1))
        return modules

    def make_conv_block(self, in_channels: int, out_channels: int):
        double_channels = out_channels * 2
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            self.make_conv(out_channels, double_channels, kernel_size=3),
            self.make_conv(double_channels, out_channels, kernel_size=1, padding=0),
            self.make_conv(out_channels, double_channels, kernel_size=3),
            self.make_conv(double_channels, out_channels, kernel_size=1, padding=0)
        )
        return modules

    def make_conv_final(self, in_channels: int, out_channels: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, in_channels * 2, kernel_size=3),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        return modules

    def make_residual_block(self, in_channels: int):
        half_channels = in_channels // 2
        block = nn.Sequential(
            self.make_conv(in_channels, half_channels, kernel_size=1, padding=0, requires_grad=False),
            self.make_conv(half_channels, in_channels, kernel_size=3, requires_grad=False)
        )
        return block

    def make_upsample(self, in_channels: int, out_channels: int, scale_factor: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return modules

    # Load original weights file
    def load_darknet_weights(self, weights_path: str):
        # Open the weights file
        with open(weights_path, "rb") as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values (0~2: version, 3~4: seen)
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        # Load Darknet-53 weights
        for key, module in self.darknet53.items():
            module_type = key.split('_')[0]

            if module_type == 'conv':
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            elif module_type == 'residual':
                for i in range(2):
                    ptr = self.load_bn_weights(module[i][1], weights, ptr)
                    ptr = self.load_conv_weights(module[i][0], weights, ptr)

        # Load YOLOv3 weights
        if weights_path.find('yolov3.weights') != -1:
            for module in self.conv_block3:
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            ptr = self.load_bn_weights(self.conv_final3[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final3[0][0], weights, ptr)
            ptr = self.load_conv_bias(self.conv_final3[1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final3[1], weights, ptr)

            ptr = self.load_bn_weights(self.upsample2[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.upsample2[0][0], weights, ptr)

            for module in self.conv_block2:
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            ptr = self.load_bn_weights(self.conv_final2[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final2[0][0], weights, ptr)
            ptr = self.load_conv_bias(self.conv_final2[1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final2[1], weights, ptr)

            ptr = self.load_bn_weights(self.upsample1[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.upsample1[0][0], weights, ptr)

            for module in self.conv_block1:
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            ptr = self.load_bn_weights(self.conv_final1[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final1[0][0], weights, ptr)
            ptr = self.load_conv_bias(self.conv_final1[1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final1[1], weights, ptr)

    # Load BN bias, weights, running mean and running variance
    def load_bn_weights(self, bn_layer, weights, ptr: int):
        num_bn_biases = bn_layer.bias.numel()

        # Bias
        bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.bias)
        bn_layer.bias.data.copy_(bn_biases)
        ptr += num_bn_biases
        # Weight
        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.weight)
        bn_layer.weight.data.copy_(bn_weights)
        ptr += num_bn_biases
        # Running Mean
        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.running_mean)
        bn_layer.running_mean.data.copy_(bn_running_mean)
        ptr += num_bn_biases
        # Running Var
        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.running_var)
        bn_layer.running_var.data.copy_(bn_running_var)
        ptr += num_bn_biases

        return ptr

    # Load convolution bias
    def load_conv_bias(self, conv_layer, weights, ptr: int):
        num_biases = conv_layer.bias.numel()

        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases]).view_as(conv_layer.bias)
        conv_layer.bias.data.copy_(conv_biases)
        ptr += num_biases

        return ptr

    # Load convolution weights
    def load_conv_weights(self, conv_layer, weights, ptr: int):
        num_weights = conv_layer.weight.numel()

        conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
        conv_weights = conv_weights.view_as(conv_layer.weight)
        conv_layer.weight.data.copy_(conv_weights)
        ptr += num_weights

        return ptr