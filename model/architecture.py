import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.ops import MultiScaleRoIAlign, nms
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
from collections import OrderedDict
import math

class PANetFPN(nn.Module):
    """Path Aggregation Network (PANet) implementation"""
    def __init__(self, in_channels_list, out_channels):
        super(PANetFPN, self).__init__()
        
        # Bottom-up pathway
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            
            # Initialize weights
            nn.init.kaiming_normal_(lateral_conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(fpn_conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(lateral_conv.bias, 0)
            nn.init.constant_(fpn_conv.bias, 0)
            
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
            
        # Top-down pathway
        self.top_down_blocks = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
            self.top_down_blocks.append(conv)
            
        # Bottom-up pathway (Path Aggregation)
        self.bottom_up_blocks = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
            self.bottom_up_blocks.append(conv)
            
    def forward(self, x):
        # Bottom-up pathway (initial)
        features = x
        laterals = []
        
        for i, feature in enumerate(features):
            laterals.append(self.lateral_convs[i](feature))
            
        # Top-down pathway
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Upsample
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')
            # Additional convolution
            laterals[i - 1] = self.top_down_blocks[used_backbone_levels - i - 1](laterals[i - 1])
            
        # Bottom-up pathway (Path Aggregation)
        outs = [laterals[0]]
        for i in range(used_backbone_levels - 1):
            out = self.bottom_up_blocks[i](outs[-1])
            out = out + laterals[i + 1]  # Skip connection
            outs.append(out)
            
        # Final convolutions
        results = []
        for i, out in enumerate(outs):
            results.append(self.fpn_convs[i](out))
            
        return tuple(results)

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    """
    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_modules()]):
            raise ValueError("return_layers are not present in model")
            
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        
        # Reorganize modules for EfficientNet
        for name, module in model.named_children():
            layers[name] = module
            if name == '_blocks':  # Special handling for EfficientNet blocks
                break
                
        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            if name == '_blocks':
                # Handle EfficientNet blocks specially
                block_outputs = []
                current_block = 0
                
                for block in module:
                    x = block(x)
                    if f'_blocks.{current_block}' in self.return_layers:
                        out_name = self.return_layers[f'_blocks.{current_block}']
                        out[out_name] = x
                    current_block += 1
            else:
                x = module(x)
                if name in self.return_layers:
                    out_name = self.return_layers[name]
                    out[out_name] = x
                    
        return list(out.values())

class BoxCoder:
    """
    This class encodes and decodes boxes from/to regression parameters
    """
    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0), bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some reference boxes
        """
        ex_widths = proposals[:, 2] - proposals[:, 0]
        ex_heights = proposals[:, 3] - proposals[:, 1]
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0]
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1]
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.
        """
        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

class ModifiedFasterRCNN(nn.Module):
    """Modified Faster R-CNN with EfficientNet-B7 backbone and PANet"""
    def __init__(self, num_classes, pretrained=True):
        super(ModifiedFasterRCNN, self).__init__()
        
        # EfficientNet-B7 backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b7') if pretrained else EfficientNet.from_name('efficientnet-b7')
        
        # Extract feature channels from EfficientNet
        self.backbone_channels = [2560, 2560, 2560, 2560]  # EfficientNet-B7 channels
        self.backbone_features = IntermediateLayerGetter(
            self.backbone,
            return_layers={
                '_blocks.32': '0',
                '_blocks.24': '1',
                '_blocks.16': '2',
                '_blocks.8': '3',
            }
        )
        
        # PANet FPN
        self.fpn = PANetFPN(self.backbone_channels, 256)
        
        # RoI Align
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Region Proposal Network
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        
        rpn_head = RPNHead(
            in_channels=256,  # FPN output channels
            num_anchors=len(aspect_ratios[0]) * len(anchor_sizes)
        )
        
        self.rpn = RegionProposalNetwork(
            anchor_generator=self.anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={'training': 2000, 'testing': 1000},
            post_nms_top_n={'training': 2000, 'testing': 1000},
            nms_thresh=0.7
        )
        
        # Box head
        self.box_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Separate box predictor heads
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
        # Initialize weights
        for module in [self.cls_score, self.bbox_pred]:
            nn.init.normal_(module.weight, std=0.01)
            nn.init.constant_(module.bias, 0)
            
        self.box_coder = BoxCoder()
        self.score_thresh = 0.05
        self.nms_thresh = 0.5
        self.detections_per_img = 100
        self.num_classes = num_classes
        
    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
            
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
            
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        
        # Get backbone features
        features = self.backbone_features(images)
        
        # Apply PANet FPN
        fpn_features = self.fpn(features)
        
        # Convert FPN features to dict for RPN and RoI heads
        feature_dict = {str(i): feature for i, feature in enumerate(fpn_features)}
        
        # Generate proposals
        proposals, proposal_losses = self.rpn(images, feature_dict, targets)
        
        # Create empty dict for all losses
        losses = {}
        
        # If training, add RPN losses
        if self.training:
            losses.update(proposal_losses)
        
        # Get ROI features
        box_features = self.roi_align(
            feature_dict,
            proposals,
            original_image_sizes
        )
        
        # Apply box head
        box_features = self.box_head(box_features.flatten(start_dim=1))
        
        # Get class scores and box regression
        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)
        
        if self.training:
            # Calculate classification and box regression losses
            gt_labels = torch.cat([t["labels"] for t in targets], dim=0)
            gt_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
            
            classification_loss = F.cross_entropy(class_logits, gt_labels)
            box_loss = F.smooth_l1_loss(box_regression, gt_