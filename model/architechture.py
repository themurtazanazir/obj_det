import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.ops import MultiScaleRoIAlign

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
            
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
            
        # Top-down pathway
        self.top_down_blocks = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.top_down_blocks.append(conv)
            
        # Bottom-up pathway (Path Aggregation)
        self.bottom_up_blocks = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            self.bottom_up_blocks.append(conv)
            
    def forward(self, x):
        # Bottom-up pathway (initial)
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(x[i]))
            
        # Top-down pathway
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')
            laterals[i - 1] = self.top_down_blocks[used_backbone_levels - i - 1](laterals[i - 1])
            
        # Bottom-up pathway (Path Aggregation)
        outs = [laterals[0]]
        for i in range(used_backbone_levels - 1):
            outs.append(self.bottom_up_blocks[i](outs[-1]) + laterals[i + 1])
            
        # Final convolutions
        for i in range(used_backbone_levels):
            outs[i] = self.fpn_convs[i](outs[i])
            
        return tuple(outs)

class ModifiedFasterRCNN(nn.Module):
    """Modified Faster R-CNN with EfficientNet-B7 backbone and PANet"""
    def __init__(self, num_classes, pretrained=True):
        super(ModifiedFasterRCNN, self).__init__()
        
        # EfficientNet-B7 backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b7') if pretrained else EfficientNet.from_name('efficientnet-b7')
        
        # Extract feature channels from EfficientNet
        self.backbone_channels = [64, 128, 256, 512]  # Example channels, adjust based on EfficientNet-B7
        
        # PANet FPN
        self.fpn = PANetFPN(self.backbone_channels, 256)
        
        # RoI Align
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Region Proposal Network
        self.rpn = RegionProposalNetwork(256, 3)  # 256 is FPN out channels
        
        # Box head
        self.box_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        )
        
        # Box predictor
        self.box_predictor = nn.Sequential(
            nn.Linear(1024, num_classes * 4),  # Box regression
            nn.Linear(1024, num_classes)  # Classification
        )
        
    def forward(self, images, targets=None):
        # Get backbone features
        features = self.backbone.extract_features(images)
        
        # Apply PANet FPN
        fpn_features = self.fpn([features])
        
        # Generate proposals
        proposals, rpn_losses = self.rpn(fpn_features, images.image_sizes, targets)
        
        if self.training:
            # Apply RoI Align and get pooled features
            pooled_features = self.roi_align(
                fpn_features,
                proposals,
                images.image_sizes
            )
            
            # Box head
            box_features = self.box_head(pooled_features.view(pooled_features.shape[0], -1))
            
            # Box predictor
            class_logits, box_regression = self.box_predictor(box_features)
            
            return {
                'loss_classifier': F.cross_entropy(class_logits, targets['labels']),
                'loss_box_reg': F.smooth_l1_loss(box_regression, targets['boxes']),
                **rpn_losses
            }
        else:
            # Inference mode
            pooled_features = self.roi_align(
                fpn_features,
                proposals,
                images.image_sizes
            )
            
            box_features = self.box_head(pooled_features.view(pooled_features.shape[0], -1))
            class_logits, box_regression = self.box_predictor(box_features)
            
            return self.postprocess_detections(
                class_logits,
                box_regression,
                proposals,
                images.image_sizes
            )