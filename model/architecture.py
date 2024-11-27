import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from efficientnet_pytorch import EfficientNet
from torchvision.ops import MultiScaleRoIAlign, box_iou, nms
from torchvision.models.detection.rpn import (
    RegionProposalNetwork,
    RPNHead,
    AnchorGenerator,
)

from .panet_fpn import PANetFPN
from .intermediate_layer_getter import IntermediateLayerGetter, ImageList


class ModifiedFasterRCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
    ):
        super(ModifiedFasterRCNN, self).__init__()

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # EfficientNet-B7 backbone
        self.backbone = (
            EfficientNet.from_pretrained("efficientnet-b7")
            if pretrained
            else EfficientNet.from_name("efficientnet-b7")
        )

        # Extract features from these blocks
        # EfficientNet-B7 block output channels:
        # Block 18: 224 channels
        # Block 25: 384 channels
        # Block 31: 640 channels
        # Block 38: 2560 channels
        self.backbone_features = IntermediateLayerGetter(
            self.backbone,
            return_indices={
                38: "0",  # Highest level feature (2560 channels)
                31: "1",  # (640 channels)
                25: "2",  # (384 channels)
                18: "3",  # (224 channels)
            },
        )

        # Update backbone channels to match EfficientNet-B7's architecture
        self.backbone_channels = [384, 224, 160, 160]

        # PANet FPN (will adapt input channels to 256)
        self.fpn = PANetFPN(self.backbone_channels, 256)

        # RoI Align
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )

        # Region Proposal Network
        anchor_sizes = (
            (32,),
            (64,),
            (128,),
            (256,),
        )  # Now 4 sizes, matching our 4 feature maps
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios
        )

        rpn_head = RPNHead(
            in_channels=256,  # FPN output channels
            num_anchors=len(aspect_ratios[0]) * len(anchor_sizes),
        )

        self.rpn = RegionProposalNetwork(
            anchor_generator=rpn_anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7,
        )

        # Box head
        representation_size = 1024
        self.box_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256 * 7 * 7, representation_size),
            nn.ReLU(inplace=True),
            nn.Linear(representation_size, representation_size),
            nn.ReLU(inplace=True),
        )

        # Separate prediction heads
        self.cls_head = nn.Linear(representation_size, num_classes)
        self.reg_head = nn.Linear(representation_size, num_classes * 4)

        # Initialize weights
        for module in [self.box_head, self.cls_head, self.reg_head]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                    nn.init.constant_(layer.bias, 0)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # Handle single image or list of images
        if isinstance(images, (list, tuple)):
            max_size = tuple(max(s) for s in zip(*[img.shape[-2:] for img in images]))
            batch_shape = (len(images),) + images[0].shape[:-2] + max_size
            batched_imgs = images[0].new_full(batch_shape, 0)

            for img, pad_img in zip(images, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

            image_sizes = [img.shape[-2:] for img in images]
            images = ImageList(batched_imgs, image_sizes)

        # Get backbone features
        features = self.backbone_features(images.tensors)

        # Apply PANet FPN
        fpn_features = {str(k): v for k, v in self.fpn(features).items()}
        # Generate proposals
        proposals, rpn_losses = self.rpn(
            images,
            fpn_features,
            targets,
        )

        if self.training:
            # Apply RoI Align and get pooled features
            pooled_features = self.roi_align(
                fpn_features, proposals, images.image_sizes
            )

            # Box head
            box_features = self.box_head(pooled_features)

            # Predictions
            class_logits = self.cls_head(box_features)
            box_regression = self.reg_head(box_features)

            # Calculate losses
            return {
                "loss_classifier": F.cross_entropy(class_logits, targets["labels"]),
                "loss_box_reg": F.smooth_l1_loss(box_regression, targets["boxes"]),
                **rpn_losses,
            }
        else:
            # Inference mode
            pooled_features = self.roi_align(
                fpn_features, proposals, images.image_sizes
            )

            box_features = self.box_head(pooled_features)
            class_logits = self.cls_head(box_features)
            box_regression = self.reg_head(box_features)

            return self.postprocess_detections(
                class_logits, box_regression, proposals, images.image_sizes
            )

    def postprocess_detections(
        self,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        proposals: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Perform post-processing on the outputs of the detector.

        Args:
            class_logits (Tensor): Classification predictions
            box_regression (Tensor): Box regression predictions
            proposals (List[Tensor]): Proposed regions
            image_sizes (List[Tuple[int, int]]): Original image sizes

        Returns:
            List[Dict[str, Tensor]]: List of dictionaries containing:
                - boxes (Tensor): Predicted boxes
                - labels (Tensor): Predicted labels
                - scores (Tensor): Prediction scores
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        # Apply softmax to class predictions
        pred_scores = F.softmax(class_logits, -1)

        # Split predictions per image
        pred_boxes = box_regression.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        results = []
        for boxes, scores, image_size in zip(pred_boxes, pred_scores, image_sizes):
            # Create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # Remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # Flatten predictions
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # Remove low scoring boxes
            inds = torch.where(scores > 0.05)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # Apply NMS
            keep = nms(boxes, scores, 0.5)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # Keep top-k scoring predictions
            keep = torch.argsort(scores, dim=0, descending=True)[:100]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            results.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                    "scores": scores,
                }
            )

        return results
