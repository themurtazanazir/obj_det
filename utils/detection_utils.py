import torch
from torchvision.ops import nms
import numpy as np

def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    boxes1, boxes2: [N, 4], [M, 4]
    Return: [N, M]
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def box_area(boxes):
    """Compute area of boxes"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

class PostProcessor:
    """Post-processing for object detection"""
    def __init__(
        self,
        score_threshold=0.05,
        nms_threshold=0.5,
        detections_per_img=100
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.detections_per_img = detections_per_img

    def __call__(self, class_logits, box_regression, proposals, image_shapes):
        """
        Args:
            class_logits: [batch_size * num_proposals, num_classes]
            box_regression: [batch_size * num_proposals, num_classes * 4]
            proposals: List[Tensor[num_proposals, 4]]
            image_shapes: List[Tuple[int, int]]
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = torch.softmax(class_logits, -1)

        # Split predictions per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        results = []

        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = clip_boxes_to_image(boxes, image_shape)

            # Create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # Remove predictions with scores below the threshold
            boxes = boxes[:, 1:]  # Remove background
            scores = scores[:, 1:]  # Remove background
            labels = labels[:, 1:]  # Remove background

            # Flatten boxes, scores and labels
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # Remove low scoring boxes
            inds = torch.where(scores > self.score_threshold)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # Apply NMS for each class independently
            keep = nms(boxes, scores, self.nms_threshold)
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            results.append({
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
            })

        return results

def clip_boxes_to_image(boxes, image_shape):
    """Clip boxes to image boundaries"""
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = image_shape

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)