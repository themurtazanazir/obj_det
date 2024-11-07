from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ModelConfig:
    # Backbone
    backbone_pretrained: bool = True
    freeze_backbone_bn: bool = True
    backbone_trainable_layers: int = 5  # Number of trainable layers in backbone
    
    # FPN
    fpn_out_channels: int = 256
    
    # RPN
    rpn_anchor_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512)
    rpn_aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    rpn_nms_thresh: float = 0.7
    rpn_pre_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_test: int = 1000
    
    # RoI
    box_roi_pool_output_size: int = 7
    box_roi_pool_sampling_ratio: int = 2
    box_head_fc_features: int = 1024
    
    # Detection
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 100
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.25

@dataclass
class TrainingConfig:
    # Basic training settings
    num_epochs: int = 100
    batch_size: int = 2
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Learning rate schedule
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.1
    lr_scheduler_min_lr: float = 1e-7
    
    # Gradient clipping
    gradient_clip_val: float = 0.1
    
    # Mixed precision training
    use_amp: bool = True
    
    # Distributed training
    num_gpus: int = 1
    
    # Logging
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    
    # Checkpointing
    save_top_k: int = 3
    checkpoint_monitor: str = 'val_loss'
    checkpoint_mode: str = 'min'

@dataclass
class DataConfig:
    # Paths
    train_img_dir: str = '/path/to/coco/train2017'
    train_ann_file: str = '/path/to/coco/annotations/instances_train2017.json'
    val_img_dir: str = '/path/to/coco/val2017'
    val_ann_file: str = '/path/to/coco/annotations/instances_val2017.json'
    
    # Dataset settings
    num_classes: int = 91  # 90 COCO classes + background
    min_size: int = 800
    max_size: int = 1333
    
    # Data augmentation
    flip_probability: float = 0.5
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2
    saturation_factor: float = 0.2
    hue_factor: float = 0.1
    
    # Normalization
    pixel_mean: List[float] = (0.485, 0.456, 0.406)
    pixel_std: List[float] = (0.229, 0.224, 0.225)

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()