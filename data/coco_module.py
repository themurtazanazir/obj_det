import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from pycocotools.coco import COCO
import torch

class CocoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ann_file,
        train_img_dir,
        val_ann_file,
        val_img_dir,
        batch_size=2,
        num_workers=4,
        pin_memory=True
    ):
        super().__init__()
        self.train_ann_file = train_ann_file
        self.train_img_dir = train_img_dir
        self.val_ann_file = val_ann_file
        self.val_img_dir = val_img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def setup(self, stage=None):
        # Data augmentation and normalization for training
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Just normalization for validation
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((800, 800)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        self.train_dataset = CocoDetection(
            root=self.train_img_dir,
            annFile=self.train_ann_file,
            transform=self.train_transform,
            target_transform=self.convert_coco_annotations
        )
        
        self.val_dataset = CocoDetection(
            root=self.val_img_dir,
            annFile=self.val_ann_file,
            transform=self.val_transform,
            target_transform=self.convert_coco_annotations
        )
    
    def convert_coco_annotations(self, target):
        """Convert COCO annotations to the format expected by the model"""
        boxes = []
        labels = []
        area = []
        iscrowd = []
        image_ids = []  # Added this
        
        for annotation in target:
            bbox = annotation['bbox']
            # Convert XYWH to XYXY format
            bbox = [
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
            ]
            boxes.append(bbox)
            labels.append(annotation['category_id'])
            area.append(annotation['area'])
            iscrowd.append(annotation['iscrowd'])
            image_ids.append(annotation['image_id'])  # Added this
        
        # If we have any annotations
        if boxes:
            target_dict = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'area': torch.tensor(area),
                'iscrowd': torch.tensor(iscrowd),
                'image_id': torch.tensor(image_ids[0])  # Use the first one since all should be same
            }
        else:
            # Handle empty annotations
            target_dict = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'area': torch.zeros(0),
                'iscrowd': torch.zeros(0),
                'image_id': torch.tensor(0)  # You might want to handle this case differently
            }
        
        return target_dict    
    
    def collate_fn(self, batch):
        """Custom collate function to handle variable size inputs"""
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            # Ensure each target dictionary has all required keys
            processed_target = {
                'boxes': target['boxes'],
                'labels': target['labels'],
                'image_id': target['image_id'],
                'area': target['area'],
                'iscrowd': target['iscrowd']
            }
            targets.append(processed_target)
        
        return images, targets
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
