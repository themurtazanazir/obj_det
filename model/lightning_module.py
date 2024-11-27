import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from .architecture import ModifiedFasterRCNN
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile
from collections import defaultdict


class FasterRCNNModule(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        learning_rate=1e-4,
        weight_decay=1e-4,
        max_epochs=100,
        pretrained=True,
        val_ann_file=None,  # Add validation annotation file path
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.model = ModifiedFasterRCNN(num_classes=num_classes, pretrained=pretrained)

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # Initialize COCO evaluator if annotation file is provided
        self.val_coco = COCO(val_ann_file) if val_ann_file else None
        self.val_predictions = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)

        # Combine all losses
        total_loss = sum(loss for loss in loss_dict.values())

        # Log losses
        for name, loss in loss_dict.items():
            self.log(f"train_{name}", loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Get predictions (inference mode)
        predictions = self.model(images)

        # Calculate validation losses (training mode)
        loss_dict = self.model(images, targets)

        # Handle losses differently based on what's returned
        if isinstance(loss_dict, dict):  # Training mode returns dict of losses
            total_loss = sum(loss for loss in loss_dict.values())
            # Log validation losses
            for name, loss in loss_dict.items():
                self.log(f"val_{name}", loss, prog_bar=True)
        else:  # Inference mode returns list of predictions
            total_loss = torch.tensor(
                0.0, device=self.device
            )  # No loss in inference mode
            predictions = (
                loss_dict  # In inference mode, loss_dict actually contains predictions
            )

        # Store predictions for COCO evaluation
        if self.val_coco is not None:
            all_image_id = targets["image_id"]

            for prediction, (image_id, boxes, scores, labels) in zip(
                predictions, all_image_id
            ):
                image_id = image_id.item()
                boxes = prediction["boxes"].cpu()
                scores = prediction["scores"].cpu()
                labels = prediction["labels"].cpu()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                    self.val_predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": bbox,
                            "score": float(score),
                        }
                    )

        return total_loss

    def on_validation_epoch_end(self):
        # Perform COCO evaluation at the end of each validation epoch
        if self.val_coco is not None and len(self.val_predictions) > 0:
            # Save predictions to temporary file
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                json.dump(self.val_predictions, f)
                tmp_file = f.name

            # Initialize COCO detections
            coco_dt = self.val_coco.loadRes(tmp_file)

            # Initialize COCOeval object
            coco_eval = COCOeval(self.val_coco, coco_dt, "bbox")

            # Run evaluation
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Log metrics
            metrics = {
                "val/AP": coco_eval.stats[0],  # AP @ IoU=0.50:0.95
                "val/AP50": coco_eval.stats[1],  # AP @ IoU=0.50
                "val/AP75": coco_eval.stats[2],  # AP @ IoU=0.75
                "val/APs": coco_eval.stats[3],  # AP for small objects
                "val/APm": coco_eval.stats[4],  # AP for medium objects
                "val/APl": coco_eval.stats[5],  # AP for large objects
            }

            self.log_dict(metrics, prog_bar=True)

            # Clear predictions for next epoch
            self.val_predictions = []

    def configure_optimizers(self):
        # Separate parameter groups for backbone and rest of the network
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {
                "params": backbone_params,
                "lr": self.learning_rate / 10,
            },  # Lower LR for backbone
            {"params": other_params, "lr": self.learning_rate},
        ]

        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)

        # OneCycleLR scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[self.learning_rate / 10, self.learning_rate],
            epochs=self.max_epochs,
            steps_per_epoch=self.trainer.estimated_stepping_batches // self.max_epochs,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
