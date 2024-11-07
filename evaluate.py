import torch
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

from model.lightning_module import FasterRCNNModule
from data.coco_module import CocoDataModule

class COCOEvaluator:
    def __init__(self, coco_gt):
        self.coco_gt = coco_gt
        self.results = []
        
    def update(self, predictions, image_ids):
        for prediction, image_id in zip(predictions, image_ids):
            boxes = prediction['boxes'].cpu()
            scores = prediction['scores'].cpu()
            labels = prediction['labels'].cpu()
            
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                # Convert to COCO format [x, y, width, height]
                bbox = [x1, y1, x2 - x1, y2 - y1]
                
                self.results.append({
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': bbox,
                    'score': float(score)
                })
    
    def evaluate(self):
        if len(self.results) == 0:
            print("No predictions to evaluate!")
            return {}
            
        # Save results to temporary file
        tmp_file = 'tmp_results.json'
        with open(tmp_file, 'w') as f:
            json.dump(self.results, f)
            
        # Initialize COCO detections
        coco_dt = self.coco_gt.loadRes(tmp_file)
        
        # Initialize COCOeval object
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Clean up temporary file
        os.remove(tmp_file)
        
        # Return results
        metrics = {
            'AP': coco_eval.stats[0],  # AP @ IoU=0.50:0.95
            'AP50': coco_eval.stats[1],  # AP @ IoU=0.50
            'AP75': coco_eval.stats[2],  # AP @ IoU=0.75
            'APs': coco_eval.stats[3],   # AP for small objects
            'APm': coco_eval.stats[4],   # AP for medium objects
            'APl': coco_eval.stats[5],   # AP for large objects
        }
        
        return metrics

def main(args):
    # Load model
    model = FasterRCNNModule.load_from_checkpoint(
        args.checkpoint_path,
        num_classes=args.num_classes,
        strict=True
    )
    model.eval()
    
    # Initialize data module
    data_module = CocoDataModule(
        train_ann_file=args.val_ann_file,  # Use validation set for evaluation
        train_img_dir=args.val_img_dir,
        val_ann_file=args.val_ann_file,
        val_img_dir=args.val_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    data_module.setup()
    
    # Initialize COCO ground truth
    coco_gt = COCO(args.val_ann_file)
    evaluator = COCOEvaluator(coco_gt)
    
    # Initialize trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=args.num_gpus,
        precision=16 if args.use_amp else 32
    )
    
    # Evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        for batch in data_module.val_dataloader():
            images, targets = batch
            images = [image.to(device) for image in images]
            
            # Get predictions
            predictions = model(images)
            
            # Get image IDs from targets
            image_ids = [target['image_id'].item() for target in targets]
            
            # Update evaluator
            evaluator.update(predictions, image_ids)
    
    # Compute metrics
    metrics = evaluator.evaluate()
    
    # Print results
    print("\nEvaluation Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    parser = ArgumentParser()
    
    # Data arguments
    parser.add_argument('--val_ann_file', type=str, required=True)
    parser.add_argument('--val_img_dir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=91)
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--use_amp', type=bool, default=True)
    
    args = parser.parse_args()
    main(args)