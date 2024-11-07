import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
from argparse import ArgumentParser
import json
from pathlib import Path

from model.lightning_module import FasterRCNNModule

class Predictor:
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image, image_tensor
    
    def postprocess_predictions(self, predictions, confidence_threshold=0.5):
        """Filter predictions based on confidence threshold"""
        boxes = predictions['boxes']
        labels = predictions['labels']
        scores = predictions['scores']
        
        # Filter based on confidence
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        return boxes, labels, scores
    
    def visualize_predictions(self, image, boxes, labels, scores):
        """Draw predictions on image"""
        draw = ImageDraw.Draw(image)
        
        # Generate random colors for each class
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.class_names), 3))
        
        for box, label, score in zip(boxes, labels, scores):
            # Convert box coordinates to integers
            box = box.cpu().numpy().astype(np.int32)
            label = label.cpu().item()
            score = score.cpu().item()
            
            # Get class name and color
            class_name = self.class_names[label]
            color = tuple(colors[label])
            
            # Draw box
            draw.rectangle(box.tolist(), outline=color, width=2)
            
            # Draw label
            label_text = f'{class_name}: {score:.2f}'
            draw.text((box[0], box[1] - 10), label_text, fill=color)
        
        return image
    
    def predict_single_image(self, image_path, confidence_threshold=0.5, save_path=None):
        """Predict and visualize single image"""
        # Load and preprocess image
        original_image, image_tensor = self.preprocess_image(image_path)
        
        # Move to device and add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]
        
        # Postprocess predictions
        boxes, labels, scores = self.postprocess_predictions(
            predictions, confidence_threshold)
        
        # Visualize predictions
        result_image = self.visualize_predictions(
            original_image.copy(), boxes, labels, scores)
        
        # Save result if path provided
        if save_path:
            result_image.save(save_path)
        
        return result_image, predictions

def main(args):
    # Load class names
    with open(args.class_names_file, 'r') as f:
        class_names = json.load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FasterRCNNModule.load_from_checkpoint(
        args.checkpoint_path,
        num_classes=len(class_names),
        strict=True
    )
    model = model.to(device)
    model.eval()
    
    # Initialize predictor
    predictor = Predictor(model, device, class_names)
    
    # Process input path
    input_path = Path(args.input_path)
    if input_path.is_file():
        # Single image
        output_path = Path(args.output_dir) / f"{input_path.stem}_pred{input_path.suffix}"
        predictor.predict_single_image(
            str(input_path),
            confidence_threshold=args.confidence_threshold,
            save_path=str(output_path)
        )
    elif input_path.is_dir():
        # Directory of images
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for image_path in input_path.glob('*'):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                output_path = output_dir / f"{image_path.stem}_pred{image_path.suffix}"
                predictor.predict_single_image(
                    str(image_path),
                    confidence_threshold=args.confidence_threshold,
                    save_path=str(output_path)
                )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save results')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--class_names_file', type=str, required=True,
                      help='Path to JSON file containing class names')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                      help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    main(args)