import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model.
    Specifically designed to work with EfficientNet backbone.
    
    Args:
        model (nn.Module): Model to extract features from
        return_indices (Dict[int, str]): Dictionary mapping block indices to 
            desired output names
    """
    def __init__(self, model: nn.Module, return_indices: Dict[int, str]) -> None:
        self.return_indices = return_indices
        
        # Store the blocks we need
        blocks = []
        if hasattr(model, '_blocks'):
            max_idx = max(return_indices.keys())
            blocks = list(model._blocks[:max_idx + 1])
        else:
            raise ValueError("Model does not have _blocks attribute")
            
        # Create ordered dict of layers
        layers = OrderedDict()
        layers['conv_stem'] = model._conv_stem
        layers['bn0'] = model._bn0
        layers['blocks'] = nn.ModuleList(blocks)
        
        super().__init__(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            List[torch.Tensor]: List of feature maps from specified intermediate layers
        """
        out = OrderedDict()
        
        # Initial stem
        x = self.conv_stem(x)
        x = self.bn0(x)
        
        # Go through blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.return_indices:
                out_name = self.return_indices[idx]
                out[out_name] = x
                
        # Return features in the order specified by return_layers
        return [out[name] for name in sorted(out.keys())]

class ImageList:
    """
    Structure that holds a list of images (of possibly varying sizes) as a single tensor.
    This works by padding the images to the same size.
    """
    def __init__(self, tensors: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> 'ImageList':
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)