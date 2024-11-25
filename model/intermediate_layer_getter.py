import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model.
    Designed to work with EfficientNet backbone.
    
    Args:
        model (nn.Module): Model to extract features from
        return_layers (Dict[str, str]): Dictionary mapping original layer names to
            desired output names. For example:
            {'_blocks.32': '0', '_blocks.24': '1', '_blocks.16': '2', '_blocks.8': '3'}
    
    Returns:
        List[Tensor]: List of feature maps from specified intermediate layers
    """

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_modules()]):
            raise ValueError(
                f"return_layers are not present in model. "
                f"Available layers: {[name for name, _ in model.named_modules()]}"
            )

        # Create OrderedDict of layers up to the last requested layer
        orig_name = model._get_name()
        layers = OrderedDict()
        
        # Track the hierarchical structure
        children_names = []
        current_path = []
        
        def _get_children_names(module: nn.Module, prefix: str = '') -> None:
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                children_names.append(full_name)
                _get_children_names(child, full_name)
        
        _get_children_names(model)
        
        # Build the layers dictionary
        for name, module in model.named_children():
            layers[name] = module
            current_path.append(name)
            current = '.'.join(current_path)
            
            if any(layer.startswith(current) for layer in return_layers.keys()):
                if hasattr(module, '_blocks'):
                    # Special handling for EfficientNet blocks
                    new_module = nn.ModuleDict()
                    for block_name, block in module.named_children():
                        new_module[block_name] = block
                        full_name = f"{current}.{block_name}"
                        if full_name in return_layers:
                            break
                    layers[name] = new_module
            else:
                current_path.pop()
                
        super().__init__(layers)
        self.return_layers = return_layers

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            List[torch.Tensor]: List of feature maps from specified intermediate layers
        """
        out = OrderedDict()
        for name, module in self.items():
            if isinstance(module, nn.ModuleDict):
                # Handle EfficientNet blocks
                for block_name, block in module.items():
                    x = block(x)
                    full_name = f"{name}.{block_name}"
                    if full_name in self.return_layers:
                        out_name = self.return_layers[full_name]
                        out[out_name] = x
            else:
                x = module(x)
                if name in self.return_layers:
                    out_name = self.return_layers[name]
                    out[out_name] = x
                    
        # Return features in the order specified by return_layers
        return [out[name] for name in sorted(out.keys())]

class ImageList:
    """
    Structure that holds a list of images (of possibly varying sizes) as a single tensor.
    This works by padding the images to the same size.
    
    Args:
        tensors (torch.Tensor): Tensors of images
        image_sizes (List[Tuple[int, int]]): List of original image sizes
    """
    def __init__(self, tensors: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> 'ImageList':
        """Move the images to the specified device"""
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)