import torch
import torch.nn as nn
import torch.nn.functional as F

class PANetFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(PANetFPN, self).__init__()
        
        # Debug print to verify channel sizes
        print("FPN input channels:", in_channels_list)
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            
            # Initialize weights
            nn.init.kaiming_normal_(lateral_conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(fpn_conv.weight, mode='fan_out', nonlinearity='relu')
            
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
            
        # Top-down pathway
        self.top_down_blocks = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
            self.top_down_blocks.append(conv)
            
        # Bottom-up pathway (Path Aggregation)
        self.bottom_up_blocks = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
            self.bottom_up_blocks.append(conv)
            
    def forward(self, features):
        """
        Args:
            features (List[Tensor]): List of feature maps from the backbone,
                                   ordered from highest resolution to lowest
        
        Returns:
            tuple(Tensor): Tuple of feature maps after PANet FPN processing
        """
        # Bottom-up pathway (initial feature transformation)
        laterals = []
        for i, feature in enumerate(features):
            laterals.append(self.lateral_convs[i](feature))
            
        # Top-down pathway
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Upsample and add
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
            # Refine features
            laterals[i - 1] = self.top_down_blocks[used_backbone_levels - i - 1](laterals[i - 1])
            
        # Bottom-up pathway (Path Aggregation)
        outs = [laterals[0]]
        for i in range(used_backbone_levels - 1):
            # Downsample current level and add feature from lateral path
            out = self.bottom_up_blocks[i](outs[-1]) + laterals[i + 1]
            outs.append(out)
            
        # Final convolutions for feature refinement
        for i in range(used_backbone_levels):
            outs[i] = self.fpn_convs[i](outs[i])
            
        return tuple(outs)