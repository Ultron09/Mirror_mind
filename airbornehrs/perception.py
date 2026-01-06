"""
Perception Module: Multi-Modal Perception Interface (Universal v1.1.1 "Sentient")
=============================================================================
Modular encoders for Vision, Audio, and Text streams with unified latent fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union

class VisionEncoder(nn.Module):
    """
    Projects [B, C, H, W] images into [B, Seq, model_dim].
    Uses a simple patch-based projection (ViT-style) for maximum flexibility.
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, model_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.projection(x) # [B, model_dim, H', W']
        x = x.flatten(2).transpose(1, 2) # [B, Seq, model_dim]
        return self.norm(x)

class AudioEncoder(nn.Module):
    """
    Projects [B, F, T] spectrograms into [B, Seq, model_dim].
    """
    def __init__(self, in_features: int = 80, model_dim: int = 256):
        super().__init__()
        self.projection = nn.Linear(in_features, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, T] -> [B, T, F]
        x = x.transpose(1, 2)
        x = self.projection(x) # [B, T, model_dim]
        return self.norm(x)

class ModalityFuser(nn.Module):
    """
    Fuses multiple modality streams into a single latent sequence.
    """
    def __init__(self, model_dim: int = 256):
        super().__init__()
        self.model_dim = model_dim
        # Optional: Add modality-specific embeddings
        self.modality_embeddings = nn.ParameterDict({
            'vision': nn.Parameter(torch.randn(1, 1, model_dim)),
            'audio': nn.Parameter(torch.randn(1, 1, model_dim)),
            'text': nn.Parameter(torch.randn(1, 1, model_dim))
        })

    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused = []
        for key, x in modalities.items():
            if key in self.modality_embeddings:
                x = x + self.modality_embeddings[key]
            fused.append(x)
        
        # Concatenate along sequence dimension
        return torch.cat(fused, dim=1) # [B, Seq_Total, model_dim]

class PerceptionGateway(nn.Module):
    """
    The main entry point for raw multi-modal inputs.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.model_dim = config.model_dim
        
        self.encoders = nn.ModuleDict()
        if hasattr(config, 'vision_dim') and config.vision_dim > 0:
            self.encoders['vision'] = VisionEncoder(in_channels=config.vision_dim, model_dim=self.model_dim)
        
        if hasattr(config, 'audio_dim') and config.audio_dim > 0:
            self.encoders['audio'] = AudioEncoder(in_features=config.audio_dim, model_dim=self.model_dim)
            
        # Text is often already embedded or handled by the base model, 
        # but we can add a projection if needed.
        self.fuser = ModalityFuser(model_dim=self.model_dim)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = {}
        for key, x in inputs.items():
            if key in self.encoders:
                encoded[key] = self.encoders[key](x)
            elif key == 'text' and x.dim() == 3 and x.size(-1) == self.model_dim:
                # Text is already projected
                encoded[key] = x
            elif key == 'text' and x.dim() == 2:
                # Text is token IDs, handle elsewhere or add Embedding layer here
                # For now, we assume text is handled by the base model if not projected
                pass
                
        if not encoded:
            # Fallback if no modalities were encoded (e.g. only raw text IDs)
            return None
            
        return self.fuser(encoded)
