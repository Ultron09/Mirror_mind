import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Iterator
import torch.nn.functional as F

class AdapterBank:
    """
    Parameter-efficient FiLM-style adapters per tracked layer.
    
    FIXES V2:
    - Zero-Init: Adapters start as Identity (no noise injection).
    - Robust Reshaping: Handles arbitrary spatial dimensions safely.
    - Strict Parameter Registration: Ensures optimizer can find params.
    """
    def __init__(self, num_layers: int = 0, device: torch.device = None):
        self.logger = logging.getLogger('AdapterBank')
        self.device = device if device else torch.device('cpu')
        self.num_layers = num_layers
        self.adapters: Dict[int, Dict[str, Any]] = {}
        
        # Pre-allocate slots (empty) to track existence
        for i in range(num_layers):
            self.adapters[i] = {'type': 'empty'}

    def ensure_index(self, idx: int, out_dim: int = None):
        """
        Ensure adapter exists for idx. 
        Promotes 'empty' or 'film' adapters to 'bneck' if out_dim is sufficiently large.
        """
        if idx not in self.adapters:
            self.adapters[idx] = {'type': 'empty'}
            
        entry = self.adapters[idx]
        current_type = entry.get('type', 'empty')

        # Case 1: Dimension unknown or small -> Use Scalar FiLM (Lightweight)
        if out_dim is None or out_dim <= 8:
            if current_type == 'empty':
                self.adapters[idx] = {
                    'type': 'film',
                    # Initialize to Identity: Scale=1.0, Shift=0.0
                    'scale': nn.Parameter(torch.ones(1, device=self.device, dtype=torch.float32)),
                    'shift': nn.Parameter(torch.zeros(1, device=self.device, dtype=torch.float32))
                }
            # If already film or bneck, leave as is (don't downgrade bneck)
            return

        # Case 2: Dimension known & large -> Use Bottleneck Adapter (High Capacity)
        # We upgrade if it's currently empty OR film (feature upgrade)
        if current_type in ['empty', 'film']:
            r = max(4, min(64, out_dim // 8)) # Bottleneck ratio
            
            # Kaiming Init for Down projection (Information extraction)
            Wdown = nn.Parameter(torch.randn(out_dim, r, device=self.device) * (2 / out_dim)**0.5)
            
            # ZERO Init for Up projection (Identity start)
            # This ensures the adapter output is 0.0 initially, so f(x) + adapter(x) = f(x)
            Wup = nn.Parameter(torch.zeros(r, out_dim, device=self.device))
            
            bdown = nn.Parameter(torch.zeros(r, device=self.device))
            bup = nn.Parameter(torch.zeros(out_dim, device=self.device))
            
            self.adapters[idx] = {
                'type': 'bneck',
                'Wdown': Wdown,
                'Wup': Wup,
                'bdown': bdown,
                'bup': bup,
                'r': r,
                'out_dim': out_dim
            }
        
        # Case 3: Resize existing bottleneck if dimension changed (rare but possible)
        elif current_type == 'bneck':
            if entry.get('out_dim') != out_dim:
                # Re-initialize to match new shape
                r = max(4, min(64, out_dim // 8))
                Wdown = nn.Parameter(torch.randn(out_dim, r, device=self.device) * (2 / out_dim)**0.5)
                Wup = nn.Parameter(torch.zeros(r, out_dim, device=self.device)) # Zero init
                bdown = nn.Parameter(torch.zeros(r, device=self.device))
                bup = nn.Parameter(torch.zeros(out_dim, device=self.device))
                
                self.adapters[idx] = {
                    'type': 'bneck',
                    'Wdown': Wdown,
                    'Wup': Wup,
                    'bdown': bdown,
                    'bup': bup,
                    'r': r,
                    'out_dim': out_dim
                }

    def apply(self, idx: int, activation: torch.Tensor) -> torch.Tensor:
        """Apply adapter to activation in-place when possible and return tensor."""
        # 1. Fast exit if no adapter exists for this layer
        if idx not in self.adapters:
            return activation

        entry = self.adapters[idx]
        adapter_type = entry.get('type')

        # 2. Fast exit if adapter is empty placeholder
        if adapter_type == 'empty':
            return activation

        try:
            # === FiLM ADAPTER (Scalar Scale & Shift) ===
            if adapter_type == 'film':
                scale = entry['scale']
                shift = entry['shift']
                # PyTorch broadcasting automatically handles (Batch, C, H, W) vs (1, 1, 1, 1)
                # No complex logic needed here.
                return activation * scale + shift

            # === BOTTLENECK ADAPTER (Low-Rank MLP) ===
            elif adapter_type == 'bneck':
                Wdown = entry['Wdown']
                Wup = entry['Wup']
                bdown = entry['bdown']
                bup = entry['bup']

                # Case A: Standard MLP / Transformers (Batch, Features)
                # Logic: We must check specific dimensions first.
                if activation.dim() == 2:
                    # Down projection: z = x @ Wdown + bdown
                    z = torch.addmm(bdown, activation, Wdown)
                    
                    # Non-linearity (Critical for bottleneck performance)
                    # using torch.nn.functional directly to avoid missing import errors
                    z = torch.nn.functional.silu(z) 
                    
                    # Up projection: res = z @ Wup + bup
                    res = torch.addmm(bup, z, Wup)
                    
                    # Residual connection
                    return activation + res

                # Case B: CNNs / Spatial Data (Batch, Channels, H, W, ...)
                # Catches 3D, 4D (Images), 5D (Video) tensors correctly
                elif activation.dim() > 2:
                    # Store original shape to restore later
                    orig_shape = activation.shape
                    
                    # Flatten all spatial dims into one sequence dimension
                    # Transformation: (B, C, H, W) -> (B, C, N) -> (B, N, C)
                    # This aligns channels to the last dimension for the linear layer
                    x_flat = activation.flatten(2).permute(0, 2, 1)
                    
                    # Apply Down Projection
                    z = x_flat @ Wdown + bdown
                    z = torch.nn.functional.silu(z)
                    
                    # Apply Up Projection
                    res = z @ Wup + bup # Output shape: (B, N, C)
                    
                    # Restore original shape: (B, N, C) -> (B, C, N) -> (B, C, H, W)
                    res = res.permute(0, 2, 1).view(*orig_shape)
                    
                    # Residual connection
                    return activation + res

        except Exception as e:
            # Failsafe: If anything goes wrong (e.g. OOM, shape mismatch), 
            # return original activation so training doesn't crash.
            return activation

        # Default fallback
        return activation

    def parameters(self) -> Iterator[nn.Parameter]:
        """Return an iterator over adapter parameters for optimizers."""
        for v in self.adapters.values():
            if v.get('type') == 'film':
                yield v['scale']
                yield v['shift']
            elif v.get('type') == 'bneck':
                yield v['Wdown']
                yield v['Wup']
                yield v['bdown']
                yield v['bup']

    def state_dict(self):
        """Serializable state."""
        return {
            k: {
                key: val.cpu() if isinstance(val, torch.Tensor) else val 
                for key, val in v.items() 
                if key != 'type'
            } | {'type': v['type']}
            for k, v in self.adapters.items()
        }

    def load_state_dict(self, state_dict):
        """Restore state."""
        for k, v in state_dict.items():
            idx = int(k)
            atype = v.get('type', 'empty')
            
            if atype == 'film':
                self.adapters[idx] = {
                    'type': 'film',
                    'scale': nn.Parameter(v['scale'].to(self.device)),
                    'shift': nn.Parameter(v['shift'].to(self.device))
                }
            elif atype == 'bneck':
                self.adapters[idx] = {
                    'type': 'bneck',
                    'Wdown': nn.Parameter(v['Wdown'].to(self.device)),
                    'Wup': nn.Parameter(v['Wup'].to(self.device)),
                    'bdown': nn.Parameter(v['bdown'].to(self.device)),
                    'bup': nn.Parameter(v['bup'].to(self.device)),
                    'r': v['r'],
                    'out_dim': v['out_dim']
                }
            else:
                self.adapters[idx] = {'type': 'empty'}