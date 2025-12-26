import torch
import torch.nn as nn
import logging
from typing import Dict, Any


class AdapterBank:
    """
    Parameter-efficient FiLM-style adapters per tracked layer.
    Each adapter stores a scale and shift tensor that is broadcast-applied
    to the layer's output during forward hooks.
    """
    def __init__(self, num_layers: int = 0, device: torch.device = None):
        self.logger = logging.getLogger('AdapterBank')
        self.device = device
        self.num_layers = num_layers
        # Each adapter entry may be one of two styles:
        # - FiLM scalar/vector: {'type':'film','scale':Parameter,'shift':Parameter}
        # - Bottleneck residual: {'type':'bneck','Wdown':Parameter,'Wup':Parameter,'bdown':Parameter|'bup':Parameter}
        self.adapters: Dict[int, Dict[str, Any]] = {}
        for i in range(num_layers):
            # default to a scalar FiLM; will be resized when out-dim is known
            self.adapters[i] = {
                'type': 'film',
                'scale': nn.Parameter(torch.ones(1, device=self.device, dtype=torch.float32)),
                'shift': nn.Parameter(torch.zeros(1, device=self.device, dtype=torch.float32))
            }

    def ensure_index(self, idx: int, out_dim: int = None):
        """Ensure adapter exists for idx. If out_dim provided, allocate vectors or bottleneck sized to out_dim.
        We prefer a small bottleneck residual adapter for expressive per-layer corrections when out_dim is known.
        """
        if idx not in self.adapters:
            if out_dim is None or out_dim <= 4:
                # keep simple film adapter
                self.adapters[idx] = {
                    'type': 'film',
                    'scale': nn.Parameter(torch.ones(1, device=self.device, dtype=torch.float32)),
                    'shift': nn.Parameter(torch.zeros(1, device=self.device, dtype=torch.float32))
                }
            else:
                # allocate a bottleneck adapter: W_down (out_dim, r), W_up (r, out_dim)
                r = max(4, min(32, out_dim // 4))
                Wdown = nn.Parameter(torch.randn(out_dim, r, device=self.device) * (0.02))
                Wup = nn.Parameter(torch.randn(r, out_dim, device=self.device) * (0.02))
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
        else:
            # resize/convert if necessary
            if out_dim is not None:
                entry = self.adapters[idx]
                if entry.get('type') == 'film':
                    # promote to vector or bottleneck depending on size
                    if out_dim <= 4:
                        entry['scale'] = nn.Parameter(torch.ones(1, device=self.device))
                        entry['shift'] = nn.Parameter(torch.zeros(1, device=self.device))
                    else:
                        r = max(4, min(32, out_dim // 4))
                        Wdown = nn.Parameter(torch.randn(out_dim, r, device=self.device) * (0.02))
                        Wup = nn.Parameter(torch.randn(r, out_dim, device=self.device) * (0.02))
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
                elif entry.get('type') == 'bneck':
                    if entry.get('out_dim', None) != out_dim:
                        r = max(4, min(32, out_dim // 4))
                        Wdown = nn.Parameter(torch.randn(out_dim, r, device=self.device) * (0.02))
                        Wup = nn.Parameter(torch.randn(r, out_dim, device=self.device) * (0.02))
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
        if idx not in self.adapters:
            return activation

        entry = self.adapters[idx]
        try:
            if entry.get('type') == 'film':
                scale = entry['scale']
                shift = entry['shift']
                if activation.dim() >= 2:
                    shape = [1, -1] + [1] * (activation.dim() - 2)
                    s = scale.view(*shape)
                    t = shift.view(*shape)
                    return activation * s + t
                else:
                    return activation * scale + shift

            elif entry.get('type') == 'bneck':
                Wdown = entry['Wdown']
                Wup = entry['Wup']
                bdown = entry.get('bdown', None)
                bup = entry.get('bup', None)
                out_dim = entry.get('out_dim', None)

                # Handle 2D activations: (batch, features)
                if activation.dim() == 2:
                    # residual = (activation @ Wdown) @ Wup
                    z = activation @ Wdown
                    if bdown is not None:
                        z = z + bdown
                    res = z @ Wup
                    if bup is not None:
                        res = res + bup
                    return activation + res

                # Handle conv-style activations: (batch, channels, H, W) or similar
                elif activation.dim() >= 3:
                    # move channels to last dim for matmul: (batch, H*W, C)
                    b = activation.size(0)
                    c = activation.size(1)
                    rest = activation.shape[2:]
                    spatial = 1
                    for d in rest:
                        spatial *= d
                    x = activation.reshape(b, c, spatial).permute(0, 2, 1)  # (b, spatial, c)
                    z = x @ Wdown  # (b, spatial, r)
                    if bdown is not None:
                        z = z + bdown
                    res = z @ Wup  # (b, spatial, c)
                    if bup is not None:
                        res = res + bup
                    res = res.permute(0, 2, 1).reshape(b, c, *rest)
                    return activation + res

        except Exception:
            return activation

    def set_adapter(self, idx: int, scale: torch.Tensor, shift: torch.Tensor):
        self.adapters[idx] = {
            'scale': nn.Parameter(scale.to(self.device).detach()),
            'shift': nn.Parameter(shift.to(self.device).detach())
        }

    def parameters(self):
        """Return an iterator over adapter parameters for optimizers."""
        for v in self.adapters.values():
            if v.get('type') == 'film':
                if 'scale' in v and isinstance(v['scale'], nn.Parameter):
                    yield v['scale']
                if 'shift' in v and isinstance(v['shift'], nn.Parameter):
                    yield v['shift']
            elif v.get('type') == 'bneck':
                for param_name in ['Wdown', 'Wup', 'bdown', 'bup']:
                    if param_name in v and isinstance(v[param_name], nn.Parameter):
                        yield v[param_name]

    def get_adapter(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.adapters.get(idx, None)

    def save(self, path: str):
        payload = {str(k): {'scale': v['scale'].cpu(), 'shift': v['shift'].cpu()} for k, v in self.adapters.items()}
        torch.save(payload, path)
        self.logger.info(f"AdapterBank saved to {path}")

    def load(self, path: str):
        payload = torch.load(path, map_location=self.device)
        for k, v in payload.items():
            idx = int(k)
            self.adapters[idx] = {'scale': nn.Parameter(v['scale'].to(self.device)), 'shift': nn.Parameter(v['shift'].to(self.device))}
        self.logger.info(f"AdapterBank loaded from {path}")

    def list(self):
        return list(self.adapters.keys())
