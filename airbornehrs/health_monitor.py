
import torch
import torch.nn as nn
import logging
from typing import Dict, Any

class NeuralHealthMonitor:
    """
    [V9.0] Autonomic Self-Repair Module.
    Monitors the 'vital signs' of the neural network:
    - Dead Neurons (zero gradients)
    - Gradient Explosions
    - Mode Collapse
    - Weight Saturation
    """
    def __init__(self, model: nn.Module, dead_threshold: float = 1e-8, explosion_threshold: float = 1e2):
        self.model = model
        self.dead_threshold = dead_threshold
        self.explosion_threshold = explosion_threshold
        self.logger = logging.getLogger("NeuralHealthMonitor")
        self.health_history = []

    def check_vital_signs(self) -> Dict[str, str]:
        """
        Analyzes gradients and weights to detect issues.
        Returns: Dict[parameter_name, status]
        """
        report = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    status = "IDLE"
                else:
                    grad_mean = param.grad.abs().mean().item()
                    if grad_mean < self.dead_threshold:
                        status = "DEAD"
                    elif grad_mean > self.explosion_threshold:
                        status = "CRITICAL"
                    else:
                        # Check for weight saturation
                        weight_std = param.data.std().item()
                        if weight_std < 1e-4:
                            status = "SATURATED"
                        else:
                            status = "HEALTHY"
                report[name] = status
        return report

    def autonomic_repair(self, report: Dict[str, str]):
        """
        Intervenes and repairs layers with poor health status.
        Uses selective re-initialization and gradient scaling.
        """
        repairs_made = 0
        for name, status in report.items():
            if status in ["DEAD", "SATURATED"]:
                self.logger.info(f"[REPAIR] Autonomic Intervention: Repairing {status} layer '{name}'")
                
                # Find the module containing this parameter
                module_path = name.split('.')[:-1]
                target_mod = self.model
                for part in module_path:
                    if hasattr(target_mod, part):
                        target_mod = getattr(target_mod, part)
                    else:
                        break # Could be a ModuleList or similar
                
                # Re-initialize the specific module
                if hasattr(target_mod, 'reset_parameters'):
                    target_mod.reset_parameters()
                elif isinstance(target_mod, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_normal_(target_mod.weight)
                    if hasattr(target_mod, 'bias') and target_mod.bias is not None:
                        nn.init.zeros_(target_mod.bias)
                repairs_made += 1
                
            elif status == "CRITICAL":
                self.logger.warning(f"[WARNING] Autonomic Intervention: Scaling gradients for exploding layer '{name}'")
                if self.model.named_parameters():
                    param = dict(self.model.named_parameters())[name]
                    if param.grad is not None:
                        param.grad.data.copy_(torch.clamp(param.grad.data, -1.0, 1.0))
                repairs_made += 1
                
        return repairs_made
