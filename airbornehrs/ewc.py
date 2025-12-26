# file: airbornehrs/ewc.py (Modified)

import torch
import torch.nn as nn
import logging

class EWCHandler:
    def __init__(self, model: nn.Module, ewc_lambda: float = 0.4):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = {}       
        self.opt_param_dict = {}    
        self.logger = logging.getLogger('EWCHandler')

    def is_enabled(self):
        return len(self.fisher_dict) > 0

    def consolidate_from_buffer(self, feedback_buffer, sample_limit=128):
        """
        SOTA FIX: Calculates Fisher Information using the internal Replay Buffer.
        Triggered automatically when the model detects a domain shift.
        """
        if feedback_buffer is None or len(feedback_buffer.buffer) < 5:
            self.logger.warning("⚠️ EWC: Buffer too small to consolidate (need at least 5 samples).")
            return

        self.logger.info("🧠 EWC: SURPRISE DETECTED. Locking memories from buffer...")
        
        # 1. Save current weights as the new "Anchor"
        self.opt_param_dict = {
            n: p.clone().detach() 
            for n, p in self.model.named_parameters() 
            if p.requires_grad
        }

        # 2. Compute Fisher Information (Sensitivity)
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        # We take a random sample from recent history (representing the "Old Task")
        # We limit samples to keep this operation FAST (under 100ms ideally)
        samples = feedback_buffer.buffer[-sample_limit:] 
        
        for snapshot in samples:
            self.model.zero_grad()
            
            # Reconstruct tensors on device
            inp = snapshot.input_data.to(next(self.model.parameters()).device)
            target = snapshot.target.to(next(self.model.parameters()).device)
            
            # Forward & Backward
            output = self.model(inp)
            if hasattr(output, 'logits'): output = output.logits
            elif isinstance(output, tuple): output = output[0]
            
            # For Fisher computation, we need matching shapes
            # If target is class indices [N] but output is logits [N, C], convert to one-hot
            if target.dim() == 1 and output.dim() == 2:
                # Classification case
                num_classes = output.shape[1]
                target = torch.nn.functional.one_hot(target, num_classes=num_classes).float()
            elif target.dim() == 1 and output.dim() == 1:
                # Regression case - unsqueeze target for MSELoss
                target = target.unsqueeze(-1)
            
            # We use the log_likelihood (or MSE equivalent) for Fisher
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalize and smooth with previous estimate (EMA) to reduce noise
        alpha = 0.3
        for name in fisher:
            fisher[name] /= len(samples)
            # Smooth with previous fisher if available
            if name in self.fisher_dict:
                fisher[name] = alpha * self.fisher_dict[name] + (1.0 - alpha) * fisher[name]
            # Clamp to avoid zero or extreme values
            fisher[name] = fisher[name].clamp(min=1e-8, max=1e9)

        self.fisher_dict = fisher
        self.model.train() # Resume training mode
        self.logger.info(f"🔒 EWC: Consolidation Complete. Protected {len(fisher)} layers.")

    def compute_penalty(self):
        loss = 0
        if not self.is_enabled(): return 0.0
            
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                opt_param = self.opt_param_dict[name]
                loss += (fisher * (param - opt_param).pow(2)).sum()
        
        return loss * (self.ewc_lambda / 2)

    # unified consolidate wrapper for compatibility
    def consolidate(self, feedback_buffer=None, **kwargs):
        return self.consolidate_from_buffer(feedback_buffer)
    
    # Insert this method inside the EWCHandler class in ewc.py

    def lock_for_ttt(self, strength: float = 1000.0):
        """
        Rapidly locks current weights as the anchor for Test-Time Training (TTT).
        Sets a uniform 'stiffness' (Fisher Information) for all parameters.
        
        Args:
            strength: The stiffness of the tether. Higher = less forgetting.
        """
        self.logger.info(f"⚓ EWC: Tethering weights for TTT (Strength: {strength})...")
        
        # 1. Save current weights as the Anchor
        self.opt_param_dict = {
            n: p.clone().detach() 
            for n, p in self.model.named_parameters() 
            if p.requires_grad
        }

        # 2. Set Uniform Fisher Information (The Tether)
        # Instead of calculating sensitivity from data, we assume ALL weights 
        # are equally important to keep (Isotropic Gaussian Prior).
        self.fisher_dict = {
            n: torch.full_like(p, strength) 
            for n, p in self.model.named_parameters() 
            if p.requires_grad
        }
        
        self.logger.info("🔒 EWC: Tether engaged.")

    # --- Task Memory Management ---
    def _task_dir(self):
        from pathlib import Path
        d = Path.cwd() / "checkpoints" / "task_memories"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_task_memory(self, name: str = None, data_loader=None, adapters=None, fingerprint=None):
        """
        Save current anchor (opt_param_dict) and fisher to disk as a named task memory.

        Args:
            name: Friendly name for the task. If None, timestamp is used.
            data_loader: Optional iterable of (input, target) pairs to recompute Fisher before saving.
        """
        from pathlib import Path
        if name is None:
            import datetime
            name = datetime.datetime.now().strftime("task_%Y%m%d_%H%M%S")

        # Optionally recompute fisher from provided data_loader
        if data_loader is not None:
            try:
                # Compute fisher over provided loader (simple loop)
                self.model.eval()
                fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
                n_samples = 0
                for batch in data_loader:
                    inp, target = batch
                    inp = inp.to(next(self.model.parameters()).device)
                    target = target.to(next(self.model.parameters()).device)
                    self.model.zero_grad()
                    output = self.model(inp)
                    if hasattr(output, 'logits'): output = output.logits
                    elif isinstance(output, tuple): output = output[0]
                    loss = torch.nn.functional.mse_loss(output, target)
                    loss.backward()
                    for name_p, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            fisher[name_p] += param.grad.data ** 2
                    n_samples += 1
                if n_samples > 0:
                    for k in fisher: fisher[k] /= max(1, n_samples)
                    # smooth and clamp
                    alpha = 0.3
                    for k in fisher:
                        if k in self.fisher_dict:
                            fisher[k] = alpha * self.fisher_dict[k] + (1.0 - alpha) * fisher[k]
                        fisher[k] = fisher[k].clamp(min=1e-8, max=1e9)
                    self.fisher_dict = fisher
                self.model.train()
            except Exception as e:
                self.logger.warning(f"Failed to recompute fisher from loader: {e}")

        # Ensure we have anchor params
        if not self.opt_param_dict:
            self.opt_param_dict = {
                n: p.clone().detach()
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }

        # Prepare payload (move to CPU for portability)
        payload = {
            'opt_param_dict': {k: v.cpu() for k, v in self.opt_param_dict.items()},
            'fisher_dict': {k: v.cpu() for k, v in self.fisher_dict.items()} if self.fisher_dict else {},
            'meta': {
                'model': type(self.model).__name__,
                'timestamp': __import__('datetime').datetime.now().isoformat()
            }
        }

        # Attach adapters dataframe if provided (AdapterBank instance expected)
        if adapters is not None:
            try:
                adapters_payload = {str(k): {'scale': v['scale'].cpu(), 'shift': v['shift'].cpu()} for k, v in adapters.adapters.items()}
                payload['adapters'] = adapters_payload
            except Exception as e:
                self.logger.warning(f"Failed to include adapters in task memory: {e}")

        if fingerprint is not None:
            try:
                payload['meta']['fingerprint'] = fingerprint.cpu().numpy().tolist()
            except Exception:
                payload['meta']['fingerprint'] = None

        path = self._task_dir() / f"{name}.pt"
        torch.save(payload, path)
        self.logger.info(f"💾 EWC: Task memory saved: {path}")
        return str(path)

    def list_task_memories(self):
        from pathlib import Path
        d = self._task_dir()
        return [p.name for p in d.glob('*.pt')]

    def load_task_memory(self, path_or_name: str):
        """Load a saved task memory (by filename or path) and set it as current anchor/fisher."""
        from pathlib import Path
        p = Path(path_or_name)
        if not p.exists():
            # try in task dir
            p = self._task_dir() / path_or_name
            if not p.exists():
                raise FileNotFoundError(f"Task memory not found: {path_or_name}")
        payload = torch.load(p, map_location='cpu')
        opt = payload.get('opt_param_dict', {})
        fish = payload.get('fisher_dict', {})

        # Move to model device
        device = next(self.model.parameters()).device
        self.opt_param_dict = {k: v.to(device) for k, v in opt.items()}
        self.fisher_dict = {k: v.to(device) for k, v in fish.items()}
        self.logger.info(f"🔁 EWC: Task memory loaded: {p}")
        return payload

    def delete_task_memory(self, name_or_path: str):
        from pathlib import Path
        p = Path(name_or_path)
        if not p.exists():
            p = self._task_dir() / name_or_path
        if p.exists():
            p.unlink()
            self.logger.info(f"🗑️ EWC: Deleted task memory: {p}")
            return True
        return False

    def apply_task_anchor(self, name_or_path: str, blend: float = 1.0):
        """Apply a saved anchor to current model. If blend < 1.0, interpolate with current params."""
        if not self.load_task_memory(name_or_path):
            return False

        # In-place interpolation
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.opt_param_dict:
                    anchor = self.opt_param_dict[name].to(param.device)
                    if blend >= 1.0:
                        param.data.copy_(anchor)
                    else:
                        param.data.mul_(1.0 - blend).add_(anchor * blend)
        self.logger.info(f"⚖️ EWC: Applied task anchor {name_or_path} (blend={blend})")
        return True


class SIHandler:
    """
    Synaptic Intelligence (SI) handler for online importance estimation.
    Tracks a path-integral style importance accumulator `omega_accum` (s)
    and computes final importance `omega` at consolidation.
    """
    def __init__(self, model: nn.Module, si_lambda: float = 1.0, xi: float = 1e-3):
        self.model = model
        self.si_lambda = si_lambda
        self.xi = xi
        self.logger = logging.getLogger('SIHandler')

        # per-parameter accumulators (same shape as params)
        self.omega_accum = {n: torch.zeros_like(p).detach() for n, p in model.named_parameters() if p.requires_grad}
        # final importance
        self.omega = {n: torch.zeros_like(p).detach() for n, p in model.named_parameters() if p.requires_grad}
        # anchor params (theta^*) at last consolidation
        self.anchor = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    def is_enabled(self):
        return any((v.abs().sum().item() > 0 for v in self.omega.values()))

    def before_step_snapshot(self):
        """Return a dict of current param tensors used to compute deltas after step."""
        return {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def accumulate_path(self, param_before: dict):
        """Call after optimizer.step(): accumulate s_i += - g_i * delta_theta_i element-wise."""
        try:
            for name, p in self.model.named_parameters():
                if not p.requires_grad: continue
                if p.grad is None: continue
                if name not in param_before: continue
                old = param_before[name]
                delta = (p.data - old).detach()
                # gradient may be on .grad.data
                g = p.grad.data.detach()
                # accumulate element-wise: s += - g * delta
                try:
                    self.omega_accum[name] += (-g * delta)
                except Exception:
                    # fallback to scalar accumulation
                    try:
                        self.omega_accum[name] += (-g * delta).clone()
                    except Exception:
                        pass
        except Exception as e:
            self.logger.debug(f"SI accumulate failed: {e}")

    def consolidate(self, feedback_buffer=None, **kwargs):
        """Compute final omega from accumulated s and anchor distances.
        omega = s / ((theta - theta_anchor)^2 + xi)
        After consolidation, reset accumulators and set new anchors.
        """
        self.logger.info("🧠 SI: Consolidation triggered. Computing Omega from path integrals...")
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not p.requires_grad: continue
                s = self.omega_accum.get(name)
                if s is None:
                    s = torch.zeros_like(p)
                anchor = self.anchor.get(name, p.clone().detach())
                denom = (p.data - anchor).pow(2) + self.xi
                # element-wise division
                try:
                    new_omega = s / denom
                except Exception:
                    new_omega = torch.zeros_like(p)
                # smooth / clip for stability
                self.omega[name] = new_omega.clamp(min=0.0, max=1e6)
                # reset accumulator and set new anchor
                self.omega_accum[name] = torch.zeros_like(p)
                self.anchor[name] = p.data.clone().detach()

        self.logger.info(f"🔒 SI: Consolidation complete. Protected {len(self.omega)} parameter tensors.")

    def compute_penalty(self):
        loss = 0.0
        for name, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if name in self.omega:
                anchor = self.anchor.get(name)
                if anchor is None: continue
                loss += (self.omega[name] * (p - anchor).pow(2)).sum()
        return loss * (self.si_lambda / 2.0)

    # task memory helpers for compatibility (store omega and anchor)
    def save_task_memory(self, name: str = None, data_loader=None, adapters=None, fingerprint=None):
        from pathlib import Path
        if name is None:
            import datetime
            name = datetime.datetime.now().strftime("si_task_%Y%m%d_%H%M%S")
        payload = {
            'anchor': {k: v.cpu() for k, v in self.anchor.items()},
            'omega': {k: v.cpu() for k, v in self.omega.items()},
            'meta': {'timestamp': __import__('datetime').datetime.now().isoformat(), 'model': type(self.model).__name__}
        }
        p = Path.cwd() / 'checkpoints' / 'task_memories' / f"{name}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, p)
        self.logger.info(f"💾 SI: Task memory saved: {p}")
        return str(p)

    def load_task_memory(self, path_or_name: str):
        from pathlib import Path
        p = Path(path_or_name)
        if not p.exists():
            p = Path.cwd() / 'checkpoints' / 'task_memories' / path_or_name
            if not p.exists():
                raise FileNotFoundError(path_or_name)
        payload = torch.load(p, map_location=next(self.model.parameters()).device)
        self.anchor = {k: v.to(next(self.model.parameters()).device) for k, v in payload.get('anchor', {}).items()}
        self.omega = {k: v.to(next(self.model.parameters()).device) for k, v in payload.get('omega', {}).items()}
        self.logger.info(f"🔁 SI: Task memory loaded: {p}")
        return payload

    def delete_task_memory(self, name_or_path: str):
        from pathlib import Path
        p = Path(name_or_path)
        if not p.exists():
            p = Path.cwd() / 'checkpoints' / 'task_memories' / name_or_path
        if p.exists():
            p.unlink()
            self.logger.info(f"🗑️ SI: Deleted task memory: {p}")
            return True
        return False