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
        if len(feedback_buffer.buffer) < 10:
            self.logger.warning("⚠️ EWC: Buffer too empty to consolidate.")
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
            
            # We use the log_likelihood (or MSE equivalent) for Fisher
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalize
        for name in fisher:
            fisher[name] /= len(samples)
            
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