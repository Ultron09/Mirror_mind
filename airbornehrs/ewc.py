import torch
import torch.nn as nn
import copy

class EWCHandler:
    def __init__(self, model: nn.Module, ewc_lambda: float = 0.4):
        """
        Args:
            model: The model to protect.
            ewc_lambda: How hard to punish forgetting (Hyperparameter).
                        Higher = More Stability, Lower = More Plasticity.
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        
        # Storage for the "Anchor" (Old Weights) and "Importance" (Fisher Matrix)
        self.fisher_dict = {}       # Importance of each weight
        self.opt_param_dict = {}    # The values of the weights we are protecting
        self.device = next(model.parameters()).device

    def is_enabled(self):
        """Check if we have stored weights to protect."""
        return len(self.fisher_dict) > 0

    def save_task_memory(self, data_sample_iterator):
        """
        Call this AFTER finishing a task. It computes which weights are important.
        """
        self.model.eval()
        fisher = {}
        params = {}
        
        # 1. Initialize Fisher Matrix with Zeros
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
                # Store the optimal parameter value (The "Anchor")
                params[name] = param.data.clone()

        # 2. Accumulate Gradients (Sensitivity Analysis)
        print("🧠 EWC: Consolidating memory (Computing Fisher Matrix)...")
        self.model.zero_grad()
        
        # Process a few batches to estimate importance
        # (We use the log_var/uncertainty output as a proxy for likelihood sensitivity)
        for input_data, target in data_sample_iterator:
            input_data = input_data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            output, log_var, _ = self.model(input_data, return_internals=True)
            
            # We use standard loss to generate gradients
            # This tells us: "Which weights, if changed, would explode the loss?"
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2 / len(data_sample_iterator)
            
            self.model.zero_grad()

        self.fisher_dict = fisher
        self.opt_param_dict = params
        print(f"🔒 EWC: Memory Locked. {len(fisher)} layers protected.")

    def compute_penalty(self):
        """
        Calculate the EWC Loss to add to the training loop.
        Loss = (Current_Weight - Old_Weight)^2 * Importance
        """
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                # Retrieve memory
                fisher = self.fisher_dict[name]
                opt_param = self.opt_param_dict[name]
                
                # Calculate penalty: Importance * (Delta^2)
                loss += (fisher * (param - opt_param).pow(2)).sum()
        
        return loss * (self.ewc_lambda / 2)
    
    def lock_pretrained_weights(self):
        """
        Locks current weights as important without needing data.
        Assumes uniform importance (Fisher = 1.0).
        Useful for preserving pretrained initialization blindly.
        """
        self.fisher_dict = {}
        self.opt_param_dict = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store current pretrained weights
                self.opt_param_dict[name] = param.data.clone()
                # Assume moderate importance for everything
                self.fisher_dict[name] = torch.ones_like(param.data) 
        
        print("🔒 Pretrained weights locked (Uniform Importance).")