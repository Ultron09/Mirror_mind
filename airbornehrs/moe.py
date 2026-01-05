import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ExpertBlock(nn.Module):
    """
    A wrapper for any neural module to act as an expert.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

class GatingNetwork(nn.Module):
    """
    Router that decides which experts to activate.
    """
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: [batch_size, input_dim]
        # If x is not flat, flatten it for gating
        if x.dim() > 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x
            
        logits = self.gate(x_flat)
        
        # Top-k gating
        # Keep top k values, set others to -inf
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)
        
        # Softmax over top-k
        weights = F.softmax(top_k_logits, dim=1)
        
        return weights, top_k_indices

class SparseMoE(nn.Module):
    """
    The Sparse Mixture of Experts Container.
    """
    def __init__(self, base_model, input_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create experts by cloning the base model
        # We deepcopy to ensure independent weights
        self.experts = nn.ModuleList([
            ExpertBlock(copy.deepcopy(base_model)) for _ in range(num_experts)
        ])
        
        self.gate = GatingNetwork(input_dim, num_experts, top_k)

    def forward(self, x):
        # x: [batch_size, ...]
        
        # 1. Gating
        # We need a feature vector for gating. 
        # If x is image [B, C, H, W], we flatten.
        # If x is sequence [B, S, D], we might gate per token or per sequence.
        # For simplicity V7.0: Gate per sample (using flattened input).
        
        if x.dim() > 2:
            gate_input = x.view(x.size(0), -1)
        else:
            gate_input = x
            
        weights, indices = self.gate(gate_input) # weights: [B, k], indices: [B, k]
        
        batch_size = x.size(0)
        
        # Initialize output
        # We need to know the output shape of the expert.
        # We run one expert on a dummy input or the first sample to get shape?
        # Or just rely on the loop.
        
        final_output = None
        
        # 2. Dispatch & Execute
        for i in range(self.num_experts):
            # Find batch indices where expert i is selected
            mask = (indices == i) # [B, k]
            batch_idx, k_idx = torch.where(mask)
            
            if len(batch_idx) == 0:
                continue
            
            # Get inputs for this expert
            selected_inputs = x[batch_idx]
            
            # Run expert
            expert_out = self.experts[i](selected_inputs)
            
            # Initialize final_output on first valid execution
            if final_output is None:
                out_shape = list(expert_out.shape)
                out_shape[0] = batch_size
                final_output = torch.zeros(out_shape, device=x.device, dtype=expert_out.dtype)
            
            # Get weights
            selected_weights = weights[batch_idx, k_idx] # [N_selected]
            
            # Reshape weights to match output dimensions for broadcasting
            # e.g. if output is [N, 10], weights need to be [N, 1]
            view_shape = [len(batch_idx)] + [1] * (expert_out.dim() - 1)
            selected_weights = selected_weights.view(*view_shape)
            
            # Accumulate
            final_output.index_add_(0, batch_idx, expert_out * selected_weights)
            
        return final_output, indices
