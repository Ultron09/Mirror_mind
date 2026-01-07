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
        # x: [batch_size, input_dim] or [batch_size, seq, input_dim]
        if x.dim() == 3:
            # SOTA Sequence Pooling: [B, S, D] -> [B, D]
            x_flat = x.mean(dim=1)
        elif x.dim() > 3:
            # Flatten multi-dim inputs (Images, etc)
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x
            
        # Verify alignment with gate input dimension
        if x_flat.size(-1) != self.gate.in_features:
            # Fallback: Force resize if possible, or raise clear error
            if x_flat.numel() % self.gate.in_features == 0:
                x_flat = x_flat.view(x.size(0), self.gate.in_features)
            else:
                raise ValueError(f"SOTA MoE Shape Mismatch: Got {x_flat.shape[1]}, expected {self.gate.in_features}. "
                                 f"Original input: {x.shape}")

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
        
        # [V9.0] Let GatingNetwork handle Pooling/Flattening
        weights, indices = self.gate(x) # weights: [B, k], indices: [B, k]
        
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
            final_output = final_output.index_add(0, batch_idx, expert_out * selected_weights)
            
        return final_output, indices

class HierarchicalMoE(nn.Module):
    """
    [V9.0] Hierarchical Mixture of Experts.
    Uses tree-based routing for multi-domain specialization.
    Level 1 Router -> Domain Clusters -> Level 2 Experts.
    """
    def __init__(self, base_model, input_dim, num_domains=2, experts_per_domain=2, top_k=1):
        super().__init__()
        self.num_domains = num_domains
        self.experts_per_domain = experts_per_domain
        self.top_k = top_k
        
        # Level 1 Router: Selects which Domain Cluster to use
        self.domain_router = GatingNetwork(input_dim, num_domains, top_k=1)
        
        # Level 2: Domain Clusters (each is a SparseMoE)
        self.domains = nn.ModuleList([
            SparseMoE(base_model, input_dim, num_experts=experts_per_domain, top_k=top_k)
            for _ in range(num_domains)
        ])
    
    def forward(self, x):
        """
        Hierarchical Routing:
        1. Select Domain Cluster (Level 1)
        2. Within Domain, Select Experts (Level 2)
        """
        # [V9.0] Let GatingNetwork handle Pooling/Flattening
        # 1. Level 1 Gating
        domain_weights, domain_indices = self.domain_router(x) # [B, 1]
        
        batch_size = x.size(0)
        final_output = None
        
        # 2. Sequential Dispatch to Domain Clusters
        for i in range(self.num_domains):
            # Mask for samples belonging to domain i
            mask = (domain_indices == i).any(dim=1)
            batch_idx = torch.where(mask)[0]
            
            if len(batch_idx) == 0:
                continue
                
            selected_inputs = x[batch_idx]
            
            # 3. Level 2 Gating happens inside the SparseMoE domain
            domain_out, _ = self.domains[i](selected_inputs)
            
            if final_output is None:
                out_shape = list(domain_out.shape)
                out_shape[0] = batch_size
                final_output = torch.zeros(out_shape, device=x.device, dtype=domain_out.dtype)
            
            # Apply Level 1 weight (importance of this domain)
            # Find the weight for domain i among the top-k (which is top-1 here)
            d_w_mask = (domain_indices[batch_idx] == i)
            # Get the weight where the index matched
            # This is simplified because top_k=1
            w = domain_weights[batch_idx, 0].view(len(batch_idx), *([1] * (domain_out.dim() - 1)))
            
            final_output = final_output.index_add(0, batch_idx, domain_out * w)
            
        return final_output, domain_indices
