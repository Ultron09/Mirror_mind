"""
MirrorMind Enterprise Dashboard - EXTREME EDITION (V3.2.1 Sync)
===============================================================
A production-grade interface for Adaptive Meta-Learning.
Updated for AirborneHRS V6.1 Compatibility.
"""

import os
import warnings
import time
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Framework Imports
from airbornehrs import (
    AdaptiveFramework, 
    AdaptiveFrameworkConfig, 
    ProductionAdapter, 
    InferenceMode
)

# ==================== 1. SYSTEM CONFIGURATION ====================

st.set_page_config(
    page_title="MirrorMind Enterprise",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 2. ADAPTIVE MODEL WRAPPER ====================

class AdaptiveModelWrapper(nn.Module):
    """
    Wraps HuggingFace models for introspection.
    """
    def __init__(self, model_name='gpt2-medium', embed_dim=256):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.hidden_dim = self.backbone.config.n_embd
        self.embed_dim = embed_dim
        
        # PLASTICITY CONTROL: Freeze 90%, Train 10%
        total_layers = len(self.backbone.transformer.h)
        freeze_until = int(total_layers * 0.9)
        
        for i, param in enumerate(self.backbone.transformer.parameters()):
            param.requires_grad = False
            
        for block in self.backbone.transformer.h[freeze_until:]:
            for param in block.parameters():
                param.requires_grad = True
        
        self.projection = nn.Linear(self.hidden_dim, self.embed_dim)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, **kwargs):
        # NOTE: AdaptiveFramework calls this. Return format must be handled carefully.
        outputs = self.backbone(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1] 
        output_proj = self.projection(hidden_states)
        log_var = self.uncertainty_head(hidden_states)
        
        # We return the projected output for the framework to use in loss
        # The framework's own IntrospectionEngine will also run parallel to this.
        return output_proj, log_var
    
    def generate_text(self, input_ids, max_new_tokens=50):
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

class GPTAdaptiveFramework(AdaptiveFramework):
    def __init__(self, config, device=None, model_name='gpt2-medium'):
        # 1. Initialize the Base Model
        model = AdaptiveModelWrapper(model_name, config.model_dim)
        
        # 2. Call Super (Initializes Optimizer, MetaController, etc.)
        super().__init__(model, config, device)
        
        self.model_name = model_name

    def get_tokenizer(self):
        return self.model.tokenizer

# ==================== 3. INITIALIZATION ====================

@st.cache_resource
def load_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = AdaptiveFrameworkConfig(
        model_dim=256,
        num_layers=6,
        learning_rate=5e-5, 
        evaluation_frequency=5,
        device=str(device),
        compile_model=False # Disable compile for Streamlit stability
    )
    
    framework = GPTAdaptiveFramework(config, device=device, model_name='gpt2-medium')
    
    # Initialize Adapter with ONLINE learning enabled
    # This enables the Reptile MetaController
    adapter = ProductionAdapter(framework, inference_mode=InferenceMode.ONLINE, enable_meta_learning=True)
    
    return adapter, framework.get_tokenizer()

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Timestamp', 'Loss', 'Uncertainty', 'Latency', 'LearningRate', 'Plasticity'])
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

adapter, tokenizer = load_system()

# ==================== 4. PROCESSING LOGIC ====================

def process_interaction(prompt):
    """Run inference, generate text, and trigger online learning."""
    start_time = time.time()
    device = adapter.framework.device
    
    # 1. Prepare Inputs
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs.input_ids.to(device)
    
    # 2. Generate Response (User Facing)
    try:
        response = adapter.framework.model.generate_text(input_ids, max_new_tokens=60)
    except Exception as e:
        response = f"[Error: {str(e)}]"

    # 3. Snapshot Weights for 3D Viz
    before_weights = {n: p.clone() for n, p in adapter.framework.model.named_parameters() if p.requires_grad}

    # 4. Online Learning Step (The Magic)
    metrics = {}
    try:
        # Self-supervised target: Predict next tokens based on current understanding
        # Note: We use the adapter to ensure thread safety
        with torch.no_grad():
            target_output, _ = adapter.framework.model(input_ids)
            target = target_output.detach()

        # Trigger update (This calls MetaController.adapt -> Reptile)
        adapter.predict(input_ids, update=True, target=target)
        
        metrics = adapter.get_metrics()
        
    except Exception as e:
        print(f"Learning step failed: {e}")

    # 5. Calculate Plasticity
    layer_changes = {}
    total_change = 0.0
    
    # Only calculate if we have weights to compare
    if before_weights:
        current_weights = dict(adapter.framework.model.named_parameters())
        for name, old_p in before_weights.items():
            if name in current_weights:
                new_p = current_weights[name]
                diff = (new_p - old_p).norm().item()
                short_name = name.split('.')[-2] if len(name.split('.')) > 1 else name
                layer_changes[short_name] = layer_changes.get(short_name, 0.0) + diff
                total_change += diff

    # 6. Update History
    st.session_state.latest_layer_changes = layer_changes
    latency = (time.time() - start_time) * 1000
    
    new_data = {
        'Timestamp': datetime.now(),
        'Loss': metrics.get('loss', 0.0),
        'Uncertainty': abs(metrics.get('uncertainty_mean', 0.0)),
        'Latency': latency,
        'LearningRate': metrics.get('learning_rate', metrics.get('current_lr', 0.0)),
        'Plasticity': total_change
    }
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_data])], ignore_index=True)
    
    return response, metrics

# ==================== 5. UI COMPONENTS ====================

def plot_3d_brain(layer_changes):
    if not layer_changes: return go.Figure()
    names, vals = list(layer_changes.keys()), list(layer_changes.values())
    max_val = max(vals) if vals else 1.0
    
    x, y, z, colors, sizes = [], [], [], [], []
    for idx, (layer, mag) in enumerate(zip(names, vals)):
        for angle in np.linspace(0, 2*np.pi, 20):
            r = 3 + (mag/max_val * 5)
            x.append(idx * 5)
            y.append(np.cos(angle) * r)
            z.append(np.sin(angle) * r)
            colors.append(mag)
            sizes.append(5 + (mag/max_val * 15))

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=sizes, color=colors, colorscale='Viridis', opacity=0.8),
        hovertext=[f"{n}: {v:.2e}" for n,v in zip(names*20, vals*20)]
    )])
    fig.update_layout(scene=dict(xaxis_title='Depth', yaxis_visible=False, zaxis_visible=False), margin=dict(l=0, r=0, b=0, t=0))
    return fig

def main():
    view = st.sidebar.radio("View", ["💤 Chat Portal", "🏢 Monitor"])
    
    if view == "💤 Chat Portal":
        st.title("User Portal")
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input():
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("Processing..."):
                resp, _ = process_interaction(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": resp})
            st.rerun()
            
    else:
        st.title("System Monitor")
        if not st.session_state.history.empty:
            last = st.session_state.history.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Plasticity", f"{last['Plasticity']:.2e}")
            c2.metric("Adaptation Rate", f"{last['LearningRate']:.2e}")
            c3.metric("Latency", f"{last['Latency']:.0f}ms")
            
            if 'latest_layer_changes' in st.session_state:
                st.plotly_chart(plot_3d_brain(st.session_state.latest_layer_changes))
            
            st.subheader("Training Dynamics")
            st.line_chart(st.session_state.history.set_index('Timestamp')[['Loss', 'Uncertainty']])

if __name__ == "__main__":
    main()