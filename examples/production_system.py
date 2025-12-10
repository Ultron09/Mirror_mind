"""
MirrorMind Enterprise Dashboard - EXTREME EDITION (V3.0)
========================================================
A production-grade interface for Adaptive Meta-Learning.

UPGRADES (V3.0):
- 🧠 MODEL: Upgraded to GPT-2 Medium (355M) for real coherence.
- 🛡️ SAFETY: Added Gradient Clipping to prevent "brain damage".
- 🔬 VISUALS: Integrated 3D Neural Topology Scanner.

Features:
- 💥 User Portal: Interact with the self-learning model
- 🏢 Enterprise Portal: Monitor health, plasticity, and drift
- 🧠 Real-time Introspection: Visualize model uncertainty
"""

# Suppress TensorFlow/System warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np
from datetime import datetime
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer

# Framework Imports
from airbornehrs import (
    AdaptiveFramework, 
    AdaptiveFrameworkConfig, 
    ProductionAdapter, 
    InferenceMode,
    MetaController
)

# ==================== 1. CORE SYSTEM SETUP ====================

st.set_page_config(
    page_title="MirrorMind Enterprise",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .status-healthy { background-color: #1b5e20; color: #a5d6a7; }
    .status-warning { background-color: #f57f17; color: #fff9c4; }
    .status-critical { background-color: #b71c1c; color: #ffcdd2; }
</style>
""", unsafe_allow_html=True)

# ==================== 2. ADAPTIVE MODEL WRAPPER ====================

class AdaptiveModelWrapper(nn.Module):
    """
    Wraps HuggingFace models (GPT-2 family) for introspection.
    Automatically detects architecture to find the last transformer block.
    """
    def __init__(self, model_name='gpt2-medium', embed_dim=256):
        super().__init__()
        # Load the heavier, smarter model
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Auto-detect hidden size (768 for small, 1024 for medium, 1280 for large)
        self.hidden_dim = self.backbone.config.n_embd
        self.embed_dim = embed_dim
        
        # PLASTICITY CONTROL: Freeze 90% of the brain, keep top 10% fluid
        # This prevents "catastrophic forgetting" (gibberish output)
        total_layers = len(self.backbone.transformer.h)
        freeze_until = int(total_layers * 0.9)  # Only train last 10%
        
        for i, param in enumerate(self.backbone.transformer.parameters()):
            param.requires_grad = False
            
        # Unfreeze the last few blocks for high-level reasoning adaptation
        for block in self.backbone.transformer.h[freeze_until:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Adapter Projection (The "Bridge" to our Framework)
        self.projection = nn.Linear(self.hidden_dim, self.embed_dim)
        
        # Uncertainty Head (The "Consciousness" Probe)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, return_internals=False):
        """Forward pass with Introspection"""
        # Expects Input IDs (LongTensor)
        outputs = self.backbone(input_ids, output_hidden_states=True)
        
        # Grab the last hidden state
        hidden_states = outputs.hidden_states[-1] 
        
        # Project to framework space
        output = self.projection(hidden_states)
        
        # Measure Uncertainty
        log_var = self.uncertainty_head(hidden_states)
        
        if return_internals:
            internals = {
                'embeddings': hidden_states,
                'logits': outputs.logits,
                'introspection': log_var
            }
            return output, log_var, internals
        
        return output, log_var
    
    def generate_text(self, input_ids, max_new_tokens=50):
        """Generate response for user"""
        with torch.no_grad():
            attention_mask = torch.ones_like(input_ids)
            output_tokens = self.backbone.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7, # Lower temp = more coherent
                top_p=0.9
            )
        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)


# ==================== 3. FRAMEWORK CONFIGURATION ====================

class GPTAdaptiveFramework(AdaptiveFramework):
    def __init__(self, config: AdaptiveFrameworkConfig, device=None, model_name='gpt2-medium'):
        self.model_name = model_name
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize the wrapper
        self.model = AdaptiveModelWrapper(model_name, config.model_dim).to(device)
        
        from airbornehrs.core import PerformanceMonitor, FeedbackBuffer
        self.monitor = PerformanceMonitor(self.model, config, device)
        self.feedback_buffer = FeedbackBuffer(config, device)
        
        # Optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
        
        self.loss_history = deque(maxlen=config.evaluation_frequency)
        self.step_count = 0
        self.logger.info(f"Framework initialized with {model_name}")
    
    def get_tokenizer(self):
        return self.model.tokenizer


# ==================== 4. SYSTEM INITIALIZATION ====================

@st.cache_resource
def load_system():
    """Initializes the upgraded AI System"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration for "GPT-2 Medium"
    config = AdaptiveFrameworkConfig(
        model_dim=256,
        num_layers=6, # Deeper meta-learning
        num_heads=8,
        
        # PRECISION TUNING:
        # 5e-5 is the "Goldilocks" zone.
        # High enough to learn "Elonville", Low enough to avoid "Euphrates River".
        learning_rate=5e-5, 
        
        evaluation_frequency=5
    )
    
    # Select Model: 'gpt2-medium' (Recommended) or 'gpt2-large' (Extreme)
    # If your GPU has >8GB VRAM, change to 'gpt2-large'
    framework = GPTAdaptiveFramework(config, device=device, model_name='gpt2-medium')
    
    adapter = ProductionAdapter(framework, inference_mode=InferenceMode.ONLINE)
    
    # Auto-Patch
    if not hasattr(adapter, 'meta_controller'):
        if adapter.inference_mode != InferenceMode.STATIC:
            adapter.meta_controller = MetaController(framework)
        else:
            adapter.meta_controller = None
    
    return adapter, framework.get_tokenizer()

# Session State
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Timestamp', 'Loss', 'Uncertainty', 'Latency', 'LearningRate', 'Plasticity'])
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

adapter, tokenizer = load_system()

# ==================== 5. PROCESSING & 3D MATH ====================

def process_interaction(prompt):
    """Run inference and track neural plasticity"""
    start_time = time.time()
    device = adapter.framework.device
    
    # 1. Input Processing
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs.input_ids.to(device)
    
    # 2. User Response
    try:
        response = adapter.framework.model.generate_text(input_ids, max_new_tokens=60)
    except Exception as e:
        response = f"[Error: {str(e)[:100]}]"

    # 3. Snapshot Weights (For 3D Map)
    before_weights = {
        name: param.clone() 
        for name, param in adapter.framework.model.named_parameters() 
        if param.requires_grad and "weight" in name
    }

    # 4. Training Cycle
    metrics = {'loss': 0.0, 'uncertainty_mean': 0.0, 'current_lr': 0.0}
    try:
        with torch.no_grad():
            output, _ = adapter.framework.model(input_ids)
            
        seq_len = input_ids.shape[1]
        train_input = input_ids[:, :-1] if seq_len > 1 else input_ids
        train_target = output[:, 1:, :].detach() if seq_len > 1 else output.detach()
            
        metrics = adapter.framework.train_step(train_input, train_target)
        
        if getattr(adapter, 'meta_controller', None):
            meta_metrics = adapter.meta_controller.adapt(metrics['loss'], metrics)
            metrics.update(meta_metrics)

    except Exception as e:
        print(f"Training failed: {e}")

    # 5. Calculate Plasticity (Weight Delta)
    layer_changes = {}
    total_change = 0.0
    
    for name, old_param in before_weights.items():
        new_param = dict(adapter.framework.model.named_parameters())[name]
        diff = (new_param - old_param).norm().item()
        
        # Parse layer name (e.g. "transformer.h.23.mlp")
        if "transformer.h" in name:
            try:
                # Extract layer number for Z-axis sorting
                parts = name.split('.')
                idx = [i for i, part in enumerate(parts) if part.isdigit()][0]
                short_name = f"Layer {parts[idx]}"
            except:
                short_name = "Attention Block"
        else:
            short_name = "Output Head"
        
        layer_changes[short_name] = layer_changes.get(short_name, 0.0) + diff
        total_change += diff

    # 6. Logging
    st.session_state.latest_layer_changes = layer_changes
    latency = (time.time() - start_time) * 1000
    
    new_data = {
        'Timestamp': datetime.now(),
        'Loss': metrics.get('loss', 0.0),
        'Uncertainty': abs(metrics.get('uncertainty_mean', 0.0)),
        'Latency': latency,
        'LearningRate': metrics.get('current_lr', 0.0),
        'Plasticity': total_change
    }
    
    new_row = pd.DataFrame([new_data])
    st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
    
    return response, metrics

def plot_3d_brain_activity(layer_changes):
    """Generates the Holographic Neural Map"""
    # Safe checks for empty data
    if not layer_changes: return go.Figure()
    
    layer_names = list(layer_changes.keys())
    plasticity_values = list(layer_changes.values())
    max_val = max(plasticity_values) if max(plasticity_values) > 0 else 1.0
    
    x, y, z, colors, sizes, texts = [], [], [], [], [], []
    neurons_per_layer = 40 
    
    for idx, (layer, magnitude) in enumerate(zip(layer_names, plasticity_values)):
        heat = magnitude / max_val
        radius = 3.0 + (heat * 2.0)
        angles = np.linspace(0, 2*np.pi, neurons_per_layer)
        
        for angle in angles:
            # X = Depth (Layer index * spacing)
            x.append(idx * 3)
            # Y/Z = Ring shape
            y.append((np.cos(angle) * radius) + np.random.normal(0, 0.1))
            z.append((np.sin(angle) * radius) + np.random.normal(0, 0.1))
            
            colors.append(magnitude)
            sizes.append(3 + (heat * 10))
            texts.append(f"{layer}: {magnitude:.2e}")

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=sizes, color=colors, colorscale='Plasma', opacity=0.8, colorbar=dict(title="Delta")),
        text=texts, hoverinfo='text'
    )])

    fig.update_layout(
        title="🧠 Live Neural Topology",
        scene=dict(
            xaxis=dict(title='Network Depth', showgrid=False, zeroline=False, showbackground=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# ==================== 6. UI COMPONENTS ====================

def render_sidebar():
    st.sidebar.title("🚀 MirrorMind V3")
    st.sidebar.markdown("---")
    view = st.sidebar.radio("Select Portal", ["💤 User Interface", "🏢 Enterprise Monitor"])
    st.sidebar.markdown("---")
    
    # System Status
    last_lr = 0.0
    if not st.session_state.history.empty:
        last_lr = st.session_state.history.iloc[-1]['LearningRate']
        
    st.sidebar.metric("Plasticity Rate", f"{last_lr:.2e}")
    
    if st.sidebar.button("🗑️ Reset Brain"):
        st.session_state.history = pd.DataFrame(columns=['Timestamp', 'Loss', 'Uncertainty', 'Latency', 'LearningRate', 'Plasticity'])
        st.session_state.chat_history = []
        st.rerun()
    return view

def user_portal():
    st.title("💤 User Interaction Portal")
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
    
    prompt = st.chat_input("Type your message...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("🧠 Rewiring Neural Pathways..."):
            response, metrics = process_interaction(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

def enterprise_portal():
    st.title("🏢 Enterprise Health Monitor")
    if st.session_state.history.empty:
        st.info("⏳ Waiting for data...")
        return

    last = st.session_state.history.iloc[-1]
    plasticity = last.get('Plasticity', 0.0)
    
    # Status Logic
    if plasticity > 0.1: status, color = "HIGH REWIRING", "status-warning"
    elif plasticity > 0.0001: status, color = "STABLE LEARNING", "status-healthy"
    else: status, color = "CONVERGED", "status-critical"

    st.markdown(f"""
        <div class="status-card {color}">
            <h2>{status}</h2>
            <p>Weight Delta: {plasticity:.6f}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Visuals
    if 'latest_layer_changes' in st.session_state:
        st.plotly_chart(plot_3d_brain_activity(st.session_state.latest_layer_changes), use_container_width=True)
    
    tab1, tab2 = st.tabs(["📉 Loss", "⚡ Latency"])
    with tab1: st.plotly_chart(px.line(st.session_state.history, x="Timestamp", y="Loss"), use_container_width=True)
    with tab2: st.plotly_chart(px.bar(st.session_state.history, x="Timestamp", y="Latency"), use_container_width=True)

def main():
    view = render_sidebar()
    if view == "💤 User Interface": user_portal()
    else: enterprise_portal()

if __name__ == "__main__":
    main()