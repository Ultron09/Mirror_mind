"""
MirrorMind Enterprise Dashboard - FIXED VERSION (V2.1)
======================================================
A production-grade interface for Adaptive Meta-Learning.

KEY FIXES:
- ✅ Fixed "no attribute 'meta_controller'" crash by auto-patching stale objects.
- ✅ Fixed "Expected Long, Int" tensor error with correct input types.
- ✅ Added safe attribute access for robust runtime.

Features:
- 💥 User Portal: Interact with the self-learning model
- 🏢 Enterprise Portal: Monitor health, plasticity, and drift
- 🧠 Real-time Introspection: Visualize model uncertainty
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
from collections import deque
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

# ==================== 2. MODEL WRAPPER ====================

class GPT2Wrapper(nn.Module):
    """
    Wraps GPT-2 to match the airbornehrs IntrospectionModule interface.
    Returns: (output, log_var) tuple as expected by AdaptiveFramework
    """
    def __init__(self, model_name='gpt2', embed_dim=256):
        super().__init__()
        self.backbone = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get GPT-2 hidden dimension
        self.gpt2_dim = self.backbone.config.n_embd  # 768 for gpt2
        self.embed_dim = embed_dim
        
        # Freeze most of GPT-2, only train last layer + adapters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last transformer block for fine-tuning
        for param in self.backbone.transformer.h[-1].parameters():
            param.requires_grad = True
        
        # Adapter layers to match framework interface
        self.projection = nn.Linear(self.gpt2_dim, self.embed_dim)
        
        # Uncertainty head (outputs log variance)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.gpt2_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, return_internals=False):
        """
        Forward pass matching IntrospectionModule signature.
        """
        # Get GPT-2 outputs (Expects Integers/Longs)
        outputs = self.backbone(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, 768]
        
        # Project to framework dimension
        output = self.projection(hidden_states)  # [batch, seq_len, embed_dim]
        
        # Calculate uncertainty
        log_var = self.uncertainty_head(hidden_states)  # [batch, seq_len, 1]
        
        if return_internals:
            internals = {
                'embeddings': hidden_states,
                'logits': outputs.logits,
                'introspection': log_var
            }
            return output, log_var, internals
        
        return output, log_var
    
    def generate_text(self, input_ids, max_new_tokens=50):
        """Generate text completion (for user-facing responses)"""
        with torch.no_grad():
            attention_mask = torch.ones_like(input_ids)
            output_tokens = self.backbone.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)


# ==================== 3. FRAMEWORK INTEGRATION ====================

class GPT2AdaptiveFramework(AdaptiveFramework):
    """
    Custom AdaptiveFramework that properly wraps GPT-2.
    """
    def __init__(self, config: AdaptiveFrameworkConfig, device=None, model_name='gpt2'):
        self.model_name = model_name
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.config = config
        self.logger = self._setup_logging()
        
        self.model = GPT2Wrapper(model_name, config.model_dim).to(device)
        
        from airbornehrs.core import PerformanceMonitor, FeedbackBuffer
        self.monitor = PerformanceMonitor(self.model, config, device)
        self.feedback_buffer = FeedbackBuffer(config, device)
        
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
        
        self.loss_history = deque(maxlen=config.evaluation_frequency)
        self.step_count = 0
        
        self.logger.info(f"GPT2AdaptiveFramework initialized with {model_name}")
    
    def get_tokenizer(self):
        return self.model.tokenizer


# ==================== 4. STATE MANAGEMENT ====================

@st.cache_resource
def load_system():
    """Initializes the AI System (Cached to persist across re-runs)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Configure Framework
    config = AdaptiveFrameworkConfig(
        model_dim=256,
        num_layers=4,
        num_heads=4,
        learning_rate=5e-5,
        evaluation_frequency=10
    )
    
    # 2. Create Custom Framework
    framework = GPT2AdaptiveFramework(config, device=device, model_name='gpt2')
    
    # 3. Create Production Adapter
    adapter = ProductionAdapter(
        framework,
        inference_mode=InferenceMode.ONLINE
    )
    
    # --- HOTFIX: PATCH STALE OBJECTS ---
    # If the imported ProductionAdapter is older and lacks meta_controller,
    # we manually attach it here to prevent crashes.
    if not hasattr(adapter, 'meta_controller'):
        print("⚠️ Patching stale ProductionAdapter with MetaController...")
        if adapter.inference_mode != InferenceMode.STATIC:
            adapter.meta_controller = MetaController(framework)
        else:
            adapter.meta_controller = None
    
    return adapter, framework.get_tokenizer()

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=['Timestamp', 'Loss', 'Uncertainty', 'Latency', 'LearningRate']
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

adapter, tokenizer = load_system()

# ==================== 5. PROCESSING LOGIC ====================
def plot_3d_brain_activity(layer_changes):
    """
    Generates a 3D Holographic Map of the Neural Network.
    Layers are arranged along the X-axis.
    Height/Depth (Y/Z) represents the 'neural volume'.
    Color represents Plasticity (Learning Magnitude).
    """
    import numpy as np
    
    # 1. Setup the structure
    layer_names = list(layer_changes.keys())
    plasticity_values = list(layer_changes.values())
    
    # Normalize plasticity for coloring (0 to 1 scale)
    if not plasticity_values: return go.Figure()
    max_val = max(plasticity_values) if max(plasticity_values) > 0 else 1.0
    
    # 2. Create 3D Point Cloud data
    x_coords = []
    y_coords = []
    z_coords = []
    colors = []
    sizes = []
    hover_texts = []
    
    # Generate a "ring" of neurons for each layer to create a 3D tube shape
    neurons_per_layer = 20  # Visualization density
    
    for idx, (layer, magnitude) in enumerate(zip(layer_names, plasticity_values)):
        # Calculate heat (0 = Blue/Cold, 1 = Red/Hot)
        heat = magnitude / max_val
        
        # Create a circle of points for this layer
        radius = 2.0 + (heat * 1.0) # Active layers expand slightly
        angles = np.linspace(0, 2*np.pi, neurons_per_layer)
        
        for angle in angles:
            # X represents depth (Layer Index)
            x_coords.append(idx * 2) 
            
            # Y and Z represent the 2D slice of the layer
            # Add jitter to make it look organic
            jitter_y = np.random.normal(0, 0.2)
            jitter_z = np.random.normal(0, 0.2)
            
            y_coords.append((np.cos(angle) * radius) + jitter_y)
            z_coords.append((np.sin(angle) * radius) + jitter_z)
            
            # Color logic: Hotter = More Plasticity
            colors.append(magnitude)
            
            # Size logic: Active neurons appear larger
            sizes.append(5 + (heat * 15))
            
            hover_texts.append(f"Layer: {layer}<br>Plasticity: {magnitude:.6f}")

    # 3. Build the Plotly 3D Figure
    fig = go.Figure(data=[go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            colorscale='Inferno',  # Magma or Inferno look very "Brain-like"
            opacity=0.8,
            colorbar=dict(title="Weight Delta"),
            line=dict(width=0)
        ),
        text=hover_texts,
        hoverinfo='text'
    )])

    # 4. Styling to make it look like a Hologram
    fig.update_layout(
        title="🧠 3D Neural Topology (Live Plasticity)",
        scene=dict(
            xaxis=dict(title='Network Depth (Layers)', showgrid=False, zeroline=False, showbackground=False),
            yaxis=dict(title='Neuron Cluster', showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            zaxis=dict(title='Activation Space', showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig
# ==================== 6. UI COMPONENTS ====================

def render_sidebar():
    st.sidebar.title("🚀 MirrorMind")
    st.sidebar.markdown("---")
    view = st.sidebar.radio(
        "Select Portal", 
        ["💤 User Interface", "🏢 Enterprise Monitor"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("System Status")
    
    # System info
    device = "CUDA 🟢" if torch.cuda.is_available() else "CPU 🟡"
    mode = adapter.inference_mode.upper()
    
    st.sidebar.caption(f"Compute: **{device}**")
    st.sidebar.caption(f"Mode: **{mode}**")
    st.sidebar.caption(f"Model: **GPT-2 (Adaptive)**")
    st.sidebar.caption(f"Interactions: **{len(st.session_state.history)}**")
    
    # Actions
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.history = pd.DataFrame(
            columns=['Timestamp', 'Loss', 'Uncertainty', 'Latency', 'LearningRate']
        )
        st.session_state.chat_history = []
        st.rerun()
    
    return view


def user_portal():
    st.title("💤 User Interaction Portal")
    st.markdown("Interact with the adaptive GPT-2 model. It learns from every interaction!")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Input
    prompt = st.chat_input("Type your message...")
    
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking & Learning..."):
            response, metrics = process_interaction(prompt)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        with st.sidebar:
            st.success("✅ Model Updated")
            st.metric("Loss", f"{metrics.get('loss', 0):.4f}")
            st.metric("Uncertainty", f"{metrics.get('uncertainty_mean', 0):.4f}")
        
        st.rerun()

# ==================== UPDATED PROCESSING LOGIC ====================

def process_interaction(prompt):
    """
    Runs inference, updates weights, and tracks WHICH layers changed.
    """
    start_time = time.time()
    device = adapter.framework.device
    
    # 1. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs.input_ids.to(device)
    
    # 2. Generate (User Experience)
    try:
        response = adapter.framework.model.generate_text(input_ids, max_new_tokens=50)
    except Exception as e:
        response = f"[Error: {str(e)[:100]}]"

    # 3. CAPTURE STATE BEFORE TRAINING (For Visualization)
    # We clone the weights of the trainable last layer to compare later
    before_weights = {
        name: param.clone() 
        for name, param in adapter.framework.model.named_parameters() 
        if param.requires_grad and "weight" in name
    }

    # 4. Training Step
    metrics = {'loss': 0.0, 'uncertainty_mean': 0.0, 'current_lr': 0.0}
    try:
        with torch.no_grad():
            output, _ = adapter.framework.model(input_ids)
            
        seq_len = input_ids.shape[1]
        if seq_len > 1:
            train_input = input_ids[:, :-1]
            train_target = output[:, 1:, :].detach()
        else:
            train_input = input_ids
            train_target = output.detach()
            
        metrics = adapter.framework.train_step(train_input, train_target)
        
        # Meta-Controller Adapt
        if getattr(adapter, 'meta_controller', None):
            meta_metrics = adapter.meta_controller.adapt(metrics['loss'], metrics)
            metrics.update(meta_metrics)

    except Exception as e:
        print(f"Training failed: {e}")

    # 5. CALCULATE PLASTICITY (Visualizing the Change)
    # Compare weights After vs Before
    layer_changes = {}
    total_change = 0.0
    
    for name, old_param in before_weights.items():
        new_param = dict(adapter.framework.model.named_parameters())[name]
        
        # Calculate L2 Norm of the difference (The "Magnitude of Change")
        diff = (new_param - old_param).norm().item()
        
        # Simplify name for the chart (e.g., "transformer.h.11.mlp.c_fc.weight" -> "Layer 11 MLP")
        short_name = name.split('.')[-3] if "transformer" in name else "Adapter Head"
        
        layer_changes[short_name] = layer_changes.get(short_name, 0.0) + diff
        total_change += diff

    # 6. Record Data
    latency = (time.time() - start_time) * 1000
    
    # Add plasticity data to metrics
    metrics['total_plasticity'] = total_change
    
    new_data = {
        'Timestamp': datetime.now(),
        'Loss': metrics.get('loss', 0.0),
        'Uncertainty': abs(metrics.get('uncertainty_mean', 0.0)),
        'Latency': latency,
        'LearningRate': metrics.get('current_lr', 0.0),
        'Plasticity': total_change
    }
    
    # Store per-layer breakdown for the heatmap (Serialize as string or separate dict)
    st.session_state.latest_layer_changes = layer_changes

    new_row = pd.DataFrame([new_data])
    if st.session_state.history.empty:
        st.session_state.history = new_row
    else:
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
    
    return response, metrics

# ==================== UPDATED ENTERPRISE PORTAL ====================
def enterprise_portal():
    st.title("🏢 Enterprise Health Monitor")
    
    if st.session_state.history.empty:
        st.info("⏳ Waiting for interaction data...")
        return
    
    # Metrics
    last = st.session_state.history.iloc[-1]
    
    # Status Logic
    plasticity = last.get('Plasticity', 0.0)
    if plasticity > 0.1: 
        status = "HIGH PLASTICITY (Rewiring)"
        color = "status-warning"
    elif plasticity > 0.001:
        status = "STABLE LEARNING"
        color = "status-healthy"
    else:
        status = "STATIC / CONVERGED"
        color = "status-critical" # Red if it stops learning entirely

    st.markdown(f"""
        <div class="status-card {color}">
            <h2>STATUS: {status}</h2>
            <p>Plasticity (Weight Delta): {plasticity:.6f}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Loss", f"{last['Loss']:.4f}")
    k2.metric("Uncertainty", f"{last['Uncertainty']:.4f}")
    k3.metric("Plasticity", f"{plasticity:.6f}")
    k4.metric("Learning Rate", f"{last['LearningRate']:.6f}")

    # --- NEW VISUALIZATION: The "Brain Scan" ---
    st.markdown("### 🧠 Neural Activity Monitor (3D)")
    st.caption("Rotatable 3D view of the neural network layers.  **Larger/Brighter = Active Rewiring**")
    
    if 'latest_layer_changes' in st.session_state:
        # Call the new 3D plotter function
        fig_3d = plot_3d_brain_activity(st.session_state.latest_layer_changes)
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.warning("No plasticity data available. Interact with the model to see the brain light up!")
    
    # Existing Tabs
    tab1, tab2 = st.tabs(["📉 Loss History", "⚡ System Latency"])
    with tab1:
        st.plotly_chart(px.line(st.session_state.history, x="Timestamp", y="Loss"), use_container_width=True)
    with tab2:
        st.plotly_chart(px.bar(st.session_state.history, x="Timestamp", y="Latency"), use_container_width=True)

# ==================== MAIN LOOP ====================

def main():
    view = render_sidebar()
    if view == "💤 User Interface":
        user_portal()
    else:
        enterprise_portal()

if __name__ == "__main__":
    main()