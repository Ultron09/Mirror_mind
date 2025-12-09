"""
MirrorMind Enterprise Dashboard
===============================
A production-grade interface for Adaptive Meta-Learning.

Features:
- 👥 User Portal: Interact with the self-learning model.
- 🏢 Enterprise Portal: Monitor health, plasticity, and drift.
- 🧠 Real-time Introspection: Visualize model uncertainty.
"""

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
# Assuming airbornehrs package is available in the path
from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig, ProductionAdapter, InferenceMode

# ==================== 1. CORE SYSTEM SETUP ====================

# Page Config
st.set_page_config(
    page_title="MirrorMind Enterprise",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Enterprise" feel
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

# ==================== 2. MODEL DEFINITION ====================

class IntrospectiveGPT2(nn.Module):
    """Wraps GPT-2 with an Introspection Head for uncertainty estimation."""
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.backbone = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Stability: Freeze backbone, only train head + introspection
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.transformer.h[-1].parameters():
            param.requires_grad = True
            
        # Introspection Head (Estimates Confidence)
        self.hidden_dim = self.backbone.config.n_embd
        self.introspection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, return_internals=False):
        outputs = self.backbone(input_ids, output_hidden_states=True)
        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1]
        
        # Calculate uncertainty (Variance estimation)
        uncertainty = self.introspection_head(last_hidden).mean(dim=1, keepdim=True)
        
        return logits, uncertainty

class CustomFramework(AdaptiveFramework):
    """Bridge between PyTorch Model and AirborneHRS"""
    def __init__(self, config, model, device):
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=config.learning_rate
        )
        self.loss_history = deque(maxlen=100)
        self.step_count = 0

# ==================== 3. STATE MANAGEMENT ====================

@st.cache_resource
def load_system():
    """Initializes the AI System (Cached to persist across re-runs)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = IntrospectiveGPT2('gpt2')
    
    # 2. Configure Framework
    config = AdaptiveFrameworkConfig(
        learning_rate=1e-4,
        evaluation_frequency=1
    )
    framework = CustomFramework(config, model, device)
    
    # 3. Create Production Adapter
    adapter = ProductionAdapter(
        framework,
        inference_mode=InferenceMode.ONLINE, # Active Learning Enabled
        enable_meta_learning=True
    )
    
    return adapter, model.tokenizer

# Initialize Session State for Metrics
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Timestamp', 'Loss', 'Uncertainty', 'Latency', 'LearningRate'])

adapter, tokenizer = load_system()

# ==================== 4. UI COMPONENTS ====================

def render_sidebar():
    st.sidebar.title("🚀 MirrorMind")
    st.sidebar.markdown("---")
    view = st.sidebar.radio("Select Portal", ["👤 User Interface", "🏢 Enterprise Monitor"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("System Status")
    
    # Dynamic System Vitals
    device = "CUDA 🟢" if torch.cuda.is_available() else "CPU 🟡"
    mode = adapter.inference_mode.upper()
    
    st.sidebar.caption(f"Compute: **{device}**")
    st.sidebar.caption(f"Mode: **{mode}**")
    st.sidebar.caption(f"Model: **GPT-2 (Introspective)**")
    
    return view

def process_interaction(prompt):
    """Runs the model, updates weights, and records metrics."""
    start_time = time.time()
    device = adapter.framework.device
    
    # 1. Prepare Data
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # 2. Predict & Learn (The Magic Line)
    # We treat the user's prompt as the 'target' for self-supervised learning
    with st.spinner("Thinking & Learning..."):
        adapter.predict(inputs, update=True, target=inputs)
        
        # Generate completion for the user
        with torch.no_grad():
            output_tokens = adapter.framework.model.backbone.generate(
                inputs, max_new_tokens=50, pad_token_id=50256
            )
            response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            
    # 3. Telemetry
    latency = (time.time() - start_time) * 1000
    metrics = adapter.get_metrics()
    
    # Record Data
    new_row = {
        'Timestamp': datetime.now(),
        'Loss': metrics.get('loss', 0.0),
        'Uncertainty': metrics.get('uncertainty', 0.0),
        'Latency': latency,
        'LearningRate': metrics.get('current_lr', 0.0)
    }
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)
    
    return response

# ==================== 5. PORTAL VIEWS ====================

def user_portal():
    st.title("👤 User Interaction Portal")
    st.markdown("Interact with the model. **Note:** The model learns from *every* interaction in real-time.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area("Enter your prompt:", height=150, placeholder="Type something...")
        if st.button("Generate Response", type="primary"):
            if prompt:
                response = process_interaction(prompt)
                st.success("Response Generated")
                st.markdown(f"**Model Output:**\n\n> {response}")
            else:
                st.warning("Please enter a prompt.")
                
    with col2:
        st.info("ℹ️ **Active Learning**\n\nYour input is immediately used to update the model's neural weights via the Stabilizer system.")
        st.markdown("### Recent Stats")
        if not st.session_state.history.empty:
            last_run = st.session_state.history.iloc[-1]
            st.metric("Latency", f"{last_run['Latency']:.0f} ms")
            st.metric("Uncertainty", f"{last_run['Uncertainty']:.4f}")
        else:
            st.write("No interactions yet.")

def enterprise_portal():
    st.title("🏢 Enterprise Health Monitor")
    
    # 1. Health Diagnostics Logic
    if not st.session_state.history.empty:
        last_metrics = st.session_state.history.iloc[-1]
        loss = last_metrics['Loss']
        unc = last_metrics['Uncertainty']
        lr = last_metrics['LearningRate']
        
        # Determine Health Status
        if loss > 2.5 or unc > 2.0:
            status = "CRITICAL DRIFT"
            style = "status-critical"
            desc = "Model is confused or data distribution has shifted significantly."
        elif lr > 0.0005:
            status = "ADAPTING (HIGH PLASTICITY)"
            style = "status-warning"
            desc = "Stabilizer has spiked learning rate to compensate for new patterns."
        else:
            status = "HEALTHY / STABLE"
            style = "status-healthy"
            desc = "System operating within nominal parameters."
    else:
        status = "WAITING FOR DATA"
        style = "status-warning"
        desc = "No inference traffic detected."
        loss, unc, lr = 0, 0, 0

    # 2. Heads-Up Display (HUD)
    st.markdown(f"""
        <div class="status-card {style}">
            <h2>SYSTEM STATUS: {status}</h2>
            <p>{desc}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Live Telemetry")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Current Loss", f"{loss:.4f}", delta_color="inverse")
    kpi2.metric("Introspection (Uncertainty)", f"{unc:.4f}", delta_color="inverse")
    kpi3.metric("Neuroplasticity (LR)", f"{lr:.6f}")
    kpi4.metric("Total Interactions", len(st.session_state.history))
    
    # 3. Deep Dive Charts
    if not st.session_state.history.empty:
        tab1, tab2, tab3 = st.tabs(["📉 Loss Trajectory", "🧠 Cognitive State", "🐢 Latency"])
        
        df = st.session_state.history
        
        with tab1:
            fig_loss = px.line(df, x="Timestamp", y="Loss", title="Real-time Prediction Error (Loss)")
            fig_loss.update_traces(line_color='#FF4B4B')
            st.plotly_chart(fig_loss, use_container_width=True)
            
        with tab2:
            # Dual axis chart for Uncertainty vs LR
            fig_cog = go.Figure()
            fig_cog.add_trace(go.Scatter(x=df['Timestamp'], y=df['Uncertainty'], name='Uncertainty', line=dict(color='cyan')))
            fig_cog.add_trace(go.Scatter(x=df['Timestamp'], y=df['LearningRate'], name='Plasticity (LR)', line=dict(color='yellow'), yaxis='y2'))
            
            fig_cog.update_layout(
                title="Introspection & Adaptation Reflex",
                yaxis=dict(title="Uncertainty"),
                yaxis2=dict(title="Learning Rate", overlaying='y', side='right')
            )
            st.plotly_chart(fig_cog, use_container_width=True)
            
        with tab3:
            fig_lat = px.bar(df, x="Timestamp", y="Latency", title="System Latency (ms)")
            st.plotly_chart(fig_lat, use_container_width=True)
            
        # 4. Raw Data Log
        with st.expander("查看 Raw System Logs"):
            st.dataframe(df.sort_values(by="Timestamp", ascending=False))
            
            # Management Actions
            if st.button("💾 Trigger Emergency Checkpoint"):
                path = f"manual_checkpoint_{int(time.time())}.pt"
                adapter.save_checkpoint(path)
                st.success(f"State frozen to {path}")

# ==================== MAIN LOOP ====================

def main():
    view = render_sidebar()
    if view == "👤 User Interface":
        user_portal()
    else:
        enterprise_portal()

if __name__ == "__main__":
    main()