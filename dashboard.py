
import streamlit as st
import requests
import pandas as pd
import time
import base64
from io import BytesIO
from PIL import Image
import altair as alt

# ==================== CONFIG ====================

st.set_page_config(
    page_title="Ultron Mind Reader",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sci-Fi look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #00ff00;
        font-family: 'Courier New', Courier, monospace;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #374151;
    }
    .stMetricValue {
        color: #00ff00 !important;
    }
</style>
""", unsafe_allow_html=True)

OBSERVER_URL = "http://localhost:8000"

# ==================== LOGIC ====================

def fetch_telemetry():
    try:
        resp = requests.get(f"{OBSERVER_URL}/telemetry", timeout=0.1)
        if resp.status_code == 200:
            return resp.json()
    except:
        return None
    return None

def fetch_history():
    try:
        resp = requests.get(f"{OBSERVER_URL}/history", timeout=0.2)
        if resp.status_code == 200:
            return resp.json()
    except:
        return []
    return []

# ==================== UI ====================

st.title("🧠 Ultron Mind Reader")

# Mission Control
with st.container():
    col_mission, col_status = st.columns([3, 1])
    with col_mission:
        new_goal = st.text_input("Current Mission (Press Enter to Update)", placeholder="e.g. Find python tutorials")
        if new_goal:
            try:
                requests.post(f"{OBSERVER_URL}/set_goal", json={"goal": new_goal})
                st.toast(f"Mission Updated: {new_goal}")
            except:
                st.error("Observer Offline")

col1, col2 = st.columns([1, 2])

# Placeholder for real-time loop using st.empty() is old school.
# Checkboxes to control refresh
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh Rate (ms)", 100, 2000, 500)

placeholder = st.empty()

while auto_refresh:
    data = fetch_telemetry()
    history = fetch_history()
    
    with placeholder.container():
        if not data or "status" in data:
            st.warning("Waiting for Ultron telemetry...")
        else:
            # Layout
            # Top: Metrics
            met1, met2, met3, met4 = st.columns(4)
            
            metrics = data.get("metrics", {})
            cons = metrics.get("consciousness", {})
            entropy = cons.get("entropy", 0.0)
            mode = metrics.get("mode", "Sys1")
            
            with met1:
                st.metric("Mode", mode)
            with met2:
                st.metric("Entropy", f"{entropy:.3f}", delta_color="inverse")
            with met3:
                st.metric("Dreaming", "ACTIVE" if metrics.get("dreaming") else "IDLE")
            with met4:
                st.metric("Action", data["action"].get("type", "IDLE"))

            # Middle: Vision & Thought
            c_vision, c_graphs = st.columns([1, 2])
            
            with c_vision:
                st.subheader("Visual Cortex")
                b64 = data.get("vision_b64")
                if b64:
                    try:
                        img_data = base64.b64decode(b64)
                        st.image(Image.open(BytesIO(img_data)), use_column_width=True)
                    except:
                        st.error("Image Decode Fail")
                else:
                    st.info("No Visual Stream")
                    
                # Action Plan
                act = data.get("action", {})
                st.code(f"EXEC: {act}", language="json")

            with c_graphs:
                st.subheader("Consciousness Trace")
                if history:
                    df = pd.DataFrame(history)
                    # Line Chart for Entropy
                    chart = alt.Chart(df).mark_line().encode(
                        x=alt.X('timestamp', axis=None),
                        y=alt.Y('entropy', scale=alt.Scale(domain=[0, 1])),
                        color=alt.value("#00ff00")
                    ).properties(height=200)
                    st.altair_chart(chart, use_container_width=True)
                    
            # Bottom: OCR / Logs
            st.subheader("Internal Monologue")
            logs = data.get("logs", [])
            if logs:
                st.text_area("Logs", "\n".join(logs[-5:]), height=100)
            
    time.sleep(refresh_rate / 1000.0)
