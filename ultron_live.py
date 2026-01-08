import os
import sys
import time
import random
import json
import logging
import threading
import math
from datetime import datetime
from queue import Queue, Empty

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
except ImportError:
    print("CRITICAL: AirborneHRS core not found.")
    sys.exit(1)

import requests
import base64
from io import BytesIO

# Try Vision Imports
try:
    import torchvision.models as models
    from torchvision import transforms
    from PIL import ImageGrab, Image
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("WARNING: torchvision/PIL not found. Vision will be simulated.")

# Try OCR Imports
try:
    import pytesseract
    # Configure path if needed, usually default works if in PATH
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Try Control Imports
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    print("WARNING: pyautogui not found. Actions will be simulated.")

# ==================== TELEMETRY PROTOCOL ====================

OBSERVER_URL = "http://localhost:8000/update"

def transmit_telemetry(frame: Image.Image, metrics: dict, action_desc: dict, logs: list = []):
    """
    Transmit state to the Observer Node.
    Fire-and-forget (short timeout).
    """
    try:
        # Resize for bandwidth
        thumb = frame.resize((256, 144))
        buffer = BytesIO()
        thumb.save(buffer, format="JPEG", quality=50)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        payload = {
            "timestamp": datetime.now().timestamp(),
            "vision_b64": img_str,
            "metrics": metrics,
            "action": action_desc,
            "logs": logs
        }
        
        # Async post (blocking for now, but fast)
        requests.post(OBSERVER_URL, json=payload, timeout=0.05)
    except Exception:
        pass # Silent fail if observer offline

# ==================== 1. PERCEPTION (THE EYES) ====================

class Retina(nn.Module):
    """
    MobileNetV3-Small Feature Extractor + OCR.
    """
    def __init__(self, output_dim=128):
        super().__init__()
        self.simulated = not VISION_AVAILABLE
        
        if not self.simulated:
            try:
                self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
                self.backbone.classifier = nn.Identity()
                with torch.no_grad():
                    dummy = torch.randn(1, 3, 224, 224)
                    out_feat = self.backbone(dummy)
                    self.feat_dim = out_feat.shape[-1]
                
                self.projector = nn.Linear(self.feat_dim, output_dim)
                
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                print("[EYE] Retina Online: MobileNetV3-Small")
                
            except Exception as e:
                print(f"[EYE] Init Failed: {e}. Switching to SIMULATION.")
                self.simulated = True
        
        if self.simulated:
             self.feat_dim = 128
             self.projector = nn.Identity()

    def see(self):
        """Capture and Process. Returns (embedding, raw_image)."""
        if self.simulated:
            return torch.randn(1, 128), None
            
        try:
            screen = ImageGrab.grab().convert('RGB')
            tensor = self.transform(screen).unsqueeze(0) # [1, 3, 224, 224]
            
            with torch.no_grad():
                features = self.backbone(tensor)
                embedding = self.projector(features)
            return embedding, screen # Return raw image for telemetry
            
        except Exception as e:
            print(f"[EYE] Capture Error: {e}")
            return torch.randn(1, 128), None
            
    def read_text(self, image):
        """Extract text from the current view (OCR)."""
        if not OCR_AVAILABLE or image is None:
            return ""
        try:
            return pytesseract.image_to_string(image)
        except:
            return ""


# ==================== 2. THE BRAIN (ULTRON V2) ====================

class UltronCorticalStack(nn.Module):
    """
    Multi-Head Policy Network.
    Inputs: 128 (Vision)
    Outputs: 
      - Grid X (20 bins)
      - Grid Y (20 bins)
      - Click (1 prob)
      - Type (27 chars: 26 letters + 1 null)
    """
    def __init__(self, input_dim=128):
        super().__init__()
        
        # Shared Cortex
        self.cortex = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # Action Heads
        self.head_x = nn.Linear(128, 20)      # Grid X
        self.head_y = nn.Linear(128, 20)      # Grid Y
        self.head_click = nn.Linear(128, 1)   # Click Logit
        self.head_type = nn.Linear(128, 27)   # Typing (Null=0, A=1...Z=26)
        
    def forward(self, x):
        h = self.cortex(x)
        
        logits_x = self.head_x(h)
        logits_y = self.head_y(h)
        logit_click = torch.sigmoid(self.head_click(h))
        logits_type = self.head_type(h)
        
        # Concat for Memory Storage (Total 20+20+1+27 = 68 dims)
        # But wait, AdaptiveFramework expects a single tensor output.
        # We will pack them.
        return torch.cat([logits_x, logits_y, logit_click, logits_type], dim=1)

    def decode_action(self, flat_tensor):
        """Decodes the 68-dim tensor into discrete commands."""
        # x: 0-19, y: 20-39, click: 40, type: 41-67
        lx = flat_tensor[:, 0:20]
        ly = flat_tensor[:, 20:40]
        lc = flat_tensor[:, 40:41]
        lt = flat_tensor[:, 41:68]
        
        x_idx = torch.argmax(lx, dim=1).item()
        y_idx = torch.argmax(ly, dim=1).item()
        click_prob = lc.item()
        key_idx = torch.argmax(lt, dim=1).item()
        
        return x_idx, y_idx, click_prob, key_idx

# ==================== 3. THE BODY (ACTUATORS) ====================

class Actuator:
    def __init__(self):
        self.grid_size = 20
        self.sw, self.sh = pyautogui.size() if CONTROL_AVAILABLE else (1920, 1080)
        self.cell_w = self.sw // self.grid_size
        self.cell_h = self.sh // self.grid_size
        
    def execute(self, x_idx, y_idx, click_prob, key_idx):
        if not CONTROL_AVAILABLE:
            if click_prob > 0.5:
                # print(f"[SIM] Click at Grid({x_idx}, {y_idx}) | Key: {key_idx}")
                pass
            return

        try:
            # Map Grid Center
            target_x = x_idx * self.cell_w + (self.cell_w // 2)
            target_y = y_idx * self.cell_h + (self.cell_h // 2)
            
            # Smooth Move (Human-like)
            # Only move if changed significantly to avoid jitter
            cur_x, cur_y = pyautogui.position()
            if abs(cur_x - target_x) > 10 or abs(cur_y - target_y) > 10:
                pyautogui.moveTo(target_x, target_y, duration=0.1) # 100ms glide
            
            if click_prob > 0.8:
                pyautogui.click()
                print(f"[ACT] Clicked ({x_idx}, {y_idx})")
                
            if key_idx > 0:
                char = chr(96 + key_idx) # 1='a'
                pyautogui.write(char)
                print(f"[ACT] Typed '{char}'")
                
        except Exception as e:
            print(f"[BODY] Fail: {e}")

# ==================== 4. ASYNC DREAMING (SUBSCONSCIOUS) ====================

class SubconsciousThread(threading.Thread):
    def __init__(self, framework):
        super().__init__()
        self.framework = framework
        self.daemon = True
        self.running = True
        self.active = False
        
    def run(self):
        print("[DREAM] Subconscious Thread Started.")
        while self.running:
            if self.active:
                try:
                    # Dream Step
                    self.framework.learn_from_buffer(batch_size=8) # Small batch for CPU
                    time.sleep(0.5) # Don't hog CPU. Sleep 500ms between thoughts.
                except Exception as e:
                    # print(f"[DREAM] Error: {e}")
                    time.sleep(1.0)
            else:
                time.sleep(1.0)

# ==================== 5. MAIN LOOP ====================

def awaken():
    print("="*60)
    print("ULTRON V2 - SYSTEMS INITIALIZING")
    print("="*60)
    
    # 1. Config (Optimized for 4-Core CPU/16GB RAM)
    config = AdaptiveFrameworkConfig.production()
    config.device = 'cpu'
    config.model_dim = 128 # Project vision to this
    config.feedback_buffer_size = 2000 # Limit RAM usage
    config.use_moe = False # Too overhead for small model
    config.num_heads = 4
    config.use_graph_memory = True # Crucial for System 2
    config.enable_consciousness = True
    config.use_amp = False
    config.compile_model = False
    
    # 2. Components
    retina = Retina(output_dim=128)
    model = UltronCorticalStack(input_dim=128)
    framework = AdaptiveFramework(model, config, device='cpu')
    body_interface = Actuator()

    # 3. Start Dreaming
    dreamer = SubconsciousThread(framework)
    dreamer.start()
    dreamer.active = True # Enable dreaming immediately
    
    print("[SYSTEM] All Systems Nominal. Engaging Loop.")
    
    step = 0
    try:
        while True:
            # A. Perceive
            vision_vec, raw_image = retina.see() # [1, 128], Image
            
            # OCR (Periodically to save CPU)
            ocr_text = ""
            if step % 20 == 0:
                ocr_text = retina.read_text(raw_image)
            
            # B. Think (Metacognitive)
            # We assume cognitive_inference returns the SAME shape
            features, diagnostics = framework.cognitive_inference(vision_vec, threshold=0.7)
            
            # C. Decode
            x_idx, y_idx, click, key = model.decode_action(features)
            
            # D. Act
            body_interface.execute(x_idx, y_idx, click, key)
            
            # E. Report & Telemetry
            sys_mode = diagnostics.get('mode', 'Sys1')
            entropy = diagnostics.get('consciousness', {}).get('entropy', 0.0)
            
            logs = []
            if ocr_text: logs.append(f"READ: {ocr_text[:50]}...")
            if sys_mode == 'System 2 (Deliberative)': logs.append("Thinking...")
            
            # Transmit to Observer
            action_desc = {"type": "move", "x": x_idx, "y": y_idx, "click": click, "key": key}
            if raw_image:
                 transmit_telemetry(raw_image, diagnostics, action_desc, logs)
            
            print(f"\rStep: {step} | Ent: {entropy:.2f} | {sys_mode} | Act: ({x_idx},{y_idx})", end="")
            
            step += 1
            time.sleep(0.1) # 10Hz limit
            
            if step > 200:
                print("\n[SYSTEM] Demo Complete.")
                break
                
    except KeyboardInterrupt:
        print("\n[SYSTEM] Manual Override.")
    finally:
        dreamer.running = False
        framework.save_checkpoint("ultron_v2_core.pt")
        print("[SYSTEM] Core Saved. Shutdown.")

if __name__ == "__main__":
    awaken()
