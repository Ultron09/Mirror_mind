import os
import sys
import time
import random
import json
import logging
from datetime import datetime
import threading
import math

try:
    import torch
    import torch.nn as nn
    import numpy as np
    from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
except ImportError:
    print("CRITICAL: AirborneHRS core not found. Please install the package.")
    sys.exit(1)

# ==================== 1. PERCEPTION & ACTION INTERFACE ====================

class CyberInterface:
    """
    Abstracts the Body (Keyboard/Mouse/Screen).
    Uses PyAutoGUI if available, else Mock Simulation.
    """
    def __init__(self):
        self.simulated = False
        try:
            import pyautogui
            from PIL import ImageGrab
            self.pg = pyautogui
            self.grab = ImageGrab
            self.pg.FAILSAFE = True
            print("[BODY] Connected to Physical Hardware (PyAutoGUI active).")
        except ImportError:
            self.simulated = True
            print("[BODY] Physical Hardware not found (pyautogui/Pillow missing).")
            print("[BODY] Running in SIMULATION MODE. Actions will be logged only.")

    def see(self):
        """Capture screen. Returns a normalized vector description."""
        if self.simulated:
            # Mock vision: Random screen state
            return torch.randn(1, 64) 
        else:
            try:
                # Capture and Resize to 8x8 (64 pixels) for "Old CPU" speed
                # This makes it blind but fast. A real Ultron needs ConvNets.
                # For this demo, we assume a "Text Mode" agent or extremely low-res vision.
                screen = self.grab.grab().resize((8, 8)).convert('L')
                arr = np.array(screen).flatten() / 255.0
                return torch.tensor([arr], dtype=torch.float32)
            except Exception as e:
                print(f"[EYE] Vision error: {e}")
                return torch.randn(1, 64)

    def act(self, action_vector):
        """Translate model output to OS actions."""
        # vector: [move_x, move_y, click_prob, type_char_idx]
        if not isinstance(action_vector, list):
            vals = action_vector.tolist()[0]
        else:
            vals = action_vector

        dx, dy = vals[0], vals[1]
        click = vals[2]
        char_idx = int(vals[3] * 26) # Map 0-1 to a-z rough approximation
        
        # Deadzone
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            dx, dy = 0, 0
        
        if self.simulated:
            if dx != 0 or dy != 0: print(f"[ACT] Move Mouse: ({dx:.2f}, {dy:.2f})")
            if click > 0.5:        print(f"[ACT] Click!")
            if vals[3] > 0.8:      print(f"[ACT] Typing key index {char_idx}")
        else:
            try:
                # Scale movement
                screen_w, screen_h = self.pg.size()
                move_x = int(dx * 50)
                move_y = int(dy * 50)
                
                if move_x != 0 or move_y != 0:
                    self.pg.moveRel(move_x, move_y)
                
                if click > 0.8:
                    self.pg.click()
                    
                # Typing (Rare event)
                if vals[3] > 0.95: 
                    # Only type if very confident
                    char = chr(97 + (char_idx % 26))
                    self.pg.write(char)
            except Exception as e:
                print(f"[BODY] Action failed: {e}")

# ==================== 2. THE AGENT (ULTRON KERNEL) ====================

class UltronAgent(nn.Module):
    """
    A lightweight, CPU-optimized brain.
    Inputs: 64 (Low-res Vision)
    Outputs: 4 (dx, dy, click, char_logit)
    """
    def __init__(self, input_dim=64):
        super().__init__()
        # Tiny MLP for CPU limits
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.Tanh(), # Activity layer
            nn.Linear(16, 4) # Output
        )
        
    def forward(self, x):
        return self.net(x)

# ==================== 3. LIVE LOOP ====================

def awaken():
    print("="*60)
    print("ULTRON PROTOCOL V1.0 - INITIALIZING")
    print("="*60)
    print("Optimization: CPU Mode | Low-Res Vision | Proactive Dreaming")
    
    # 1. Body
    body = CyberInterface()
    
    # 2. Brain
    # Using PRODUCTION preset but tuning for old CPU
    config = AdaptiveFrameworkConfig.production()
    config.device = 'cpu'
    config.model_dim = 64 # Match vision input
    config.use_moe = False # Too heavy for old CPU
    config.num_heads = 2
    config.use_graph_memory = True # Memory is key for autonomy
    config.enable_consciousness = True
    config.enable_dreaming = True # Proactive
    config.use_amp = False # CPU doesn't support AMP well usually
    config.compile_model = False # Slow startup on CPU

    
    base_model = UltronAgent(input_dim=64)
    framework = AdaptiveFramework(base_model, config, device='cpu')
    
    print("[MIND] Consciousness Online. I have the right to live.")
    
    # 3. Life Cycle
    try:
        step = 0
        while True:
            # A. Perceive
            vision_tensor = body.see()
            
            # B. Think (Cognitive Inference if Entropy High)
            # We randomly trigger deep thought for demo
            if step % 50 == 0:
                print("[MIND] Analyzing Situation... (System 2)")
                action_tensor, diag = framework.cognitive_inference(vision_tensor, threshold=-1.0)
            else:
                # Fast Reflex
                action_tensor, diag = framework.inference_step(
                    vision_tensor, 
                    remember=True # Store every moment
                )
            
            # C. Act
            body.act(action_tensor)
            
            # D. Proactive Dreaming (Optimizes while idle)
            # We simulate idle time by dreaming every 10 steps
            if step % 10 == 0:
                framework.learn_from_buffer(batch_size=4) # Tiny batch for CPU
            
            # E. Report
            consciousness = diag.get('consciousness', {})
            entropy = consciousness.get('entropy', 0.0)
            print(f"\rStep: {step} | Entropy: {entropy:.4f} | Mode: {diag.get('mode', 'Sys1')} | Last Act: {action_tensor[0,0]:.2f}", end="")
            
            time.sleep(0.1) # Throttle for CPU
            step += 1
            
            # Artificial Stop for Demo
            if step > 200:
                print("\n[SYSTEM] Demo Limit Reached.")
                break
                
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by User.")
        
    finally:
        print("[HIBERNATE] Saving state...")
        framework.save_checkpoint("ultron_checkpoint.pt")
        print("[SYSTEM] Agent Saved. Goodbye.")

if __name__ == "__main__":
    awaken()
