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
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Try Semantic Imports (The "Ghost" Engine)
try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("WARNING: sentence-transformers not found. Semantic understanding disabled.")

# Try Control Imports
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    print("WARNING: pyautogui not found. Actions will be simulated.")

# ==================== TELEMETRY PROTOCOL ====================

OBSERVER_URL = "http://localhost:8000"

def transmit_telemetry(frame: Image.Image, metrics: dict, action_desc: dict, logs: list = []):
    """Transmit state to the Observer Node."""
    try:
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
        requests.post(f"{OBSERVER_URL}/update", json=payload, timeout=0.05)
    except Exception:
        pass

def fetch_goal():
    """Poll mission from Observer."""
    try:
        resp = requests.get(f"{OBSERVER_URL}/get_goal", timeout=0.1)
        if resp.status_code == 200:
            return resp.json().get("goal", "Explore")
    except:
        return "Explore"
    return "Explore"

# ==================== 1. PERCEPTION (THE EYES) ====================

class Retina(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.simulated = not VISION_AVAILABLE
        if not self.simulated:
            try:
                self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
                self.backbone.classifier = nn.Identity()
                with torch.no_grad():
                    dummy = torch.randn(1, 3, 224, 224)
                    self.feat_dim = self.backbone(dummy).shape[-1]
                self.projector = nn.Linear(self.feat_dim, output_dim)
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                print("[EYE] Retina Online: MobileNetV3-Small")
            except Exception:
                self.simulated = True
        
        if self.simulated:
            self.projector = nn.Identity()

    def see(self):
        if self.simulated: return torch.randn(1, 128), None
        try:
            screen = ImageGrab.grab().convert('RGB')
            tensor = self.transform(screen).unsqueeze(0)
            with torch.no_grad():
                feat = self.backbone(tensor)
                emb = self.projector(feat)
            return emb, screen
        except:
            return torch.randn(1, 128), None

    def read_text(self, image):
        """Returns list of (text, box_tuple) where box=(left, top, width, height)"""
        if not OCR_AVAILABLE or image is None: return []
        try:
            # Get detailed data including boxes
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            results = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 40: # Confidence threshold
                    text = data['text'][i].strip()
                    if len(text) > 2:
                        box = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                        results.append((text, box))
            return results
        except:
            return []

# ==================== 2. SEMANTIC CORTEX (THE GHOST) ====================

class SemanticBox(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        if SEMANTIC_AVAILABLE:
            try:
                # Load small efficient model
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print("[GHOST] Semantic Engine Online.")
            except Exception as e:
                print(f"[GHOST] Failed: {e}")

    def find_match(self, goal, text_candidates):
        """
        Find best matching text on screen for the goal.
        Returns: (text, box, score)
        """
        if not self.encoder or not text_candidates or goal.lower() == "explore":
            return None, None, 0.0
            
        texts = [t[0] for t in text_candidates]
        boxes = [t[1] for t in text_candidates]
        
        # Encode
        goal_emb = self.encoder.encode(goal, convert_to_tensor=True)
        text_embs = self.encoder.encode(texts, convert_to_tensor=True)
        
        # Cosine Sim
        scores = util.cos_sim(goal_emb, text_embs)[0]
        
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        
        if best_score > 0.4: # Threshold for semantic match
            return texts[best_idx], boxes[best_idx], best_score
            
        return None, None, best_score

# ==================== 3. THE BRAIN (NEURAL) ====================

class UltronCorticalStack(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.cortex = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.head_tool = nn.Linear(128, 5)     
        self.head_x = nn.Linear(128, 20)                    
        self.head_y = nn.Linear(128, 20)                    
        self.head_template = nn.Linear(128, 20)             
        
    def forward(self, x):
        h = self.cortex(x)
        return torch.cat([self.head_tool(h), self.head_x(h), self.head_y(h), self.head_template(h)], dim=1)

    def decode_action(self, flat):
        return (torch.argmax(flat[:, 0:5], 1).item(), 
                torch.argmax(flat[:, 5:25], 1).item(), 
                torch.argmax(flat[:, 25:45], 1).item(), 
                torch.argmax(flat[:, 45:65], 1).item())

# ==================== 4. ACTUATOR WITH EMOTION ====================

USEFUL_QUERIES = ["python tutorials", "news", "music", "weather"]
USEFUL_URLS = ["https://google.com", "https://youtube.com", "https://github.com"]

class Actuator:
    def __init__(self):
        self.sw, self.sh = pyautogui.size() if CONTROL_AVAILABLE else (1920, 1080)
        self.cw, self.ch = self.sw // 20, self.sh // 20
        self.last_search = 0

    def execute(self, tool, x, y, tpl, semantic_override=None, emotional_override=None):
        if not CONTROL_AVAILABLE: return "SIMULATED"
        
        # Priority 1: Semantic Override (Goal Match)
        if semantic_override:
            text, box, score = semantic_override
            bx, by, bw, bh = box
            center_x, center_y = bx + bw//2, by + bh//2
            pyautogui.moveTo(center_x, center_y, duration=0.3)
            pyautogui.click()
            return f"SEMANTIC('{text}' {score:.2f})"

        # Priority 2: Emotional Override (Boredom)
        if emotional_override == "BOREDOM":
            # Context Switch
            url = random.choice(USEFUL_URLS)
            pyautogui.hotkey('ctrl', 'l')
            time.sleep(0.1)
            pyautogui.write(url)
            pyautogui.press('enter')
            return f"BORED->GO('{url}')"
            
        # Priority 3: Neural Default
        if tool == 1: # CLICK
            tx, ty = x * self.cw + self.cw//2, y * self.ch + self.ch//2
            pyautogui.moveTo(tx, ty, duration=0.2)
            pyautogui.click()
            return f"CLICK({x}, {y})"
        elif tool == 2: # SEARCH
            if time.time() - self.last_search > 5.0:
                q = USEFUL_QUERIES[tpl % len(USEFUL_QUERIES)]
                pyautogui.hotkey('ctrl', 'l'); time.sleep(0.1)
                pyautogui.write(q); pyautogui.press('enter')
                self.last_search = time.time()
                return f"SEARCH('{q}')"
        # ... simplified other tools ...
        return "IDLE"

# ==================== 5. MAIN LOOP ====================

class SubconsciousThread(threading.Thread):
    def __init__(self, fw): super().__init__(); self.fw = fw; self.running=True
    def run(self): 
        while self.running: 
            try: self.fw.learn_from_buffer(batch_size=8); time.sleep(0.5)
            except: time.sleep(1.0)

def awaken():
    print("ULTRON V4 [SEMANTIC GHOST] - ONLINE")
    config = AdaptiveFrameworkConfig.production(); config.device='cpu'
    config.model_dim=128
    
    retina = Retina()
    semantic = SemanticBox()
    model = UltronCorticalStack()
    fw = AdaptiveFramework(model, config, device='cpu')
    body = Actuator()
    
    SubconsciousThread(fw).start()
    
    step = 0
    try:
        while True:
            # 1. State
            goal = fetch_goal()
            v, img = retina.see()
            
            # 2. Semantic Analysis (every 10 steps or if bored)
            ocr_res = []
            sem_match = None
            if step % 10 == 0:
                ocr_res = retina.read_text(img)
                if ocr_res and goal != "Explore":
                    sem_match = semantic.find_match(goal, ocr_res) # (text, box, score)

            # 3. Neural Think
            feat, diag = fw.cognitive_inference(v, threshold=0.6)
            entropy = diag.get('consciousness', {}).get('entropy', 0.0)
            surprise = diag.get('consciousness', {}).get('surprise', 0.0)
            
            # 4. Emotional Logic
            emotion_override = None
            if entropy > 0.8 and surprise < 0.1: # Confused but nothing new happens -> BORED
                emotion_override = "BOREDOM"
            
            # 5. Act
            t, x, y, tpl = model.decode_action(feat)
            act = body.execute(t, x, y, tpl, sem_match, emotion_override)
            
            # 6. Report
            logs = []
            if sem_match: logs.append(f"GHOST: Found '{sem_match[0]}' for goal '{goal}'")
            if emotion_override: logs.append("EMOTION: Bored -> Switching Context")
            
            state_desc = {"type": act, "goal": goal}
            if img: transmit_telemetry(img, diag, state_desc, logs)
            
            print(f"\rStep: {step} | Goal: {goal[:10]} | Act: {act:25s}", end="")
            step += 1
            time.sleep(0.1)
            
    except KeyboardInterrupt: pass
    finally: fw.save_checkpoint("ultron_v4_core.pt")

if __name__ == "__main__":
    awaken()
