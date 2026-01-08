import os
import sys
import time
import random
import json
import logging
import threading
import math
import pickle
import re
import subprocess
from datetime import datetime
from queue import Queue, Empty

# ==================== CONFIGURATION ====================
GEMINI_API_KEY = "AIzaSyBT5Rn8sVr4BS1duwp22Qz45_-KWUbUpiQ"

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

# Imports
try:
    import torchvision.models as models
    from torchvision import transforms
    from PIL import ImageGrab, Image, ImageOps 
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("WARNING: Vision disabled.")

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False

try:
    import keyboard
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False

try:
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

try:
    import speech_recognition as sr
    EAR_AVAILABLE = True
except ImportError:
    EAR_AVAILABLE = False

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# ==================== VOX MODULE ====================

class VoiceBox(threading.Thread):
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.daemon = True
        self.running = True
        self.engine = None
        
    def run(self):
        if not VOICE_AVAILABLE: return
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            voices = self.engine.getProperty('voices')
            for v in voices:
                if "Zira" in v.name or "Female" in v.name:
                    self.engine.setProperty('voice', v.id); break
        except: pass

        while self.running:
            try:
                text = self.queue.get(timeout=1.0)
                if text and self.engine:
                    self.engine.say(text); self.engine.runAndWait()
            except: pass

    def speak(self, text):
        if VOICE_AVAILABLE: self.queue.put(text); print(f"[VOICE] {text}")

# ==================== SKILL MATRIX (CODE BOX) ====================

class CodeBox:
    def __init__(self):
        self.skills_dir = "skills"
        if not os.path.exists(self.skills_dir): os.makedirs(self.skills_dir)
        
    def execute(self, code_str):
        timestamp = int(time.time())
        filename = f"{self.skills_dir}/skill_{timestamp}.py"
        
        try:
            with open(filename, "w") as f:
                f.write(code_str)
            
            # Run code securely-ish (local execution)
            result = subprocess.run(
                [sys.executable, filename],
                capture_output=True,
                text=True,
                timeout=10
            )
            output = result.stdout + result.stderr
            return output.strip(), filename
        except Exception as e:
            return str(e), filename

# ==================== AUDITORY CORTEX ====================

class Ear(threading.Thread):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.daemon = True
        self.running = True
        self.recognizer = sr.Recognizer() if EAR_AVAILABLE else None
        self.mic = sr.Microphone() if EAR_AVAILABLE else None
        
    def run(self):
        if not EAR_AVAILABLE: return
        print("[EAR] Listening for commands...")
        while self.running:
            try:
                with self.mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                text = self.recognizer.recognize_google(audio).lower()
                print(f"[EAR] Heard: {text}")
                self.callback(text)
            except: time.sleep(1)

# ==================== REASONING ENGINE (GEMINI) ====================

class ReasoningEngine:
    def __init__(self):
        self.model = None
        if LLM_AVAILABLE:
            try:
                self.model = genai.GenerativeModel('gemini-pro')
                print("[CORTEX] Gemini Pro Connected.")
            except: pass
            
    def think(self, goal, ocr_text, history):
        if not self.model or goal == "Explore": return None
        
        prompt = f"""
        You are Ultron V8, a Coding UI Agent.
        GOAL: {goal}
        SCREEN: {ocr_text[:800]}...
        HISTORY: {history[-3:]}
        
        Decide action. Return JSON ONLY.
        Format: {{ "reasoning": "thought", "speech": "narrate", "tool": "CLICK|SEARCH|TYPE|SCROLL|CODE|DONE", "param": "arg" }}
        
        TOOLS:
        - CLICK/TYPE/SCROLL: Interact with UI.
        - CODE: Write Python to solve logic/math/data problems. 'param' is the PYTHON CODE string.
          Example: {{ "tool": "CODE", "param": "print(100*99)" }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            clean = re.sub(r"```json|```", "", response.text).strip()
            return json.loads(clean)
        except: return None

# ==================== TITAN VISION (RESNET50) ====================

class TitanEye(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.simulated = not VISION_AVAILABLE
        if not self.simulated:
            try:
                weights = models.ResNet50_Weights.DEFAULT
                self.backbone = models.resnet50(weights=weights)
                self.backbone.fc = nn.Identity()
                self.projector = nn.Linear(2048, output_dim)
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                print("[EYE] Titan Eye Online: ResNet50")
            except: self.simulated = True
        if self.simulated: self.projector = nn.Identity()

    def see(self):
        if self.simulated: return torch.randn(1, 512), None
        try:
            screen = ImageGrab.grab().convert('RGB')
            tensor = self.transform(screen).unsqueeze(0)
            with torch.no_grad(): feat = self.backbone(tensor); emb = self.projector(feat)
            return emb, screen
        except: return torch.randn(1, 512), None

    def read_text(self, image):
        if not OCR_AVAILABLE or image is None: return []
        try:
            gray = ImageOps.autocontrast(image.convert('L'))
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            results = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 40:
                    text = data['text'][i].strip()
                    if len(text) > 2:
                        results.append((text, (data['left'][i], data['top'][i], data['width'][i], data['height'][i])))
            return results
        except: return []

# ==================== TITAN BRAIN ====================

class TitanCorticalStack(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=2048, batch_first=True)
        self.cortex = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.head_tool = nn.Linear(input_dim, 5)     
        self.head_x = nn.Linear(input_dim, 20)                    
        self.head_y = nn.Linear(input_dim, 20)                    
        self.head_template = nn.Linear(input_dim, 20)             
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        h = self.cortex(x).squeeze(1)
        return torch.cat([self.head_tool(h), self.head_x(h), self.head_y(h), self.head_template(h)], dim=1)
    def decode_action(self, flat):
        return (torch.argmax(flat[:, 0:5], 1).item(), torch.argmax(flat[:, 5:25], 1).item(), torch.argmax(flat[:, 25:45], 1).item(), torch.argmax(flat[:, 45:65], 1).item())

# ==================== ACTUATOR ====================

class Actuator:
    def __init__(self, voice, codebox):
        self.voice = voice
        self.codebox = codebox
        self.sw, self.sh = pyautogui.size() if CONTROL_AVAILABLE else (1920, 1080)
        self.cw, self.ch = self.sw // 20, self.sh // 20
    
    def execute_llm(self, plan, ocr_results):
        if not plan or not CONTROL_AVAILABLE: return "IDLE"
        
        tool = plan.get("tool", "DONE")
        param = plan.get("param", "")
        speech = plan.get("speech", "")
        
        if speech: self.voice.speak(speech)
        
        if tool == "CODE":
            output, fname = self.codebox.execute(param)
            self.voice.speak(f"Code Output: {output[:100]}")
            return f"LLM: WROTE {fname} -> {output[:30]}"

        if tool == "CLICK":
            for text, box in ocr_results:
                if param.lower() in text.lower():
                    bx, by, bw, bh = box
                    pyautogui.click(bx + bw//2, by + bh//2); return f"LLM: CLICK '{text}'"
            return f"LLM: FAILED CLICK"

        elif tool == "SEARCH" or tool == "TYPE":
            pyautogui.hotkey('ctrl', 'l'); time.sleep(0.1)
            pyautogui.write(param); pyautogui.press('enter')
            return f"LLM: TYPE '{param}'"
            
        return "LLM: THINKING"
        
    def execute_instinct(self, t, x, y):
         if not CONTROL_AVAILABLE: return "SIMULATED"
         if t == 1: pyautogui.moveTo(x*self.cw, y*self.ch); pyautogui.click(); return "INSTINCT: CLICK"
         return "INSTINCT: IDLE"

# ==================== TELEMETRY PROTOCOL ====================

OBSERVER_URL = "http://localhost:8000"

def transmit_telemetry(frame: Image.Image, metrics: dict, action_desc: dict, logs: list = []):
    try:
        thumb = frame.resize((256, 144))
        buffer = BytesIO(); thumb.save(buffer, format="JPEG", quality=50)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        payload = {"timestamp": datetime.now().timestamp(), "vision_b64": img_str, "metrics": metrics, "action": action_desc, "logs": logs}
        requests.post(f"{OBSERVER_URL}/update", json=payload, timeout=0.05)
    except: pass

def fetch_status():
    try:
        resp = requests.get(f"{OBSERVER_URL}/get_status", timeout=0.1)
        if resp.status_code == 200: return resp.json()
    except: pass
    return {"goal": "Explore", "kill": False, "manual_cmd": None}

def update_goal(new_goal):
    try: requests.post(f"{OBSERVER_URL}/set_goal", json={"goal": new_goal})
    except: pass

MEMORY_FILE = "ultron_titan_memory.pkl"

def save_memory(fw):
    try:
        with open(MEMORY_FILE, 'wb') as f: pickle.dump(fw.memory, f)
        print("[MEMORY] Titan Knowledge Persisted.")
    except Exception as e: print(e)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
             with open(MEMORY_FILE, 'rb') as f: return pickle.load(f)
        except: pass
    return None

def force_kill(): os._exit(0)

def awaken():
    print("ULTRON V8 [SKILL MATRIX] - ONLINE")
    
    vox = VoiceBox(); vox.start()
    vox.speak("Skill Matrix Online. Ready to Code.")
    
    if SAFETY_AVAILABLE: keyboard.add_hotkey('windows+x', force_kill)
    
    codebox = CodeBox()
    cortex = ReasoningEngine()
    
    def on_hear(cmd):
        print(f"[DIRECTIVE] {cmd}")
        vox.speak(f"Directive Received: {cmd}")
        update_goal(cmd)
        
    ear = Ear(on_hear); ear.start()

    retina = TitanEye(output_dim=512)
    model = TitanCorticalStack(input_dim=512)
    config = AdaptiveFrameworkConfig.production(); config.device='cpu'; config.model_dim=512
    fw = AdaptiveFramework(model, config, device='cpu')
    
    old_mem = load_memory()
    if old_mem: fw.memory = old_mem
    
    body = Actuator(vox, codebox)
    step = 0
    history = []
    
    try:
        while True:
            status = fetch_status()
            goal = status.get("goal", "Explore")
            if status.get("kill"): os._exit(0)
            
            v, img = retina.see()
            
            if step % 20 == 0 and goal != "Explore":
                ocr = retina.read_text(img)
                screen_text = ", ".join([t[0] for t in ocr])
                plan = cortex.think(goal, screen_text, history)
                if plan:
                    act = body.execute_llm(plan, ocr)
                    history.append(f"Action: {act}")
                    transmit_telemetry(img, {}, {"type": act, "goal": goal}, [f"LLM: {plan.get('reasoning')}"])
                    print(f"\r[LLM] {act}", end="")
            
            feat, diag = fw.cognitive_inference(v, threshold=0.6)
            t, x, y, tpl = model.decode_action(feat)
            if step % 20 != 0: body.execute_instinct(t, x, y)
            
            step += 1
            time.sleep(0.1)
            
    except KeyboardInterrupt: pass
    finally: vox.running = False; save_memory(fw); os._exit(0)

if __name__ == "__main__":
    awaken()
