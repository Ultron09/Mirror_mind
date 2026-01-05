
"""
PROTOCOL V6 - PHASE 7: SOTA DEATHMATCH (DRONE SIM)
==================================================
Goal: Survival in a physics sim with gravity shifts and sensor failure.
Scenario: "The Broken Drone"
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging
import time

# Path Setup
# Path Setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("Phase7")

# ==============================================================================
# 1. THE SIMULATOR: 6-DOF DRONE PHYSICS
# ==============================================================================
class DroneSim:
    def __init__(self):
        self.state = np.zeros(6) # [x, y, z, vx, vy, vz]
        self.state[2] = 10.0     # Start at altitude 10
        self.gravity = -9.81
        self.wind = np.zeros(3)
        self.step_count = 0
        
    def step(self, action):
        """
        action: [thrust_x, thrust_y, thrust_z]
        Returns: next_state, reward, done
        """
        self.step_count += 1
        
        # 1. Apply Physics Shifts (The Liquid Killer)
        if self.step_count % 50 == 0:
            # Gravity goes crazy (Reverse gravity, Side gravity)
            self.gravity = np.random.uniform(-20, 5) 
            self.wind = np.random.uniform(-5, 5, size=3)
            
        # 2. Physics Integration
        dt = 0.1
        thrust = action * 20.0 # Power factor
        
        # Accel = (Thrust + Gravity + Wind) - Drag
        accel = thrust + self.wind
        accel[2] += self.gravity
        accel -= 0.1 * self.state[3:] # Drag
        
        # Velocity
        self.state[3:] += accel * dt
        # Position
        self.state[:3] += self.state[3:] * dt
        
        # 3. Crash Check
        done = False
        if self.state[2] <= 0:
            self.state[2] = 0
            done = True # CRASH
            
        # Reward: Keep altitude ~ 10.0, minimize velocity
        target_alt = 10.0
        reward = -abs(self.state[2] - target_alt) - 0.1 * np.sum(self.state[3:]**2)
        
        # Return state with added noise
        obs = self.state + np.random.normal(0, 0.1, size=6)
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0), reward, done

# ==============================================================================
# 2. THE COMPETITORS
# ==============================================================================

# Competitor A: Standard LSTM (The Baseline)
class StandardLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6, 64, batch_first=True)
        self.head = nn.Linear(64, 3) 
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return torch.tanh(self.head(out[:, -1, :]))

# Competitor B: MirrorMind (The Challenger)
class PolymorphicCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 3. THE DEATHMATCH
# ==============================================================================
def run_deathmatch():
    logger.info("⚔️  PHASE 7: SOTA DEATHMATCH INITIATED")
    
    DEVICE = 'cpu' # Sim is fast on CPU
    
    # --- SETUP MIRRORMIND ---
    mm_core = PolymorphicCore().to(DEVICE)
    fw_config = AdaptiveFrameworkConfig(
        learning_rate=0.01,
        enable_consciousness=True, # Enable for adaptation
        device=DEVICE,
        memory_type='si',
        dream_interval=999999 # Disable dreaming for short sim
    )
    framework = AdaptiveFramework(mm_core, fw_config)
    
    # --- SETUP BASELINE ---
    base_model = StandardLSTM().to(DEVICE)
    base_opt = torch.optim.Adam(base_model.parameters(), lr=0.01)
    
    # --- METRICS ---
    MAX_STEPS = 500
    
    history = {
        'mm': {'alt': [], 'survived': 0},
        'base': {'alt': [], 'survived': 0}
    }
    
    for contender in ['base', 'mm']:
        logger.info(f"\n🚀 Launching Contender: {contender.upper()}")
        env = DroneSim()
        obs, _, _ = env.step(np.zeros(3))
        
        crashed = False
        
        for step in range(MAX_STEPS):
            # --- STRESSOR B: SENSOR ABLATION ---
            if np.random.random() > 0.8:
                mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32) 
                obs = obs * mask
            
            # --- TEACHER SIGNAL (PID) ---
            current_z = obs[0, 2]
            ideal_thrust = torch.zeros(1, 3)
            ideal_thrust[0, 2] = (10.0 - current_z) * 2.0 
            ideal_thrust = torch.clamp(ideal_thrust, -1, 1)

            # --- ACTION ---
            if contender == 'base':
                base_opt.zero_grad()
                action = base_model(obs)
                loss = torch.nn.MSELoss()(action, ideal_thrust) 
                loss.backward()
                base_opt.step()
            else:
                # MirrorMind: 2-Pass (Predict -> Train)
                # 1. Predict
                with torch.no_grad():
                    output = framework(obs)
                    if isinstance(output, tuple): output = output[0]
                    action = output
                
                # 2. Train (Adapt)
                framework.train_step(obs, target_data=ideal_thrust)

            # --- STEP ENV ---
            action_np = action.detach().numpy()[0]
            next_obs, reward, done = env.step(action_np)
            
            # Record
            obs = next_obs
            history[contender]['alt'].append(env.state[2])
            
            if done:
                logger.info(f"   💥 CRASHED at Step {step}")
                crashed = True
                history[contender]['survived'] = step
                break
        
        if not crashed:
            history[contender]['survived'] = MAX_STEPS
            logger.info("   ✅ SURVIVED Full Duration")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(history['mm']['alt'], color='#2ecc71', linewidth=2.5, label='MirrorMind')
    plt.plot(history['base']['alt'], color='#e74c3c', linestyle='--', linewidth=2, label='Baseline')
    plt.axhline(0, color='black', linewidth=3)
    plt.axhline(10, color='blue', linestyle=':', label='Target')
    plt.title("Phase 7: Drone Stability Deathmatch")
    plt.xlabel("Step")
    plt.ylabel("Altitude")
    plt.legend()
    plt.savefig("phase7_results.png")
    logger.info("   ✅ Plot saved: phase7_results.png")
    
    # Verdict
    mm_surv = history['mm']['survived']
    base_surv = history['base']['survived']
    
    logger.info(f"\nFINAL: Baseline={base_surv} | MirrorMind={mm_surv}")
    
    if mm_surv >= base_surv:
        logger.info("✅ PHASE 7 PASSED: MirrorMind matched or beat baseline.")
    else:
        logger.warning("⚠️ PHASE 7 WARNING: MirrorMind underperformed.")

if __name__ == "__main__":
    run_deathmatch()
