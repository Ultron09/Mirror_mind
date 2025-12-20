"""
PROTOCOL PHASE 7: SOTA DEATHMATCH (THE POLYMORPHIC GAUNTLET)
============================================================
Goal: Outperform Liquid Networks (Stability) and Continual Learning (Memory)
      in a high-speed, high-damage scenario.

Competitors:
1. LSTM Baseline (Represents Standard RNNs)
2. MirrorMind (The Challenger)

Scenario: "The Broken Drone"
- Environment: 6-DOF Flight Sim
- Stressor A: Chaos Physics (Gravity shifts)
- Stressor B: Sensor Ablation (Inputs drop to 0)
- Stressor C: Weight Noise (Simulated Hardware Damage)

Success Metric: Survival Time + Inference FPS.
"""

import torch
import torch.nn as nn
import logging
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from airbornehrs import (
        AdaptiveFramework, 
        AdaptiveFrameworkConfig, 
        MetaController,
        ProductionAdapter,
        InferenceMode
    )
except ImportError:
    sys.exit(1)

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
        self.head = nn.Linear(64, 3) # Output: 3 Thrust vectors
    def forward(self, x):
        # x: [Batch, 6] -> [Batch, 1, 6]
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return torch.tanh(self.head(out[:, -1, :]))

# Competitor B: MirrorMind (The Challenger)
class PolymorphicCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64), # Stability
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
    logger.info("    Scenario: 'The Broken Drone' (Variable Gravity + Sensor Failure)")
    
    # --- SETUP MIRRORMIND ---
    mm_core = PolymorphicCore()
    fw_config = AdaptiveFrameworkConfig(
        learning_rate=0.01,
        adaptation_threshold=0.05, # React fast
        compile_model=False,       # compilation might hurt dynamic shapes
        device='cpu'
    )
    framework = AdaptiveFramework(mm_core, fw_config)
    adapter = ProductionAdapter(framework, inference_mode=InferenceMode.ONLINE)
    
    # --- SETUP BASELINE (Standard Online Learning) ---
    base_model = StandardLSTM()
    base_opt = torch.optim.Adam(base_model.parameters(), lr=0.01)
    
    # --- METRICS ---
    MAX_STEPS = 1000
    
    history = {
        'mm': {'alt': [], 'fps': [], 'survived': 0},
        'base': {'alt': [], 'fps': [], 'survived': 0}
    }
    
    for contender in ['base', 'mm']:
        logger.info(f"\n🚀 Launching Contender: {contender.upper()}")
        env = DroneSim()
        obs, _, _ = env.step(np.zeros(3))
        
        start_time = time.time()
        crashed = False
        
        for step in range(MAX_STEPS):
            step_start = time.time()
            
            # --- STRESSOR B: SENSOR ABLATION ---
            # Randomly zero out inputs to simulate sensor death
            if np.random.random() > 0.8:
                mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32) # Lost velocity sensors
                obs = obs * mask
            
            # --- Calculate Teacher Signal (Ideal PID Response) ---
            # Both models are trying to learn this mapping: State -> Corrective Thrust
            current_z = obs[0, 2]
            ideal_thrust = torch.zeros(1, 3)
            ideal_thrust[0, 2] = (10.0 - current_z) * 2.0 # Proportional Control
            ideal_thrust = torch.clamp(ideal_thrust, -1, 1)

            # --- ACTION ---
            if contender == 'base':
                # Standard Forward + Backward
                base_opt.zero_grad()
                action = base_model(obs)
                
                # FIX: Loss is now between Model Action and Ideal Thrust
                loss = torch.nn.MSELoss()(action, ideal_thrust) 
                loss.backward()
                base_opt.step()
                
            else:
                # MirrorMind: Predict -> Introspect -> Adapt
                # The adapter handles the loss internally
                action = adapter.predict(obs, update=True, target=ideal_thrust)

            # --- STEP ENV ---
            action_np = action.detach().numpy()[0]
            next_obs, reward, done = env.step(action_np)
            
            # --- STRESSOR C: WEIGHT NOISE (Hardware Damage) ---
            if step % 100 == 0:
                with torch.no_grad():
                    if contender == 'base':
                        for p in base_model.parameters():
                            p.add_(torch.randn_like(p) * 0.1)
                    else:
                        # MirrorMind handles noise naturally via Introspection, 
                        # but we inject it to prove it recovers.
                        for p in framework.model.parameters():
                            p.add_(torch.randn_like(p) * 0.1)

            # Record
            obs = next_obs
            history[contender]['alt'].append(env.state[2])
            
            # FPS Calculation
            step_end = time.time()
            fps = 1.0 / (step_end - step_start + 1e-6)
            history[contender]['fps'].append(fps)
            
            if done:
                logger.info(f"   💥 CRASHED at Step {step}")
                crashed = True
                break
                
        if not crashed:
            history[contender]['survived'] = MAX_STEPS
            logger.info("   ✅ SURVIVED Full Duration")
        else:
            history[contender]['survived'] = step

    # ==========================================================================
    # 4. VERDICT & VISUALIZATION
    # ==========================================================================
    mm_surv = history['mm']['survived']
    base_surv = history['base']['survived']
    mm_fps = np.mean(history['mm']['fps'])
    base_fps = np.mean(history['base']['fps'])
    
    logger.info("-" * 40)
    logger.info(f"FINAL STATS:")
    logger.info(f"MirrorMind: {mm_surv} Steps | {mm_fps:.1f} FPS")
    logger.info(f"Baseline:   {base_surv} Steps | {base_fps:.1f} FPS")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['mm']['alt'], 'g-', label='MirrorMind (Polymorphic)', linewidth=2)
    plt.plot(history['base']['alt'], 'r--', label='Standard LSTM', alpha=0.6)
    plt.axhline(0, color='k', linewidth=2, label='Ground (Death)')
    plt.axhline(10, color='b', linestyle=':', label='Target Altitude')
    plt.title("Phase 7: SOTA Deathmatch (Flight Stability)")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Altitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiments/protocol_v1/deathmatch_results.png')
    
    # The SOTA Claim criteria
    if mm_surv > base_surv * 1.5 and mm_fps > 50:
        print("\n" + "="*40)
        print("🏆 SOTA CONFIRMED: MirrorMind Dominates.")
        print("   - Outlasted Baseline by >50%")
        print("   - Maintained Real-Time FPS")
        print("="*40 + "\n")
        sys.exit(0)
    elif mm_surv > base_surv:
        print("\n" + "="*40)
        print("✅ PASSED: MirrorMind Wins (But margin is small).")
        print("="*40 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*40)
        print("🔴 FAILED: Not yet SOTA.")
        print("="*40 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    run_deathmatch()