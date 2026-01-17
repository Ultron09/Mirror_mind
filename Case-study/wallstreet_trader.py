
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Force local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# --- MARKET SIMULATOR ---
class MarketSimulator:
    def __init__(self, seq_len=10):
        self.seq_len = seq_len
        
    def generate_regime(self, mode='bull', steps=1000):
        # Generate price data
        t = np.linspace(0, 50, steps)
        noise = np.random.normal(0, 0.5, steps)
        
        if mode == 'bull':
            # Uptrend + Sine wave
            price = 100 + t + 5*np.sin(t) + noise
            label = 1 # BUY
        elif mode == 'bear':
            # Downtrend + Sine wave
            price = 150 - t + 5*np.sin(t) + noise
            label = 0 # SELL
            
        # Create sliding windows
        X, y = [], []
        for i in range(len(price) - self.seq_len):
            window = price[i:i+self.seq_len]
            # Normalize window
            window = (window - window.mean()) / (window.std() + 1e-5)
            X.append(window)
            y.append(label)
            
        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))
        return X, y

class TraderNet(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32), # Compressed knowledge
            nn.ReLU(),
            nn.Linear(32, 2) # Buy (1) / Sell (0)
        )
    def forward(self, x):
        return self.net(x)

def run_simulation():
    print(">>> INITIALIZING PROJECT WALLSTREET (Market Crash Sim)...")
    market = MarketSimulator()
    
    # 1. THE BULL RUN (2020-2021)
    print("Generating Phase 1: THE BULL RUN...")
    x_bull, y_bull = market.generate_regime('bull', steps=500)
    
    # 2. THE CRASH (2022)
    print("Generating Phase 2: THE CRASH...")
    x_bear, y_bear = market.generate_regime('bear', steps=500)
    
    # 3. THE RECOVERY (2024) - Test Set (Same dynamics as Bull)
    print("Generating Phase 3: THE RECOVERY (Test)...")
    x_rec, y_rec = market.generate_regime('bull', steps=200)

    systems = {
        'Legacy Algo': AdaptiveFrameworkConfig(memory_type='none', device='cpu', learning_rate=0.005),
        'Airborne Quint': AdaptiveFrameworkConfig(
            memory_type='hybrid', 
            ewc_lambda=5000.0, 
            dream_interval=10, 
            enable_consciousness=True,
            device='cpu',
            learning_rate=0.005
        )
    }
    
    perf_log = {}

    for name, cfg in systems.items():
        print(f"\n[{name}] TRADING DESK OPEN.")
        model = AdaptiveFramework(TraderNet(), cfg, device='cpu')
        
        # PHASE 1: BULL MARKET TRAINING
        print(f"[{name}] Trading the Bull Market...")
        for _ in range(10):
            model.train_step(x_bull, target_data=y_bull)
            
        if 'Airborne' in name:
            model.memory.consolidate(feedback_buffer=model.prioritized_buffer)
            
        # PHASE 2: BEAR MARKET ADAPTATION
        print(f"[{name}] CRASH DETECTED! Adapting to Bear Market...")
        for _ in range(10):
             model.train_step(x_bear, target_data=y_bear)
             
        # PHASE 3: RECOVERY (The Examination)
        # Does the model recognize the Bull pattern again?
        # Or is it stuck in "Bear Mode" (Shorting everything)?
        print(f"[{name}] RECOVERY DETECTED. Executing Trades...")
        
        model.model.eval()
        balance = 10000 # Starting Capital
        equity_curve = [balance]
        
        with torch.no_grad():
            out, _, _ = model(x_rec)
            signals = out.argmax(dim=1) # 1=Buy, 0=Sell
            
            # Simulate Trading on Recovery Data
            # Correct Action for Recovery is BUY (1). 
            # If Sell (0), we lose money.
            for s in signals:
                if s.item() == 1: # Long
                    balance *= 1.02 # Profit
                else: # Short
                    balance *= 0.98 # Loss
                equity_curve.append(balance)
                
        perf_log[name] = equity_curve
        final_pnl = ((balance - 10000)/10000)*100
        print(f"[{name}] Final PnL: {final_pnl:.1f}%")

    # --- VISUALIZATION ---
    print("\n>>> Generating PnL Report...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, curve in perf_log.items():
        color = '#2ecc71' if 'Airborne' in name else '#e74c3c'
        ax.plot(curve, label=f"{name}", linewidth=2, color=color)
        
    ax.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Portfolio Performance: Recovery Phase", color='white', fontsize=14)
    ax.set_xlabel("Trading Days", color='white')
    ax.set_ylabel("Account Balance ($)", color='white')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    save_path = os.path.join(os.path.dirname(__file__), 'portfolio_performance.png')
    plt.savefig(save_path)
    print(f"Graph saved: {save_path}")

if __name__ == "__main__":
    run_simulation()
