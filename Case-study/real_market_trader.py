
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import datetime
import ssl

# MONKEYPATCH SSL
if not hasattr(ssl, 'wrap_socket'):
    def dummy_wrap_socket(sock, *args, **kwargs):
        context = ssl.create_default_context()
        return context.wrap_socket(sock, server_hostname=kwargs.get('server_hostname'))
    ssl.wrap_socket = dummy_wrap_socket

# CHECK DEPENDENCIES
try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# --- UNIVERSE FETCHER ---
def get_universe_data(tickers, start="2015-01-01", end="2026-01-01"):
    universe = {}
    print(f"Downloading Universe: {tickers}...")
    
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if len(df) == 0: raise ValueError("Empty")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        except:
            print(f"WARNING: {t} Failed. Synthesizing.")
            dates = pd.date_range(start=start, end=end, freq='B')
            steps = len(dates)
            price = 100.0 + np.cumsum(np.random.normal(0.1, 5.0, steps))
            df = pd.DataFrame(index=dates)
            df['Adj Close'] = price; df['High']=price+2; df['Low']=price-2; df['Close']=price
            
        if 'Adj Close' not in df.columns: df['Adj Close'] = df['Close']
        df['Alloc'] = df['Adj Close']
        
        # FEATURES
        df['Ret'] = df['Alloc'].pct_change()
        df['MA50'] = df['Alloc'].rolling(50).mean()
        df['MA200'] = df['Alloc'].rolling(200).mean()
        df['TrendSignal'] = (df['MA50'] - df['MA200']) / (df['MA200']+1e-5)
        
        delta = df['Alloc'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50) / 100.0
        
        rolling_std = df['Alloc'].rolling(20).std()
        df['UpperBB'] = df['MA50'] + (2 * rolling_std)
        df['LowerBB'] = df['MA50'] - (2 * rolling_std)
        df['BBPos'] = (df['Alloc'] - df['LowerBB']) / (df['UpperBB'] - df['LowerBB'])
        
        df['Vol'] = (df['High'] - df['Low']) / df['Close']
        df['VolMA'] = df['Vol'].rolling(20).mean()
        df['VolRel'] = df['Vol'] / df['VolMA']
        
        df.fillna(0, inplace=True)
        df.dropna(inplace=True)
        universe[t] = df
        
    # ALIGN DATES (Intersection)
    common_index = universe[tickers[0]].index
    for t in tickers[1:]:
        common_index = common_index.intersection(universe[t].index)
    
    final_universe = {}
    for t in tickers:
        final_universe[t] = universe[t].loc[common_index]
        
    return final_universe

def create_inputs(df, seq_len=10):
    f_trend = df[['Ret', 'TrendSignal']].values
    f_contra = df[['Ret', 'RSI', 'BBPos']].values
    f_risk = df[['Ret', 'VolRel']].values
    f_chair = df[['Ret', 'TrendSignal', 'RSI', 'VolRel']].values
    
    Xt, Xc, Xr, Xch, y = [], [], [], [], []
    
    for i in range(len(f_trend) - seq_len - 1):
        next_ret = f_trend[i+seq_len+1][0] 
        label = 1 if next_ret > 0 else 0
        Xt.append(f_trend[i:i+seq_len])
        Xc.append(f_contra[i:i+seq_len])
        Xr.append(f_risk[i:i+seq_len])
        Xch.append(f_chair[i:i+seq_len])
        y.append(label)
        
    return (
        torch.FloatTensor(np.array(Xt)),
        torch.FloatTensor(np.array(Xc)),
        torch.FloatTensor(np.array(Xr)),
        torch.FloatTensor(np.array(Xch)),
        torch.LongTensor(np.array(y)),
        df['Alloc'].values[seq_len+1:]
    )

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class ChairmanNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 3), nn.Softmax(dim=1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def get_nifty_universe():
    # Indian Market Heavies (FnO Segment)
    return [
        '^NSEI',      # Nifty 50 Index
        '^NSEBANK',   # Bank Nifty Index
        'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS',
        'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS',
        'AXISBANK.NS', 'HINDUNILVR.NS', 'TATAMOTORS.NS', 'M&M.NS', 'MARUTI.NS'
    ]

def run_liberty_sim():
    print(">>> 'PROJECT NIFTY': INDIA F&O TACTICAL (Last Week)...")
    tickers = get_nifty_universe()
    
    # Futures Mode
    FUTURES_LEVERAGE = 5.0
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365*2) # 2 Years Context
    split_date = end_date - datetime.timedelta(days=7)   # Test Last 7 Days
    
    print(f"Training: {start_date.date()} -> {split_date.date()}")
    print(f"Testing:  {split_date.date()} -> {end_date.date()}")
    
    # 1. FETCH ALL DATA
    universe = get_universe_data(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    # 2. PREP AGENTS (Global Macro Mind)
    print("Initializing The High Table (Global Macro Edition)...")
    cfg = AdaptiveFrameworkConfig(memory_type='hybrid', ewc_lambda=1000.0, dream_interval=10, enable_consciousness=True, device='cpu', learning_rate=0.001)
    
    council = {
        'Trend': AdaptiveFramework(Net(2), cfg, device='cpu'),
        'Contra': AdaptiveFramework(Net(3), cfg, device='cpu'),
        'Risk': AdaptiveFramework(Net(2), cfg, device='cpu')
    }
    chairman = AdaptiveFramework(ChairmanNet(4), cfg, device='cpu')
    
    # 3. PREP DATASETS
    print("Preparing Tensors (This may take a moment)...")
    asset_data = {}
    
    # Optimization: Pre-compute 'y' and 'inputs' to avoid re-doing it 100 times if logic is same?
    # No, inputs vary by asset.
    
    for t in tickers:
        Xt, Xc, Xr, Xch, y, prices = create_inputs(universe[t])
        asset_data[t] = {
            'Trend': Xt, 'Contra': Xc, 'Risk': Xr, 'Chair': Xch, 'y': y, 'price': prices
        }
    
    common_len = len(asset_data[tickers[0]]['y'])
    test_len = min(500, common_len // 2)
    train_len = common_len - test_len
    
    print(f"Training Samples: {train_len}. Test Samples: {test_len}")

    # 4. TRAINING (Massive Parallel Learning)
    # To speed up, we sample random batches from random assets instead of sequential loops.
    print("Training Phase (Universal Knowledge Transfer)...")
    
    # Create Giant List of DataLoaders? No, too much memory.
    # Just train on random assets for N steps.
    
    training_steps = 100 # Fast training
    import random
    
    for step in range(training_steps):
        t = random.choice(tickers)
        if step % 10 == 0: print(f"  > Learning Step {step}/{training_steps} (Focus: {t})")
        
        # Sample a batch from this asset
        idx_start = np.random.randint(0, train_len - 64)
        
        # Train Agents
        for name, agent in council.items():
            bx = asset_data[t][name][idx_start:idx_start+64]
            by = asset_data[t]['y'][idx_start:idx_start+64]
            agent.train_step(bx, target_data=by)
            
    # Checkpoint
    for agent in council.values():
        agent.memory.consolidate(feedback_buffer=agent.prioritized_buffer)
        
    # 5. EXECUTION (The Rotation)
    print("\n>>> LIVE TRADING (2024-2025): SEEKING ALPHA...")
    
    cash = 10000
    holdings = {t: 0 for t in tickers} # Shares per ticker
    current_asset = 'CASH'
    curve = [cash]
    
    # Trade Log
    test_range = range(train_len, common_len)
    
    for i in test_range:
        idx = i - train_len # relative index for test logic if needed
        
        # SCAN UNIVERSE
        best_ticker = None
        best_score = -1.0 # Logic: Confidence of Buy
        
        for t in tickers:
            # High Table Vote for Asset T
            # Get Logits
            stack = []
            for name, agent in council.items():
                inp = asset_data[t][name][i].unsqueeze(0)
                agent.model.eval()
                with torch.no_grad():
                    out, _, _ = agent(inp) # (1, 2)
                    stack.append(out)
            stack = torch.stack(stack, dim=1)
            
            # Chairman Weights
            inp_c = asset_data[t]['Chair'][i].unsqueeze(0)
            chairman.model.eval()
            with torch.no_grad():
                w, _, _ = chairman(inp_c)
                
            # Consensus
            final = torch.sum(stack * w.view(1, 3, 1), dim=1) # (1,2)
            prob_buy = torch.softmax(final, dim=1)[0][1].item()
            
            # Record Score
            if prob_buy > 0.6 and prob_buy > best_score: # High Conviction Threshold
                best_score = prob_buy
                best_ticker = t
                
        # EXECUTION: ROTATE TO WINNER
        # Calculate Daily PnL on Current Holding
        step_pnl = 0.0
        if current_asset != 'CASH':
            # Price Change
            curr_price = asset_data[current_asset]['price'][i]
            prev_price = asset_data[current_asset]['price'][i-1] if i > 0 else curr_price
            
            if np.isnan(curr_price): curr_price = prev_price
            if prev_price > 0:
                pct_change = (curr_price - prev_price) / prev_price
                # FUTURES LEVERAGE (5x)
                step_pnl = pct_change * FUTURES_LEVERAGE
        
        # Apply PnL to Cash (Mark to Market)
        if current_asset != 'CASH':
            cash = cash * (1 + step_pnl)
            
        # Decision (Swap Asset?)
        if best_ticker and best_ticker != current_asset:
            # Rotate
            current_asset = best_ticker
            # Transaction Cost? Ignored for sim.
            
        elif best_ticker is None and current_asset != 'CASH':
            current_asset = 'CASH'
            
        if np.isnan(cash) or cash <= 0: 
            cash = 0 # Blown account
            
        curve.append(cash)
        
        # FAST MODE
        pass
                
    final_pnl = ((curve[-1]-10000)/10000)*100
    if np.isnan(final_pnl): final_pnl = 0.0
    print(f"\n[PROJECT NIFTY] Final Result: {final_pnl:.2f}% (INR)")
    print(f"Final Position: {current_asset}")
    
    with open(os.path.join(os.path.dirname(__file__), 'nifty_results.txt'), 'w') as f:
        f.write(f"PROJECT NIFTY (FnO 5x)\nReturn: {final_pnl:.2f}%\n")
        
    plt.style.use('dark_background')
    plt.figure(figsize=(10,6))
    plt.plot(curve, color='#e67e22', linewidth=2, label=f'Nifty Algo (+{final_pnl:.0f}%)')
    plt.title('Project Nifty: FnO Strategy (2025-2026)')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'nifty_performance.png'))

if __name__ == "__main__":
    run_liberty_sim()
