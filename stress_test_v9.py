
import torch
import torch.nn as nn
import numpy as np
import time
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

def run_stress_test():
    print("="*60)
    print("AIRBORNE HRS V9.0 - STRESS TEST & STABILITY ANALYSIS")
    print("="*60)
    
    # --- CONFIG ---
    config = AdaptiveFrameworkConfig(
        model_dim=64,
        enable_perception=True,
        vision_dim=3,
        audio_dim=80,
        enable_world_model=False, # Focus on MoE/Memory
        use_moe=True,
        use_hierarchical_moe=True,
        use_graph_memory=True,
        num_domains=2,
        experts_per_domain=4,
        learning_rate=0.001
    )
    
    dummy_model = nn.Linear(64, 64)
    framework = AdaptiveFramework(dummy_model, config, device='cpu')
    
    print("\n[TEST 1] MoE Expert Collapse Analysis (500 Steps)")
    # ---------------------------------------------------------
    # Problem: Without load balancing, router picks same expert.
    # Metric: Variance of expert usage counts. High variance = Collapse.
    # ---------------------------------------------------------
    
    print("   Simulating diverse inputs (High Entropy)...")
    for _ in tqdm(range(500), desc="Training MoE"):
        # Generate random high-entropy inputs to encourage diversity
        inputs = {'vision': torch.randn(4, 3, 32, 32), 'audio': torch.randn(4, 80, 50)}
        framework.train_step(inputs, target_data=torch.randn(4, 64), enable_dream=False)
        
    usage = framework.model.get_expert_usage().flatten()
    usage_std = np.std(usage)
    usage_cv = usage_std / (np.mean(usage) + 1e-6)
    
    print(f"   [RESULT] Expert Usage: {usage}")
    print(f"   [METRIC] Coefficient of Variation (CV): {usage_cv:.2f}")
    
    collapse_verdict = "POSSIBLE COLLAPSE" if usage_cv > 0.5 else "BALANCED"
    print(f"   [VERDICT] {collapse_verdict}")
    
    # Plotting MoE Usage
    plt.figure(figsize=(10, 5))
    x_pos = np.arange(len(usage))
    plt.bar(x_pos, usage, color='skyblue', edgecolor='navy')
    plt.title(f'MoE Expert Usage Distribution (CV={usage_cv:.2f})\nVerdict: {collapse_verdict}')
    plt.xlabel('Expert Index')
    plt.ylabel('Activation Count')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('stress_test_moe.png')
    print(f"   [SAVED] Plot saved to {os.path.abspath('stress_test_moe.png')}")
    
    print("\n[TEST 2] Relational Memory Linear Scaling Analysis")
    # --------------------------------------------------------
    # Problem: O(N) scan. 
    # Metric: Latency per add operation as N grows.
    # --------------------------------------------------------
    
    if not framework.memory.graph_memory:
        print("   [SKIP] Graph memory disabled.")
        return

    print("   Injecting 2000 nodes...")
    latencies = []
    
    snapshot = type('Snapshot', (), {})()
    snapshot.embedding = torch.randn(64)
    
    # Warmup
    for _ in range(10): 
        framework.memory.graph_memory.add(snapshot, torch.randn(64))

    for i in tqdm(range(2000), desc="Filling Memory"):
        start_t = time.time()
        # Random vector
        framework.memory.graph_memory.add(snapshot, torch.randn(64))
        dt = (time.time() - start_t) * 1000 # ms
        latencies.append(dt)
        
    # Stats
    avg_latency_early = np.mean(latencies[:100])
    avg_latency_late = np.mean(latencies[-100:])
    scale_factor = avg_latency_late / avg_latency_early
    
    print(f"   [RESULT] Latency (First 100): {avg_latency_early:.3f} ms")
    print(f"   [RESULT] Latency (Last 100):  {avg_latency_late:.3f} ms")
    print(f"   [METRIC] Scaling Factor: {scale_factor:.2f}x")
    
    start_t = time.time()
    framework.memory.graph_memory.retrieve(torch.randn(64), k=5)
    retrieve_t = (time.time() - start_t) * 1000
    print(f"   [RESULT] Retrieval Time (N=2000): {retrieve_t:.3f} ms")
    
    scaling_verdict = "LINEAR SCALING (O(N)) DETECTED" if scale_factor > 1.5 else "CONSTANT TIME (O(1))"
    print(f"   [VERDICT] {scaling_verdict}")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(latencies, label='Add Latency (ms)', alpha=0.5)
    # Moving average
    ma = np.convolve(latencies, np.ones(50)/50, mode='valid')
    plt.plot(range(len(ma)), ma, label='Moving Avg', color='red')
    plt.title(f'Memory Scaling Analysis (O(N) Check)\nVerdict: {scaling_verdict}')
    plt.xlabel('Memory Size (Nodes)')
    plt.ylabel('Time per Add (ms)')
    plt.legend()
    plt.savefig('stress_test_memory.png')
    print(f"   [SAVED] Plot saved to {os.path.abspath('stress_test_memory.png')}")

if __name__ == "__main__":
    run_stress_test()
