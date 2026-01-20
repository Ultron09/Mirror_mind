
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TEST-ABCD")

def generate_data(task_id, batch_size=32):
    x = torch.randn(batch_size, 10)
    # Define 4 distinct, conflicting tasks on the same input space
    if task_id == 'A':
        y = x.sum(dim=1, keepdim=True)                   # Sum all
    elif task_id == 'B':
        y = -x.sum(dim=1, keepdim=True)                  # Neg sum
    elif task_id == 'C':
        y = (x[:, ::2]).sum(dim=1, keepdim=True)         # Sum evens
    elif task_id == 'D':
        y = (x[:, 1::2]).sum(dim=1, keepdim=True)        # Sum odds
    return x, y

def train_until_convergence(agent, task_id, threshold=0.1, max_steps=500):
    logger.info(f"--- Training Task {task_id} until Loss < {threshold} ---")
    
    for step in range(max_steps):
        x, y = generate_data(task_id)
        metrics = agent.train_step(x, target_data=y)
        loss = metrics['loss']
        
        if step % 50 == 0:
            logger.info(f"Task {task_id} Step {step}: Loss {loss:.4f}")
            
        if loss < threshold:
            logger.info(f"✅ Task {task_id} Converged at Step {step} (Loss {loss:.4f})")
            return step
            
    logger.warning(f"⚠️ Task {task_id} reached max steps without strict convergence (Final: {loss:.4f})")
    return max_steps

def evaluate_all(agent, tasks):
    results = {}
    with torch.no_grad():
        for t in tasks:
            # Generate valid test set
            x, y = generate_data(t, batch_size=100) 
            # Handle tuple output from AdaptiveFramework
            output = agent(x)
            pred = output[0] if isinstance(output, tuple) else output
            loss = nn.MSELoss()(pred, y).item()
            results[t] = loss
    return results

def run_abcd_test():
    # 1. Setup with Strong Memory
    base_model = nn.Sequential(
        nn.Linear(10, 64), 
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(), 
        nn.Linear(64, 1)
    )
    
    cfg = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='hybrid',
        ewc_lambda=1000.0, # Balanced for Multitask
        dream_interval=1,
        enable_consciousness=True
    )
    agent = AdaptiveFramework(base_model, cfg, device='cpu')
    
    tasks = ['A', 'B', 'C', 'D']
    history = {t: [] for t in tasks}
    timeline_steps = []
    current_total_steps = 0
    
    # 2. Sequential Learning Loop
    for t_idx, current_task in enumerate(tasks):
        # Train
        steps_taken = train_until_convergence(agent, current_task, threshold=1.0) # threshold 1.0 is aggressive for high variance data
        
        # Eval all tasks (including future ones? no, mostly past)
        # We eval ALL to show zero-shot vs retention
        res = evaluate_all(agent, tasks)
        
        for t in tasks:
            history[t].append(res[t])
            
        current_total_steps += steps_taken
        timeline_steps.append(current_task)
        
        # Force huge consolidation after task switch
        if agent.prioritized_buffer:
            logger.info(f"💾 Consolidating Memory after Task {current_task}...")
            agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')

    # 3. Print Final Report
    print("\n📊 Final Retention Report:")
    for t in tasks:
        print(f"Task {t} Loss: {history[t][-1]:.4f}")
        
    # 4. Plot
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    for i, t in enumerate(tasks):
        plt.plot(tasks, history[t], marker=markers[i], label=f'Task {t} Error', linewidth=2)
    
    plt.title("AirborneHRS Multi-Task Retention (A->B->C->D)")
    plt.xlabel("Training Phase (After Task X)")
    plt.ylabel("MSE Loss (Lower is Better)")
    plt.legend()
    plt.grid(True)
    plt.savefig("tests/retention_plot.png")
    print("✅ Plot saved to tests/retention_plot.png")

if __name__ == "__main__":
    run_abcd_test()
