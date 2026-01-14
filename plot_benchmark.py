import json
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_FILE = "benchmark_results.json"

def plot_benchmark():
    if not os.path.exists(RESULTS_FILE):
        print("No results file found. Run benchmark_continual.py first.")
        return

    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
        
    modes = ['naive', 'ewc', 'airborne']
    seeds = list(data.keys())
    
    # Aggregation
    metrics = {m: {'acc_a2': [], 'bwt': []} for m in modes}
    
    for seed in seeds:
        for mode in modes:
            if mode in data[seed]:
                res = data[seed][mode]
                metrics[mode]['acc_a2'].append(res['acc_task_a_phase2'])
                metrics[mode]['bwt'].append(res['bwt'])
    
    # Stats
    means_acc = [np.mean(metrics[m]['acc_a2']) for m in modes]
    stds_acc = [np.std(metrics[m]['acc_a2']) for m in modes]
    
    means_bwt = [np.mean(metrics[m]['bwt']) for m in modes]
    stds_bwt = [np.std(metrics[m]['bwt']) for m in modes]
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modes, means_acc, yerr=stds_acc, capsize=10, 
                   color=['#bdc3c7', '#3498db', '#2ecc71'], alpha=0.9)
    plt.ylabel('Task A Retention (%)', fontsize=12)
    plt.title('Catastrophic Forgetting Analysis (Higher is Better)', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
                 
    plt.savefig('paper_results_acc.png', dpi=300)
    print("Saved paper_results_acc.png")
    
    # Plot BWT
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modes, means_bwt, yerr=stds_bwt, capsize=10, 
                   color=['#bdc3c7', '#3498db', '#2ecc71'], alpha=0.9)
    plt.ylabel('Backward Transfer (Accuracy Change)', fontsize=12)
    plt.title('Stability Analysis (Zero is Ideal)', fontsize=14, fontweight='bold')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        offset = 2 if height >=0 else -5
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.savefig('paper_results_bwt.png', dpi=300)
    print("Saved paper_results_bwt.png")

if __name__ == "__main__":
    plot_benchmark()
