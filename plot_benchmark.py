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
        
    # SYNC: Match modes in benchmark_continual.py
    # User removed 'ewc', renamed 'naive' -> 'base'
    modes = ['base', 'airborne'] 
    mode_labels = ['Standard (Base)', 'AirborneHRS']
    colors = ['#95a5a6', '#2ecc71'] # Grey for Base, Green for Airborne
    
    seeds = list(data.keys())
    if not seeds:
        print("Results file is empty.")
        return

    # Aggregation
    metrics = {m: {'acc_a2': [], 'bwt': []} for m in modes}
    
    for seed in seeds:
        for mode in modes:
            if mode in data[seed]:
                res = data[seed][mode]
                metrics[mode]['acc_a2'].append(res['acc_task_a_phase2'])
                # BWT might be missing if only 1 task runs, handle safely
                bwt = res.get('bwt', 0.0)
                # If manual calc needed: A2 - A1
                if 'bwt' not in res:
                     a1 = res.get('acc_task_a_phase1', 0.0)
                     a2 = res.get('acc_task_a_phase2', 0.0)
                     bwt = a2 - a1
                metrics[mode]['bwt'].append(bwt)
            else:
                print(f"Warning: Seed {seed} missing mode {mode}")

    # Stats
    means_acc = []
    stds_acc = []
    means_bwt = []
    stds_bwt = []
    
    valid_modes = [] # Only plot modes that actually have data
    valid_labels = []
    valid_colors = []

    for i, m in enumerate(modes):
        if metrics[m]['acc_a2']:
            means_acc.append(np.mean(metrics[m]['acc_a2']))
            stds_acc.append(np.std(metrics[m]['acc_a2']))
            means_bwt.append(np.mean(metrics[m]['bwt']))
            stds_bwt.append(np.std(metrics[m]['bwt']))
            valid_modes.append(m)
            valid_labels.append(mode_labels[i])
            valid_colors.append(colors[i])
    
    if not valid_modes:
        print("No valid data found to plot.")
        return

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    bars = plt.bar(valid_labels, means_acc, yerr=stds_acc, capsize=10, 
                   color=valid_colors, alpha=0.9, width=0.6)
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
    plt.figure(figsize=(8, 6))
    bars = plt.bar(valid_labels, means_bwt, yerr=stds_bwt, capsize=10, 
                   color=valid_colors, alpha=0.9, width=0.6)
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
