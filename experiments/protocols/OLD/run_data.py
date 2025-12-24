"""
MIRRORMIND BENCHMARK: Mathematical Reasoning & Code Tasks (V8)
==============================================================
Production-grade suite with diverse, stable evaluation tasks.

Dataset: GSM8K (Grade School Math) + MBPP (Python Code Problems)
- 📐 GSM8K: 8,000+ grade school math word problems
- 💻 MBPP: 1,000 basic Python programming tasks
- ✅ Stable, well-formatted, reliable benchmarks

Key Features:
- 🚀 PERFORMANCE: Batch processing, memory optimization
- 📊 MODERN DASHBOARD: Multi-metric real-time visualization
- 💾 EFFICIENT I/O: Buffered saves, reduced disk operations
- 🧠 Dual-Task Evaluation: Math reasoning + code generation
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import pandas as pd
import logging
import sys
from collections import deque
from typing import Optional, Tuple, List, Dict
import re

# Configure plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['grid.color'] = '#333333'

# Windows encoding fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Optimized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("mirrormind_run.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
for logger_name in ["transformers", "datasets", "matplotlib"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

try:
    from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
    from airbornehrs.meta_controller import MetaController, MetaControllerConfig
except ImportError:
    logging.error("CRITICAL: 'airbornehrs' not found.")
    exit(1)


# ==============================================================================
# ENHANCED LIVE VISUALIZER WITH TASK TYPE BREAKDOWN
# ==============================================================================
class EnhancedLiveVisualizer:
    """
    Advanced real-time visualization with task-type breakdown:
    - Separate tracking for Math vs Code tasks
    - Performance comparison across domains
    - Modern dark theme
    """
    
    def __init__(self, total_tasks: int, window_size: int = 50):
        self.total = total_tasks
        self.window_size = window_size
        
        # Data storage
        self.task_ids = []
        self.task_types = []  # 'math' or 'code'
        self.base_losses = []
        self.mm_losses = []
        self.improvements = []
        self.win_rates = []
        self.rolling_avg = deque(maxlen=window_size)
        
        # Per-type tracking
        self.math_wins = 0
        self.math_total = 0
        self.code_wins = 0
        self.code_total = 0
        self.wins = 0
        
        # Setup figure with modern layout
        plt.ion()
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.35, wspace=0.3)
        
        # Main plots
        self.ax_loss = self.fig.add_subplot(gs[0, :3])      # Top: Loss comparison
        self.ax_type = self.fig.add_subplot(gs[0, 3])       # Top right: Type breakdown
        self.ax_imp = self.fig.add_subplot(gs[1, :3])       # Middle: Improvements
        self.ax_dist = self.fig.add_subplot(gs[1, 3])       # Middle right: Distribution
        self.ax_win = self.fig.add_subplot(gs[2, 0])        # Bottom: Win rate
        self.ax_rolling = self.fig.add_subplot(gs[2, 1])    # Bottom: Rolling avg
        self.ax_compare = self.fig.add_subplot(gs[2, 2])    # Bottom: Math vs Code
        self.ax_stats = self.fig.add_subplot(gs[2, 3])      # Bottom: Statistics
        
        self._setup_axes()
        
        print("\n" + "="*70)
        print("🚀 ENHANCED VISUALIZATION SYSTEM INITIALIZED")
        print("   ├─ Multi-domain tracking (Math + Code)")
        print("   ├─ Real-time performance comparison")
        print("   ├─ Auto-saving to 'mirrormind_dashboard.png'")
        print("   └─ Memory-optimized rendering")
        print("="*70 + "\n")
    
    def _setup_axes(self):
        """Configure axis properties"""
        self.ax_loss.set_title("Loss Trajectory: Base vs MirrorMind", 
                              fontsize=14, fontweight='bold', color='#00ff88')
        self.ax_loss.set_ylabel("Loss", fontsize=10)
        self.ax_loss.grid(True, alpha=0.2)
        
        self.ax_type.set_title("Task Mix", fontsize=10, fontweight='bold')
        self.ax_type.axis('off')
        
        self.ax_imp.set_title("Performance Improvement per Task", 
                             fontsize=14, fontweight='bold', color='#00ff88')
        self.ax_imp.set_ylabel("Improvement (%)", fontsize=10)
        self.ax_imp.axhline(0, color='white', linewidth=1, alpha=0.5)
        
        self.ax_dist.set_title("Improvement\nDistribution", fontsize=10, fontweight='bold')
        
        self.ax_win.set_title("Cumulative Win Rate", fontsize=10, fontweight='bold')
        self.ax_win.set_ylabel("Win Rate (%)", fontsize=10)
        self.ax_win.set_ylim(0, 100)
        
        self.ax_rolling.set_title("Rolling Avg", fontsize=10, fontweight='bold')
        self.ax_rolling.set_ylabel("Avg Imp (%)", fontsize=10)
        
        self.ax_compare.set_title("Math vs Code", fontsize=10, fontweight='bold')
        self.ax_compare.set_ylabel("Win Rate (%)", fontsize=10)
        
        self.ax_stats.axis('off')
        self.ax_stats.set_title("Statistics", fontsize=10, fontweight='bold')
    
    def update(self, task_id: int, task_type: str, base_loss: float, mm_loss: float):
        """Update all visualizations with new data"""
        # Store data
        self.task_ids.append(task_id)
        self.task_types.append(task_type)
        self.base_losses.append(base_loss)
        self.mm_losses.append(mm_loss)
        
        # Calculate improvement (clamped for stability)
        imp = ((base_loss - mm_loss) / (base_loss + 1e-6)) * 100
        imp = np.clip(imp, -100, 100)
        self.improvements.append(imp)
        self.rolling_avg.append(imp)
        
        # Update win stats
        is_win = mm_loss < base_loss
        if is_win:
            self.wins += 1
            
        if task_type == 'math':
            self.math_total += 1
            if is_win:
                self.math_wins += 1
        else:
            self.code_total += 1
            if is_win:
                self.code_wins += 1
        
        win_rate = (self.wins / len(self.task_ids)) * 100
        self.win_rates.append(win_rate)
        
        # Update plots (every 2 tasks for performance)
        if len(self.task_ids) % 2 == 0:
            self._render_all_plots()
    
    def _render_all_plots(self):
        """Render all plot components"""
        # 1. Loss trajectory
        self.ax_loss.clear()
        display_range = slice(-100, None)
        x_data = self.task_ids[display_range]
        
        self.ax_loss.plot(x_data, self.base_losses[display_range], 
                         color='#ff4444', linewidth=2, label='Base Model', alpha=0.8)
        self.ax_loss.plot(x_data, self.mm_losses[display_range], 
                         color='#44ff44', linewidth=2, label='MirrorMind', alpha=0.8)
        self.ax_loss.fill_between(x_data, self.base_losses[display_range], 
                                 self.mm_losses[display_range], alpha=0.2, color='cyan')
        self.ax_loss.set_ylabel("Loss", fontsize=10, color='white')
        self.ax_loss.set_xlabel("Task ID", fontsize=10, color='white')
        self.ax_loss.legend(loc='upper right', framealpha=0.3)
        self.ax_loss.grid(True, alpha=0.2)
        self.ax_loss.set_facecolor('#1a1a1a')
        
        # 2. Task type pie chart
        self.ax_type.clear()
        self.ax_type.axis('off')
        if self.math_total > 0 or self.code_total > 0:
            sizes = [self.math_total, self.code_total]
            labels = [f'Math\n{self.math_total}', f'Code\n{self.code_total}']
            colors = ['#ff6b9d', '#4ecdc4']
            wedges, texts , autotexts = self.ax_type.pie(sizes, labels=labels, colors=colors, 
                                             autopct='%1.1f%%', startangle=90,
                                             textprops={'color': 'white', 'fontsize': 9})
        
        # 3. Improvement bars with color coding by type
        self.ax_imp.clear()
        display_range = slice(-50, None)
        x_imp = self.task_ids[display_range]
        y_imp = self.improvements[display_range]
        types = self.task_types[display_range]
        
        colors = []
        for y, t in zip(y_imp, types):
            if y > 0:
                colors.append('#00ff88' if t == 'math' else '#00ccff')
            else:
                colors.append('#ff4444' if t == 'math' else '#ff8844')
        
        self.ax_imp.bar(x_imp, y_imp, color=colors, alpha=0.7, width=0.8)
        self.ax_imp.axhline(0, color='white', linewidth=1, alpha=0.5)
        self.ax_imp.set_ylabel("Improvement (%)", fontsize=10, color='white')
        self.ax_imp.set_xlabel("Task ID", fontsize=10, color='white')
        self.ax_imp.set_ylim(-60, 60)
        self.ax_imp.grid(True, alpha=0.2, axis='y')
        self.ax_imp.set_facecolor('#1a1a1a')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#00ff88', label='Math Win'),
            Patch(facecolor='#00ccff', label='Code Win'),
            Patch(facecolor='#ff4444', label='Math Loss'),
            Patch(facecolor='#ff8844', label='Code Loss')
        ]
        self.ax_imp.legend(handles=legend_elements, loc='upper right', 
                          framealpha=0.3, fontsize=7)
        
        # 4. Distribution histogram
        self.ax_dist.clear()
        if len(self.improvements) > 10:
            self.ax_dist.hist(self.improvements, bins=30, color='#00aaff', 
                            alpha=0.7, edgecolor='white', linewidth=0.5)
            self.ax_dist.axvline(np.mean(self.improvements), color='#ff00ff', 
                               linewidth=2, linestyle='--', label='Mean')
            self.ax_dist.set_xlabel("Improvement (%)", fontsize=8, color='white')
            self.ax_dist.set_ylabel("Count", fontsize=8, color='white')
            self.ax_dist.legend(fontsize=7, framealpha=0.3)
        self.ax_dist.set_facecolor('#1a1a1a')
        
        # 5. Win rate curve
        self.ax_win.clear()
        self.ax_win.plot(self.task_ids, self.win_rates, color='#ff00ff', 
                        linewidth=2.5, marker='o', markersize=2, alpha=0.8)
        self.ax_win.fill_between(self.task_ids, 0, self.win_rates, 
                                alpha=0.2, color='purple')
        self.ax_win.set_ylim(0, 100)
        self.ax_win.set_ylabel("Win Rate (%)", fontsize=10, color='white')
        self.ax_win.set_xlabel("Tasks", fontsize=10, color='white')
        self.ax_win.grid(True, alpha=0.2)
        self.ax_win.set_facecolor('#1a1a1a')
        
        # 6. Rolling average
        self.ax_rolling.clear()
        if len(self.rolling_avg) > 5:
            rolling_data = list(self.rolling_avg)
            x_roll = self.task_ids[-len(rolling_data):]
            self.ax_rolling.plot(x_roll, rolling_data, color='#ffaa00', 
                               linewidth=2, alpha=0.8)
            self.ax_rolling.axhline(np.mean(rolling_data), color='cyan', 
                                   linewidth=1.5, linestyle='--', alpha=0.6)
            self.ax_rolling.set_ylabel("Avg Imp (%)", fontsize=10, color='white')
            self.ax_rolling.set_xlabel("Tasks", fontsize=10, color='white')
            self.ax_rolling.grid(True, alpha=0.2)
        self.ax_rolling.set_facecolor('#1a1a1a')
        
        # 7. Math vs Code comparison
        self.ax_compare.clear()
        math_wr = (self.math_wins / self.math_total * 100) if self.math_total > 0 else 0
        code_wr = (self.code_wins / self.code_total * 100) if self.code_total > 0 else 0
        
        bars = self.ax_compare.bar(['Math', 'Code'], [math_wr, code_wr], 
                                   color=['#ff6b9d', '#4ecdc4'], alpha=0.8)
        self.ax_compare.set_ylim(0, 100)
        self.ax_compare.set_ylabel("Win Rate (%)", fontsize=10, color='white')
        self.ax_compare.grid(True, alpha=0.2, axis='y')
        self.ax_compare.set_facecolor('#1a1a1a')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            self.ax_compare.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom',
                               color='white', fontsize=9, fontweight='bold')
        
        # 8. Statistics text
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        stats_text = f"""
Tasks: {len(self.task_ids)}/{self.total}
━━━━━━━━━━━━━━━
Overall:
  Wins: {self.wins}
  Rate: {self.win_rates[-1]:.1f}%
━━━━━━━━━━━━━━━
Math: {self.math_wins}/{self.math_total}
  Rate: {math_wr:.1f}%
━━━━━━━━━━━━━━━
Code: {self.code_wins}/{self.code_total}
  Rate: {code_wr:.1f}%
━━━━━━━━━━━━━━━
Avg: {np.mean(self.improvements):.2f}%
Best: {np.max(self.improvements):.2f}%
        """
        
        self.ax_stats.text(0.05, 0.5, stats_text, fontsize=9, 
                          family='monospace', verticalalignment='center',
                          color='#00ff88', weight='bold')
        
        # Refresh display
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if len(self.task_ids) % 5 == 0:
                plt.savefig("mirrormind_dashboard.png", dpi=150, 
                          facecolor='#0a0a0a', edgecolor='none')
        except Exception as e:
            logging.debug(f"Render error: {e}")
    
    def finalize(self):
        """Final save and display"""
        self._render_all_plots()
        plt.savefig("mirrormind_dashboard.png", dpi=200, 
                   facecolor='#0a0a0a', edgecolor='none')
        plt.ioff()
        plt.show()


# ==============================================================================
# OPTIMIZED MODELS & FRAMEWORK
# ==============================================================================
class OptimizedLanguageFramework(AdaptiveFramework):
    """Memory-efficient framework with gradient accumulation"""
    
    def __init__(self, config, device, use_amp=True):
        super().__init__(config, device)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
    def train_step(self, input_data, target):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=self.use_amp):
            output, log_var, internals = self.model(input_data, return_internals=True)
            
            shift_logits = output[..., :-1, :].contiguous()
            shift_labels = target[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
            nll = nll.view(shift_labels.size())
            
            mask = (shift_labels != -100).float()
            log_var = torch.clamp(log_var, min=-5.0, max=5.0)
            precision = torch.exp(-log_var)
            
            raw_loss = (nll * precision + 0.5 * log_var) * mask
            weighted_loss = raw_loss.sum() / (mask.sum() + 1e-6)
        
        if self.scaler:
            self.scaler.scale(weighted_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        
        self.loss_history.append(weighted_loss.item())
        self.step_count += 1
        
        return {'loss': weighted_loss.item()}


class IntrospectiveGPT2(nn.Module):
    """Optimized GPT2 with uncertainty estimation"""
    
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.backbone = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.backbone.config
        
        self.uncertainty_head = nn.Linear(self.config.n_embd, 1)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.zeros_(self.uncertainty_head.bias)

    def forward(self, input_ids, return_internals=False):
        outputs = self.backbone(input_ids, output_hidden_states=True, 
                               return_dict=True)
        log_var = self.uncertainty_head(outputs.hidden_states[-1]).mean(dim=1, keepdim=True)
        
        internals = {}
        if return_internals:
            internals['introspection'] = log_var
            internals['layer_last'] = outputs.hidden_states[-1].detach()
        
        return outputs.logits, log_var, internals


# ==============================================================================
# DATA PROCESSING FOR MATH AND CODE TASKS
# ==============================================================================
def prepare_math_task(example, tokenizer, max_len=1024) -> Optional[Tuple]:
    """Prepare GSM8K math problem"""
    try:
        question = example['question'].strip()
        answer = example['answer'].strip()
        
        # Format as few-shot learning
        prompt = f"Question: {question}\nAnswer:"
        full_text = f"{prompt} {answer}"
        
        # Tokenize
        prompt_ids = tokenizer.encode(prompt, truncation=True, max_length=max_len)
        full_ids = tokenizer.encode(full_text, truncation=True, max_length=max_len)
        
        if len(full_ids) <= len(prompt_ids) + 5:
            return None
        
        return (torch.tensor([prompt_ids]), 
                torch.tensor([full_ids]), 
                len(prompt_ids),
                'math')
    except Exception as e:
        logging.debug(f"Math task prep error: {e}")
        return None


def prepare_code_task(example, tokenizer, max_len=1024) -> Optional[Tuple]:
    """Prepare MBPP code problem"""
    try:
        text = example['text'].strip()
        code = example['code'].strip()
        
        # Format as code completion
        prompt = f"Problem: {text}\nSolution:\n"
        full_text = f"{prompt}{code}"
        
        # Tokenize
        prompt_ids = tokenizer.encode(prompt, truncation=True, max_length=max_len)
        full_ids = tokenizer.encode(full_text, truncation=True, max_length=max_len)
        
        if len(full_ids) <= len(prompt_ids) + 5:
            return None
        
        return (torch.tensor([prompt_ids]), 
                torch.tensor([full_ids]), 
                len(prompt_ids),
                'code')
    except Exception as e:
        logging.debug(f"Code task prep error: {e}")
        return None


# ==============================================================================
# MAIN BENCHMARK RUNNER
# ==============================================================================
def run_mirrormind_benchmark():
    """Execute benchmark on math and code tasks"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load tokenizer
    print("📥 Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print("📦 Loading GSM8K (Math) dataset...")
    try:
        math_dataset = load_dataset("gsm8k", "main", split="train")
        math_dataset = math_dataset.select(range(min(200, len(math_dataset))))
        print(f"   ✅ Loaded {len(math_dataset)} math problems")
    except Exception as e:
        logging.error(f"Failed to load GSM8K: {e}")
        math_dataset = []
    
    print("📦 Loading MBPP (Code) dataset...")
    try:
        code_dataset = load_dataset("mbpp", split="train")
        code_dataset = code_dataset.select(range(min(200, len(code_dataset))))
        print(f"   ✅ Loaded {len(code_dataset)} code problems")
    except Exception as e:
        logging.error(f"Failed to load MBPP: {e}")
        code_dataset = []
    
    if not math_dataset and not code_dataset:
        print("❌ No datasets loaded. Exiting.")
        return
    
    # Interleave datasets for mixed evaluation
    total_tasks = len(math_dataset) + len(code_dataset)
    
    # Initialize models
    print("🤖 Initializing Base Model...")
    base_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    base_model.eval()
    
    print("🧠 Initializing MirrorMind Framework...")
    fw_config = AdaptiveFrameworkConfig(
        learning_rate=1e-5,
        adaptation_threshold=0.01,
        compile_model=False
    )
    framework = OptimizedLanguageFramework(fw_config, device=device)
    framework.model = IntrospectiveGPT2('gpt2').to(device)
    
    controller = MetaController(
        framework, 
        MetaControllerConfig(base_lr=1e-5, max_lr=5e-5, use_reptile=True)
    )
    
    # Initialize visualizer
    visualizer = EnhancedLiveVisualizer(total_tasks)
    
    results = []
    save_buffer = []
    
    print(f"\n⚡ Starting benchmark on {total_tasks} tasks...\n")
    
    # Process tasks in alternating fashion
    task_counter = 0
    math_idx = 0
    code_idx = 0
    
    while math_idx < len(math_dataset) or code_idx < len(code_dataset):
        # Alternate between math and code
        if task_counter % 2 == 0 and math_idx < len(math_dataset):
            task = math_dataset[math_idx]
            task_data = prepare_math_task(task, tokenizer)
            math_idx += 1
        elif code_idx < len(code_dataset):
            task = code_dataset[code_idx]
            task_data = prepare_code_task(task, tokenizer)
            code_idx += 1
        elif math_idx < len(math_dataset):
            task = math_dataset[math_idx]
            task_data = prepare_math_task(task, tokenizer)
            math_idx += 1
        else:
            break
        
        if task_data is None:
            task_counter += 1
            continue
        
        adapt_tokens, eval_tokens, mask_len, task_type = task_data
        adapt_tokens = adapt_tokens.to(device)
        eval_tokens = eval_tokens.to(device)
        
        eval_labels = eval_tokens.clone()
        eval_labels[:, :mask_len] = -100
        
        # Evaluate base model
        with torch.no_grad():
            try:
                base_loss = base_model(eval_tokens, labels=eval_labels).loss.item()
                if np.isnan(base_loss) or np.isinf(base_loss):
                    continue
            except:
                continue
        
        # Reset and train MirrorMind
        framework.optimizer = AdamW(framework.model.parameters(), lr=1e-5)
        framework.model.train()
        
        try:
            for _ in range(3):
                metrics = framework.train_step(adapt_tokens, adapt_tokens)
                controller.adapt(metrics['loss'])
        except Exception as e:
            logging.debug(f"Training error: {e}")
            continue
        
        # Evaluate MirrorMind
        framework.model.eval()
        with torch.no_grad():
            try:
                mm_logits, _, _ = framework.model(eval_tokens)
                shift_logits = mm_logits[..., :-1, :].contiguous()
                shift_labels = eval_labels[..., 1:].contiguous()
                
                mm_loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                ).item()
                
                if np.isnan(mm_loss) or np.isinf(mm_loss):
                    continue
            except:
                continue
        
        # Update visualization
        visualizer.update(task_counter, task_type, base_loss, mm_loss)
        
        # Log results
        imp = ((base_loss - mm_loss) / base_loss) * 100
        winner = "✅" if mm_loss < base_loss else "❌"
        type_emoji = "📐" if task_type == 'math' else "💻"
        
        print(f"{type_emoji} Task {task_counter:>4} | Base: {base_loss:6.3f} | "
              f"MM: {mm_loss:6.3f} | Δ {imp:+6.2f}% | {winner}")
        
        # Buffer results
        save_buffer.append({
            'task_id': task_counter,
            'task_type': task_type,
            'base_loss': base_loss,
            'mm_loss': mm_loss,
            'improvement_pct': imp,
            'winner': 'MM' if mm_loss < base_loss else 'Base'
        })
        
        # Periodic save
        if len(save_buffer) >= 20:
            results.extend(save_buffer)
            pd.DataFrame(results).to_csv('mirrormind_results.csv', index=False)
            save_buffer.clear()
        
        task_counter += 1
    
    # Final save
    if save_buffer:
        results.extend(save_buffer)
        pd.DataFrame(results).to_csv('mirrormind_results.csv', index=False)
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print(f"   ├─ Results saved: 'mirrormind_results.csv'")
    print(f"   ├─ Dashboard saved: 'mirrormind_dashboard.")


if __name__ == "__main__":
    run_mirrormind_benchmark()