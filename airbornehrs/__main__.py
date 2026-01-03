"""
AirborneHRS - Next-Gen Production Dashboard
============================================
Ultra-Advanced CLI with Real-Time Monitoring, Interactive Demos, and AI Health Checks

Usage: python -m airbornehrs [OPTIONS]
"""
import sys
import subprocess
import importlib
import platform
import time
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
import torch
import torch.nn as nn

# Ensure local package is used
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- CONFIG ---
VERSION = "1.0.9"
AUTHOR = "Suryaansh Prithvijit Singh"
ASCII_LOGO = """
 █████╗ ██╗██████╗ ██████╗  ██████╗ ██████╗ ███╗   ██╗███████╗
██╔══██╗██║██╔══██╗██╔══██╗██╔═══██╗██╔══██╗████╗  ██║██╔════╝
███████║██║██████╔╝██████╔╝██║   ██║██████╔╝██╔██╗ ██║█████╗
██╔══██║██║██╔══██╗██╔══██╗██║   ██║██╔══██╗██║╚██╗██║██╔══╝
██║  ██║██║██║  ██║██████╔╝╚██████╔╝██║  ██║██║ ╚████║███████╗
╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝
"""

# --- 1. SELF-HEALING DEPENDENCY CHECK ---
def ensure_dependencies() -> Tuple[bool, bool]:
    """
    Checks and auto-installs optional dependencies.
    Returns: (has_rich, has_psutil)
    """
    has_rich = False
    has_psutil = False
    
    # Check Rich
    try:
        import rich
        has_rich = True
    except ImportError:
        print("\n⚡ Installing 'rich' for enhanced UI...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "rich", "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            importlib.invalidate_caches()
            import rich
            has_rich = True
            print("✅ Rich installed successfully!")
        except Exception as e:
            print(f"⚠️  Could not install rich: {e}")
    
    # Check psutil for system monitoring
    try:
        import psutil
        has_psutil = True
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "psutil", "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            importlib.invalidate_caches()
            import psutil
            has_psutil = True
        except:
            pass
    
    return has_rich, has_psutil

HAS_RICH, HAS_PSUTIL = ensure_dependencies()

# --- IMPORTS & UI SETUP ---
if HAS_RICH:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        from rich.layout import Layout
        from rich import box
        from rich.text import Text
        from rich.align import Align
        from rich.live import Live
        from rich.syntax import Syntax
        from rich.tree import Tree
        from rich.columns import Columns
        from rich.prompt import Prompt, Confirm
        from rich.markdown import Markdown
        console = Console()
    except ImportError:
        HAS_RICH = False

if HAS_PSUTIL:
    import psutil

if not HAS_RICH:
    class Console:
        def print(self, *args, **kwargs):
            msg = args[0] if args else ""
            print(str(msg).replace('[bold]', '').replace('[/bold]', ''))
        def clear(self): pass
    console = Console()


# --- HARDWARE & SYSTEM MONITORING ---
class SystemMonitor:
    """Advanced system monitoring with real-time stats"""
    
    @staticmethod
    def get_hardware_info() -> Dict[str, str]:
        """Comprehensive hardware detection"""
        info = {
            "System": platform.system(),
            "Platform": platform.platform(),
            "Processor": platform.processor() or "Unknown",
            "Architecture": platform.machine(),
            "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
        
        # CPU Info
        if HAS_PSUTIL:
            info["CPU Cores"] = f"{psutil.cpu_count(logical=False)} Physical / {psutil.cpu_count(logical=True)} Logical"
            info["CPU Usage"] = f"{psutil.cpu_percent(interval=0.1)}%"
            
            # Memory
            mem = psutil.virtual_memory()
            info["RAM"] = f"{mem.total / (1024**3):.1f} GB ({mem.percent}% used)"
        
        # PyTorch Detection
        try:
            import torch
            info["PyTorch"] = torch.__version__
            
            if torch.cuda.is_available():
                info["Compute"] = "CUDA (NVIDIA)"
                info["GPU"] = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info["VRAM"] = f"{vram:.1f} GB"
                info["CUDA Version"] = torch.version.cuda
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info["Compute"] = "MPS (Apple Silicon)"
                info["GPU"] = "Apple Neural Engine"
                info["VRAM"] = "Unified Memory"
            else:
                info["Compute"] = "CPU Only"
                info["GPU"] = "None"
                info["VRAM"] = "N/A"
                
        except ImportError:
            info["PyTorch"] = "❌ Not Installed"
            info["Compute"] = "Unknown"
        
        return info
    
    @staticmethod
    def check_gpu_health() -> Dict[str, any]:
        """Detailed GPU health check"""
        health = {"status": "unknown", "details": {}}
        
        try:
            import torch
            if torch.cuda.is_available():
                health["status"] = "optimal"
                health["details"] = {
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
                    "memory_reserved": f"{torch.cuda.memory_reserved() / 1e9:.2f} GB",
                }
            else:
                health["status"] = "cpu_mode"
        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
        
        return health


# --- MODULE INTEGRITY CHECKER ---
class ModuleChecker:
    """Advanced module verification with dependency analysis"""
    
    CORE_MODULES = [
        ("Core Framework", "airbornehrs.core", "AdaptiveFramework"),
        ("Memory System", "airbornehrs.memory", "UnifiedMemoryHandler"),
        ("Consciousness", "airbornehrs.consciousness_v2", "ConsciousnessCore"),
        ("Meta Controller", "airbornehrs.meta_controller", "MetaController"),
        ("Adapters", "airbornehrs.adapters", "AdapterBank"),
        ("Presets", "airbornehrs.presets", "PRESETS"),
        ("Production", "airbornehrs.production", "ProductionAdapter"),
    ]
    
    @staticmethod
    def check_module(path: str, class_name: Optional[str] = None) -> Tuple[bool, str]:
        """Check if module exists and optionally verify a class"""
        try:
            mod = importlib.import_module(path)
            
            if class_name:
                if not hasattr(mod, class_name):
                    return False, f"Missing class: {class_name}"
            
            return True, "✓ OK"
        except ImportError as e:
            return False, f"Import Error: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    @classmethod
    def run_full_check(cls) -> List[Tuple[str, str, bool, str]]:
        """Run comprehensive module checks"""
        results = []
        for name, path, class_name in cls.CORE_MODULES:
            success, message = cls.check_module(path, class_name)
            results.append((name, path, success, message))
        return results


class DemoSessionReport:
    def __init__(self):
        self.steps = []
        self.summary = {
            "panic_events": 0,
            "novelty_events": 0,
            "dream_events": 0,
            "meta_updates": 0,
            "emotions": {},
            "max_surprise": 0.0,
            "max_importance": 0.0
        }

    def record_step(self, step, choice, metrics, raw):
        entry = {
            "step": step,
            "choice": choice,
            "loss": metrics["loss"],
            "z_score": metrics["z_score"],
            "mode": metrics["mode"],
            "plasticity": metrics["plasticity"],
            "emotion": raw.get("emotion") if raw else None,
            "surprise": raw.get("surprise") if raw else None,
            "importance": raw.get("importance") if raw else None,
        }
        self.steps.append(entry)

        # Aggregate stats
        if metrics["mode"] == "PANIC":
            self.summary["panic_events"] += 1
        if metrics["mode"] == "NOVELTY":
            self.summary["novelty_events"] += 1
        if choice == "dream":
            self.summary["dream_events"] += 1

        if raw:
            emo = raw.get("emotion")
            if emo:
                self.summary["emotions"][emo] = self.summary["emotions"].get(emo, 0) + 1
            self.summary["max_surprise"] = max(
                self.summary["max_surprise"],
                raw.get("surprise", 0.0)
            )
            self.summary["max_importance"] = max(
                self.summary["max_importance"],
                raw.get("importance", 0.0)
            )

# --- INTERACTIVE DEMO (HUMAN-IN-THE-LOOP) ---
# --- INTERACTIVE DEMO (HUMAN-IN-THE-LOOP + RAW CONSCIOUSNESS) ---
class InteractiveDemo:
    """
    Human-in-the-loop cognitive demo.
    User controls reality.
    System exposes internal consciousness when it reacts.
    """

    @staticmethod
    def render_consciousness(console, raw):
        if not raw:
            console.print("[dim]Consciousness data unavailable[/dim]")
            return

        table = Table(
            title="🧠 Raw Consciousness State",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold magenta"
        )

        table.add_column("Signal", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        for k, v in raw.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = f"{v.item():.4f}"
                else:
                    v = f"Tensor(shape={tuple(v.shape)})"
            elif isinstance(v, float):
                v = f"{v:.4f}"
            table.add_row(str(k), str(v))

        console.print(table)

    @staticmethod
    def run_quick_demo(model=None):
        if not HAS_RICH:
            print("\n[Demo requires Rich UI library]")
            return

        console.clear()

        console.print(
            Panel(
                "[bold cyan]🧠 MirrorMind — Interactive Cognitive Experiment[/bold cyan]\n\n"
                "You are inside the learning loop.\n"
                "Each step YOU decide what the world looks like.\n"
                "The system adapts — and exposes its mind when stressed.\n\n"
                "[bold]Type 'exit' to finish and receive a report.[/bold]",
                title="🎮 Human-in-the-Loop Mode",
                border_style="cyan"
            )
        )

        session_report = DemoSessionReport()
        step = 0

        try:
            import torch.nn as nn
            from airbornehrs.core import AdaptiveFramework
            from airbornehrs.presets import PRESETS

            # Demo config
            config = PRESETS.fast()
            config.warmup_steps = 2
            config.novelty_z_threshold = 0.5
            config.survival_z_threshold = 1.0
            config.panic_threshold = 0.8
            config.enable_consciousness = True
            config.evaluation_frequency = 1
            config.dream_interval = 3

            if model is None:
                model = nn.Sequential(
                    nn.Linear(10, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            else:
                console.print("[green]✓ Using Custom User Model[/green]")

            framework = AdaptiveFramework(model, config)

            console.print("[green]✓ Cognitive core online[/green]")
            console.print("[green]✓ Consciousness active[/green]")
            console.print("[green]✓ Reflex thresholds lowered[/green]\n")

            while True:
                console.rule(f"[bold cyan]STEP {step}[/bold cyan]")

                choice = Prompt.ask(
                    "[bold]Choose reality[/bold]",
                    choices=["normal", "shift", "chaos", "freeze", "dream", "exit"],
                    default="normal"
                )

                # -------- EXIT --------
                if choice == "exit":
                    console.print("\n[bold green]👋 User exited cognitive experiment[/bold green]\n")
                    break

                # -------- DREAM --------
                if choice == "dream":
                    console.print("[bold magenta]💤 Dreaming from memory buffer[/bold magenta]")
                    framework.learn_from_buffer(batch_size=16, num_epochs=1)

                    session_report.record_step(
                        step=step,
                        choice="dream",
                        metrics={
                            "loss": None,
                            "z_score": None,
                            "mode": "DREAM",
                            "plasticity": None
                        },
                        raw={}
                    )

                    step += 1
                    continue

                # -------- ENVIRONMENTS --------
                if choice == "normal":
                    x = torch.randn(4, 10)
                    y = torch.randn(4, 1)
                    regime = "STABLE ENVIRONMENT"
                elif choice == "shift":
                    x = torch.randn(4, 10) * 5
                    y = torch.randn(4, 1)
                    regime = "DOMAIN SHIFT"
                elif choice == "chaos":
                    x = torch.randn(4, 10) * torch.randn(1).abs()
                    y = torch.randn(4, 1) * 3
                    regime = "CHAOTIC INPUT"
                elif choice == "freeze":
                    x = torch.zeros(4, 10)
                    y = torch.zeros(4, 1)
                    regime = "SIGNAL FREEZE"

                console.print(f"[dim]Environment:[/dim] [bold]{regime}[/bold]")

                # -------- TRAIN --------
                metrics = framework.train_step(x, target_data=y)

                console.print(
                    f"[green]Loss[/green]: {metrics['loss']:.4f} | "
                    f"[cyan]Z[/cyan]: {metrics['z_score']:.2f} | "
                    f"[yellow]Mode[/yellow]: [bold]{metrics['mode']}[/bold] | "
                    f"[magenta]Plasticity[/magenta]: {metrics['plasticity']:.2f}"
                )

                # -------- NARRATION --------
                if metrics["mode"] == "PANIC":
                    console.print("[bold red]🚨 PANIC — system protecting stability[/bold red]")
                elif metrics["mode"] == "SURVIVAL":
                    console.print("[bold red]🛡 SURVIVAL — preserving core knowledge[/bold red]")
                elif metrics["mode"] == "NOVELTY":
                    console.print("[bold yellow]✨ NOVELTY — exploring unknown patterns[/bold yellow]")
                elif metrics["mode"] == "NORMAL":
                    console.print("[bold green]✓ NORMAL — learning stable[/bold green]")
                elif metrics["mode"] == "BOOTSTRAP":
                    console.print("[dim]Bootstrapping internal models[/dim]")

                # -------- CONSCIOUSNESS --------
                raw = None
                if framework.consciousness:
                    raw = getattr(framework.consciousness, "last_metrics", None)

                if raw and metrics["mode"] in {"PANIC", "NOVELTY", "SURVIVAL"}:
                    console.print("[bold magenta]🧠 RAW CONSCIOUSNESS SPIKE[/bold magenta]")
                    InteractiveDemo.render_consciousness(console, raw)

                # -------- RECORD STEP (CRITICAL) --------
                session_report.record_step(
                    step=step,
                    choice=choice,
                    metrics=metrics,
                    raw=raw or {}
                )

                step += 1
                time.sleep(0.1)

            # -------- FINAL REPORT --------
            InteractiveDemo.render_final_report(console, session_report)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Demo interrupted by user[/bold yellow]")
            InteractiveDemo.render_final_report(console, session_report)

        except Exception as e:
            console.print(f"\n[bold red]Demo error:[/bold red] {e}")
            InteractiveDemo.render_final_report(console, session_report)

    @staticmethod
    def render_final_report(console, report):
        console.rule("[bold cyan]🧾 COGNITIVE SESSION REPORT[/bold cyan]")

        total_steps = len(report.steps)
        dominant_emotion = max(
            report.summary["emotions"].items(),
            key=lambda x: x[1],
            default=("neutral", 0)
        )[0]

        console.print(
            Panel(
                f"""
Total interactions: {total_steps}
Panic responses: {report.summary['panic_events']}
Dream events: {report.summary['dream_events']}
Dominant emotional state: {dominant_emotion}

This session demonstrates how the system adapts its learning
rate, memory, and behavior based on stress, novelty, and repetition.
""".strip(),
                title="🧠 Session Interpretation",
                border_style="cyan"
            )
        )

        console.print(
            Panel(
                f"""
Peak surprise: {report.summary['max_surprise']:.2f}
Peak importance: {report.summary['max_importance']:.2f}

Surprise reflects unexpected situations.
Importance reflects what the system chose to remember.
""".strip(),
                title="⚡ Cognitive Signals",
                border_style="green"
            )
        )

        console.print("\n[bold green]✓ End of cognitive report[/bold green]\n")


# --- RICH UI (ULTRA-ADVANCED) ---
def create_header() -> Panel:
    """Create animated header with ASCII art"""
    logo = Text(ASCII_LOGO, style="bold cyan", justify="center")
    subtitle = Text(f"v{VERSION} | Adaptive Meta-Learning Framework", style="dim white", justify="center")
    author = Text(f"by {AUTHOR}", style="italic yellow", justify="center")
    
    content = Text()
    content.append_text(logo)
    content.append("\n")
    content.append_text(subtitle)
    content.append("\n")
    content.append_text(author)
    
    return Panel(
        content,
        box=box.DOUBLE_EDGE,
        style="cyan",
        border_style="bold blue"
    )


def create_system_table() -> Table:
    """Create beautiful system information table"""
    hw = SystemMonitor.get_hardware_info()
    
    table = Table(
        title="🖥️  System Configuration",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="blue"
    )
    
    table.add_column("Component", style="cyan", width=20)
    table.add_column("Specification", style="green")
    
    # Color-code important items
    for key, value in hw.items():
        if "GPU" in key or "CUDA" in key:
            table.add_row(key, f"[bold yellow]{value}[/bold yellow]")
        elif "CPU" in key or "RAM" in key:
            table.add_row(key, f"[bold green]{value}[/bold green]")
        elif "❌" in value:
            table.add_row(key, f"[bold red]{value}[/bold red]")
        else:
            table.add_row(key, value)
    
    return table


def create_module_tree() -> Tree:
    """Create hierarchical module status tree"""
    results = ModuleChecker.run_full_check()
    
    tree = Tree("📦 [bold]AirborneHRS Modules[/bold]")
    
    for name, path, success, message in results:
        if success:
            branch = tree.add(f"[green]✓[/green] {name}")
            branch.add(f"[dim]{path}[/dim]")
        else:
            branch = tree.add(f"[red]✗[/red] {name}")
            branch.add(f"[dim]{path}[/dim]")
            branch.add(f"[red]{message}[/red]")
    
    return tree


def create_health_panel() -> Panel:
    """Create GPU health monitoring panel"""
    health = SystemMonitor.check_gpu_health()
    
    if health["status"] == "optimal":
        content = "[bold green]✓ GPU ONLINE[/bold green]\n\n"
        for key, val in health["details"].items():
            content += f"[cyan]{key}:[/cyan] {val}\n"
        style = "green"
    elif health["status"] == "cpu_mode":
        content = "[bold yellow]⚠ CPU MODE[/bold yellow]\n\n"
        content += "No GPU detected. Running on CPU.\n"
        content += "Performance may be limited."
        style = "yellow"
    else:
        content = "[bold red]✗ ERROR[/bold red]\n\n"
        content += f"Status: {health.get('error', 'Unknown')}"
        style = "red"
    
    return Panel(
        content,
        title="🔥 Compute Health",
        box=box.ROUNDED,
        border_style=style
    )


def create_features_panel() -> Panel:
    """Highlight key features"""
    features = """
🧠 **Meta-Learning**: Reptile algorithm for stable online adaptation
🎯 **Memory System**: Hybrid EWC + SI with adaptive regularization
🌟 **Consciousness**: 5D self-awareness with emotional states
⚡ **Active Shield**: Hierarchical reflex system (PANIC/NOVELTY/NORMAL)
🔄 **Dreaming**: Prioritized experience replay
🎨 **Adapters**: Dynamic FiLM layers for task-specific modulation
📊 **Presets**: 10+ production-ready configurations
    """
    
    md = Markdown(features)
    return Panel(
        md,
        title="✨ Key Features",
        box=box.ROUNDED,
        border_style="magenta"
    )


def run_interactive_dashboard():
    """Main interactive dashboard with live updates"""
    console.clear()
    
    # Header
    console.print(create_header())
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=20),
        Layout(name="middle", size=15),
        Layout(name="bottom")
    )
    
    layout["top"].split_row(
        Layout(create_system_table(), name="system"),
        Layout(create_health_panel(), name="health")
    )
    
    layout["middle"].update(create_module_tree())
    layout["bottom"].update(create_features_panel())
    
    console.print(layout)
    
    # Footer with options
    console.print("\n" + "─" * console.width)
    console.print("[bold white]🎮 Interactive Options:[/bold white]")
    console.print("  [cyan]1.[/cyan] Run Quick Demo")
    console.print("  [cyan]2.[/cyan] Show Documentation")
    console.print("  [cyan]3.[/cyan] Export System Report")
    console.print("  [cyan]4.[/cyan] Model Playground (No-Code)")
    console.print("  [cyan]5.[/cyan] Exit")
    console.print("─" * console.width + "\n")


def show_documentation():
    """Display interactive documentation"""
    docs = """
# 📚 Quick Start Guide

## Installation
```python
pip install airbornehrs
```

## Basic Usage
```python
from airbornehrs import AdaptiveFramework, PRESETS
import torch.nn as nn

# Your model
model = nn.Sequential(...)

# Initialize framework
config = PRESETS.production()
framework = AdaptiveFramework(model, config)

# Training loop
for x, y in dataloader:
    metrics = framework.train_step(x, target_data=y)
    print(f"Loss: {metrics['loss']:.4f}, Mode: {metrics['mode']}")
```

## Production Deployment
```python
from airbornehrs.production import ProductionAdapter

adapter = ProductionAdapter(framework, inference_mode="online")
prediction = adapter.predict(input_data, update=True, target=target)
```

## Available Presets
- `production()` - Maximum accuracy & stability
- `fast()` - Real-time learning
- `balanced()` - Good default
- `memory_efficient()` - Mobile/edge devices
- `accuracy_focus()` - Maximum correctness
- And 5 more...

For full documentation: https://github.com/Ultron09/Mirror_mind
    """
    
    console.print(Panel(Markdown(docs), title="📖 Documentation", border_style="cyan"))


def export_system_report():
    """Export comprehensive system report to JSON"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "version": VERSION,
        "system": SystemMonitor.get_hardware_info(),
        "gpu_health": SystemMonitor.check_gpu_health(),
        "modules": [
            {"name": name, "path": path, "status": "ok" if success else "failed", "message": msg}
            for name, path, success, msg in ModuleChecker.run_full_check()
        ]
    }
    
    filename = f"airbornehrs_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    console.print(f"\n[green]✓[/green] Report exported to: [bold]{filename}[/bold]")


# --- MODEL PLAYGROUND (NO-CODE) ---
class ModelPlayground:
    """
    Interactive No-Code Model Builder & Trainer.
    Allows users to design, train, and test models on the fly.
    """
    
    @staticmethod
    def run():
        console.clear()
        console.print(Panel("[bold cyan]🛠️  Model Playground (No-Code Mode)[/bold cyan]", border_style="cyan"))
        
        # 1. Select Dataset
        dataset_name = Prompt.ask("Select Dataset", choices=["MNIST", "FashionMNIST", "Synthetic"], default="MNIST")
        console.print(f"[green]Selected: {dataset_name}[/green]")
        
        # Shape tracking: (Channels, Height, Width) or (Features,)
        current_shape = (1, 28, 28) # Default for MNIST/FashionMNIST
        output_dim = 10
        
        if dataset_name == "Synthetic":
            current_shape = (20,)
            output_dim = 2
        
        # 2. Build Model
        console.print("\n[bold]🏗️  Build Your Neural Network[/bold]")
        layers = []
        
        while True:
            # Display current shape
            shape_str = f"{current_shape}"
            console.print(f"Current Output Shape: [cyan](Batch, {shape_str})[/cyan]")
            
            # Determine available layer types based on current shape
            is_image = len(current_shape) == 3
            
            choices = ["Linear", "ReLU", "LeakyReLU", "Dropout", "Finish"]
            if is_image:
                choices = ["Conv2d", "MaxPool2d", "VGG Block", "Flatten"] + choices
            
            layer_type = Prompt.ask(
                "Add Layer", 
                choices=choices, 
                default="Linear" if not is_image else "Conv2d"
            )
            
            if layer_type == "Finish":
                break
            
            # --- CNN LAYERS ---
            if layer_type == "Conv2d":
                while True:
                    try:
                        out_channels = int(Prompt.ask("Output Channels", default="32"))
                        kernel_size = int(Prompt.ask("Kernel Size", default="3"))
                        stride = int(Prompt.ask("Stride", default="1"))
                        padding = int(Prompt.ask("Padding", default="1"))
                        if out_channels > 0 and kernel_size > 0:
                            break
                        console.print("[red]Values must be positive[/red]")
                    except ValueError:
                        console.print("[red]Invalid input. Please enter integers.[/red]")
                
                in_channels = current_shape[0]
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
                
                # Update shape: H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
                h_out = int((current_shape[1] + 2*padding - 1*(kernel_size-1) - 1)/stride + 1)
                w_out = int((current_shape[2] + 2*padding - 1*(kernel_size-1) - 1)/stride + 1)
                current_shape = (out_channels, h_out, w_out)
                
            elif layer_type == "MaxPool2d":
                while True:
                    try:
                        kernel_size = int(Prompt.ask("Kernel Size", default="2"))
                        stride = int(Prompt.ask("Stride", default="2"))
                        if kernel_size > 0:
                            break
                    except ValueError:
                        console.print("[red]Invalid input.[/red]")
                        
                layers.append(nn.MaxPool2d(kernel_size, stride))
                h_out = int((current_shape[1] - 1*(kernel_size-1) - 1)/stride + 1)
                w_out = int((current_shape[2] - 1*(kernel_size-1) - 1)/stride + 1)
                current_shape = (current_shape[0], h_out, w_out)
                
            elif layer_type == "VGG Block":
                # Conv -> ReLU -> MaxPool
                out_channels = 64 if current_shape[0] < 64 else current_shape[0] * 2
                console.print(f"[dim]Adding VGG Block (Conv{current_shape[0]}->{out_channels}, ReLU, MaxPool2)[/dim]")
                
                layers.append(nn.Conv2d(current_shape[0], out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(2, 2))
                
                # Update shape (Conv3x3 pad1 preserves size, MaxPool2x2 halves it)
                h_out = current_shape[1] // 2
                w_out = current_shape[2] // 2
                current_shape = (out_channels, h_out, w_out)

            elif layer_type == "Flatten":
                layers.append(nn.Flatten())
                flat_dim = current_shape[0] * current_shape[1] * current_shape[2]
                current_shape = (flat_dim,)

            # --- LINEAR LAYERS ---    
            elif layer_type == "Linear":
                # Auto-flatten if input is 3D image
                if len(current_shape) == 3:
                    console.print("[yellow]Auto-flattening 3D input for Linear layer...[/yellow]")
                    layers.append(nn.Flatten())
                    flat_dim = current_shape[0] * current_shape[1] * current_shape[2]
                    current_shape = (flat_dim,)
                
                while True:
                    try:
                        out_features = int(Prompt.ask("Output Size", default="128"))
                        if out_features > 0:
                            break
                        console.print("[red]Size must be positive[/red]")
                    except ValueError:
                        console.print("[red]Invalid input. Please enter an integer.[/red]")
                
                layers.append(nn.Linear(current_shape[0], out_features))
                current_shape = (out_features,)
                
            elif layer_type == "ReLU":
                layers.append(nn.ReLU())
            elif layer_type == "LeakyReLU":
                layers.append(nn.LeakyReLU())
            elif layer_type == "Dropout":
                while True:
                    try:
                        p = float(Prompt.ask("Dropout Probability", default="0.2"))
                        if 0.0 <= p <= 1.0:
                            break
                        console.print("[red]Probability must be between 0 and 1[/red]")
                    except ValueError:
                        console.print("[red]Invalid input. Please enter a number.[/red]")
                
                layers.append(nn.Dropout(p))
                
            console.print(f"[dim]Added {layer_type}[/dim]")
            
        # Add final classifier
        if len(current_shape) == 3:
             layers.append(nn.Flatten())
             current_shape = (current_shape[0] * current_shape[1] * current_shape[2],)
             
        layers.append(nn.Linear(current_shape[0], output_dim))
        model = nn.Sequential(*layers)
        
        console.print("\n[bold green]✓ Model Constructed:[/bold green]")
        console.print(model)
        
        if not Confirm.ask("\nStart Training?"):
            return

        # 3. Setup Framework
        from airbornehrs.core import AdaptiveFramework
        from airbornehrs.presets import PRESETS
        import torch.optim as optim
        from torchvision import datasets, transforms
        
        config = PRESETS.fast()
        config.enable_consciousness = True
        framework = AdaptiveFramework(model, config)
        
        # 4. Load Data
        console.print("[dim]Loading data...[/dim]")
        if dataset_name == "Synthetic":
            train_loader = [(torch.randn(32, 20), torch.randint(0, 2, (32,))) for _ in range(100)]
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            if dataset_name == "MNIST":
                ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
            else:
                ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
            train_loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
            
        # 5. Training Loop
        console.clear()
        console.print(Panel(f"[bold]Training on {dataset_name}[/bold]", border_style="green"))
        
        layout = Layout()
        layout.split_column(
            Layout(name="metrics", size=10),
            Layout(name="consciousness")
        )
        
        with Live(layout, refresh_per_second=4) as live:
            step = 0
            for epoch in range(1): # Demo: 1 epoch
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Only flatten if model expects 1D input (i.e. first layer is Linear)
                    # But here we built the model to handle whatever shape we tracked.
                    # If the user built a CNN, the model expects (B, 1, 28, 28).
                    # If the user built a MLP, the model expects (B, 784).
                    
                    # Check first layer type to decide if we flatten
                    first_layer_is_linear = isinstance(model[0], nn.Linear) or (isinstance(model[0], nn.Flatten) and isinstance(model[1], nn.Linear))
                    
                    # Actually, we can just check if the model starts with a Conv layer
                    is_cnn = any(isinstance(m, nn.Conv2d) for m in model)
                    
                    if dataset_name != "Synthetic":
                        if not is_cnn:
                            data = data.view(data.size(0), -1) # Flatten for MLP
                        # Else keep as (B, 1, 28, 28) for CNN
                        
                    # TRAIN STEP
                    # Ensure target is 1D for classification
                    if target.dim() > 1:
                         target = target.view(-1)
                         
                    metrics = framework.train_step(data, target_data=target)
                    
                    # Update UI
                    metrics_table = Table(title="Training Metrics")
                    metrics_table.add_column("Loss", style="red")
                    metrics_table.add_column("Mode", style="yellow")
                    metrics_table.add_row(
                        f"{metrics['loss']:.4f}",
                        metrics['mode']
                    )
                    
                    cons_panel = Panel("Consciousness Inactive")
                    if framework.consciousness:
                        raw = getattr(framework.consciousness, "last_metrics", {})
                        if raw:
                            emo = raw.get('emotion', 'neutral')
                            surprise = raw.get('surprise', 0.0)
                            cons_panel = Panel(
                                f"Emotion: [bold magenta]{emo.upper()}[/bold magenta]\n"
                                f"Surprise: {surprise:.2f}\n"
                                f"Confidence: {raw.get('confidence', 0.0):.2f}",
                                title="🧠 Machine Consciousness",
                                border_style="magenta"
                            )
                    
                    layout["metrics"].update(metrics_table)
                    layout["consciousness"].update(cons_panel)
                    
                    step += 1
                    if step > 200: # Short demo
                        break
                if step > 200:
                    break
                    
        console.print("\n[bold green]✓ Training Complete![/bold green]")
        
        # --- SAVE & INTEGRATE ---
        if Confirm.ask("Save this model?"):
            filename = Prompt.ask("Enter filename", default=f"model_{int(time.time())}.pth")
            torch.save(model.state_dict(), filename)
            console.print(f"[green]✓ Model saved to {filename}[/green]")
            
            if Confirm.ask("\n🚀 Test this model in the Interactive Simulation (Option 1)?"):
                InteractiveDemo.run_quick_demo(model=model)
                return

        Prompt.ask("Press Enter to return to menu")


# --- FALLBACK UI ---
def run_basic_dashboard():
    """Fallback for environments without Rich"""
    print("\n" + "=" * 60)
    print(f"  AirborneHRS v{VERSION}")
    print(f"  by {AUTHOR}")
    print("=" * 60)
    
    hw = SystemMonitor.get_hardware_info()
    print("\nSystem Information:")
    print("-" * 60)
    for k, v in hw.items():
        print(f"  {k:<20} : {v}")
    
    print("\nModule Status:")
    print("-" * 60)
    for name, path, success, msg in ModuleChecker.run_full_check():
        status = "OK" if success else f"FAIL ({msg})"
        print(f"  [{'✓' if success else '✗'}] {name:<20} : {status}")
    
    print("\n" + "=" * 60)
    print("System ready. Import with: from airbornehrs import AdaptiveFramework")
    print("=" * 60 + "\n")


# --- CLI ARGUMENT PARSER ---
def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="AirborneHRS - Adaptive Meta-Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    parser.add_argument('--docs', action='store_true', help='Show documentation')
    parser.add_argument('--report', action='store_true', help='Export system report')
    parser.add_argument('--basic', action='store_true', help='Use basic UI (no Rich)')
    parser.add_argument('--version', action='version', version=f'AirborneHRS v{VERSION}')
    
    return parser.parse_args()


# --- MAIN ---
def main():
    """Main entry point with argument handling"""
    args = parse_args()
    
    # Handle CLI arguments
    if args.demo:
        InteractiveDemo.run_quick_demo()
        return
    
    if args.docs:
        if HAS_RICH:
            show_documentation()
        else:
            print("\nDocumentation requires Rich UI. Install with: pip install rich")
        return
    
    if args.report:
        export_system_report()
        return
    
    # Main dashboard
    if HAS_RICH and not args.basic:
        try:
            run_interactive_dashboard()
            
            # Interactive menu
            while True:
                choice = Prompt.ask(
                    "\n[bold]Select option[/bold]",
                    choices=["1", "2", "3", "4", "5"],
                    default="5"
                )
                
                if choice == "1":
                    InteractiveDemo.run_quick_demo()
                elif choice == "2":
                    show_documentation()
                elif choice == "3":
                    export_system_report()
                elif choice == "4":
                    ModelPlayground.run()
                    run_interactive_dashboard() # Re-render menu
                elif choice == "5":
                    console.print("\n[bold green]👋 Goodbye![/bold green]\n")
                    break
                
        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]Interrupted by user[/bold yellow]")
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            console.print("\n[dim]Falling back to basic mode...[/dim]\n")
            run_basic_dashboard()
    else:
        run_basic_dashboard()


if __name__ == "__main__":
    main()