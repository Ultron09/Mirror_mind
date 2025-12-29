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

# --- CONFIG ---
VERSION = "1.0.1"
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


# --- INTERACTIVE DEMO ---
class InteractiveDemo:
    """Hands-on demonstration of framework capabilities"""
    
    @staticmethod
    def run_quick_demo():
        """Run a quick demonstration"""
        if not HAS_RICH:
            print("\n[Demo requires Rich UI library]")
            return
        
        console.print("\n[bold cyan]🚀 Running Quick Demo...[/bold cyan]\n")
        
        try:
            import torch
            import torch.nn as nn
            from airbornehrs.core import AdaptiveFramework
            from airbornehrs.presets import PRESETS
            
            # Create a tiny test model
            model = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
            # Initialize with fast preset
            config = PRESETS.fast()
            framework = AdaptiveFramework(model, config)
            
            console.print("[green]✓[/green] Framework initialized")
            
            # Generate synthetic data
            with Progress() as progress:
                task = progress.add_task("[cyan]Training...", total=10)
                
                for i in range(10):
                    x = torch.randn(4, 10)
                    y = torch.randn(4, 1)
                    
                    metrics = framework.train_step(x, y)
                    progress.update(task, advance=1)
                    
                    if i == 9:
                        console.print(f"\n[green]Final Loss:[/green] {metrics['loss']:.4f}")
                        console.print(f"[blue]Mode:[/blue] {metrics['mode']}")
                        console.print(f"[yellow]Plasticity:[/yellow] {metrics['plasticity']:.2f}")
            
            console.print("\n[bold green]✅ Demo completed successfully![/bold green]")
            
        except Exception as e:
            console.print(f"\n[bold red]Demo failed:[/bold red] {e}")


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
    console.print("  [cyan]4.[/cyan] Exit")
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
    metrics = framework.train_step(x, y)
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

For full documentation: https://github.com/yourusername/airbornehrs
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
                    choices=["1", "2", "3", "4"],
                    default="4"
                )
                
                if choice == "1":
                    InteractiveDemo.run_quick_demo()
                elif choice == "2":
                    show_documentation()
                elif choice == "3":
                    export_system_report()
                elif choice == "4":
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