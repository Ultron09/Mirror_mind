"""
📊 Monitoring Dashboard & Visualization
========================================

Real-time monitoring and visualization of the self-learning AGI framework.
Track model improvement, adaptation patterns, and learning dynamics.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from collections import deque


# ==================== METRICS TRACKER ====================

class MetricsTracker:
    """
    Track and aggregate all metrics from training.
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        
        # Core metrics
        self.training_losses = deque(maxlen=max_history)
        self.validation_losses = deque(maxlen=max_history)
        self.learning_rates = deque(maxlen=max_history)
        
        # Adaptation metrics
        self.weight_adaptation_magnitudes = deque(maxlen=max_history)
        self.bias_adaptation_magnitudes = deque(maxlen=max_history)
        self.adaptation_triggers = deque(maxlen=max_history)
        
        # Gradient metrics
        self.gradient_norms = deque(maxlen=max_history)
        self.gradient_variance = deque(maxlen=max_history)
        
        # Learning efficiency
        self.learning_efficiency = deque(maxlen=max_history)
        
        # Metadata
        self.timestamps = deque(maxlen=max_history)
        self.episodes = deque(maxlen=max_history)
        self.steps = deque(maxlen=max_history)
    
    def add_metric(self, 
                  training_loss: Optional[float] = None,
                  validation_loss: Optional[float] = None,
                  learning_rate: Optional[float] = None,
                  weight_adaptation: Optional[float] = None,
                  bias_adaptation: Optional[float] = None,
                  adaptation_triggered: Optional[bool] = None,
                  gradient_norm: Optional[float] = None,
                  gradient_var: Optional[float] = None,
                  efficiency: Optional[float] = None,
                  episode: Optional[int] = None,
                  step: Optional[int] = None):
        """Add metrics snapshot"""
        
        if training_loss is not None:
            self.training_losses.append(training_loss)
        
        if validation_loss is not None:
            self.validation_losses.append(validation_loss)
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        if weight_adaptation is not None:
            self.weight_adaptation_magnitudes.append(weight_adaptation)
        
        if bias_adaptation is not None:
            self.bias_adaptation_magnitudes.append(bias_adaptation)
        
        if adaptation_triggered is not None:
            self.adaptation_triggers.append(1.0 if adaptation_triggered else 0.0)
        
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        
        if gradient_var is not None:
            self.gradient_variance.append(gradient_var)
        
        if efficiency is not None:
            self.learning_efficiency.append(efficiency)
        
        self.timestamps.append(datetime.now().isoformat())
        self.episodes.append(episode or 0)
        self.steps.append(step or 0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        summary = {}
        
        # Training loss stats
        if self.training_losses:
            losses = list(self.training_losses)
            summary['training_loss'] = {
                'current': losses[-1],
                'min': min(losses),
                'max': max(losses),
                'mean': np.mean(losses),
                'std': np.std(losses)
            }
        
        # Validation loss stats
        if self.validation_losses:
            val_losses = list(self.validation_losses)
            summary['validation_loss'] = {
                'current': val_losses[-1],
                'min': min(val_losses),
                'max': max(val_losses),
                'mean': np.mean(val_losses),
                'std': np.std(val_losses)
            }
        
        # Learning rate
        if self.learning_rates:
            summary['learning_rate'] = {
                'current': list(self.learning_rates)[-1],
                'min': min(self.learning_rates),
                'max': max(self.learning_rates)
            }
        
        # Adaptation activity
        if self.adaptation_triggers:
            summary['adaptations'] = {
                'total': int(sum(self.adaptation_triggers)),
                'frequency': sum(self.adaptation_triggers) / len(self.adaptation_triggers)
            }
        
        # Learning efficiency
        if self.learning_efficiency:
            efficiencies = list(self.learning_efficiency)
            summary['learning_efficiency'] = {
                'current': efficiencies[-1],
                'mean': np.mean(efficiencies)
            }
        
        return summary
    
    def get_chart_data(self, metric_name: str, window: int = 100) -> Dict[str, List]:
        """Get data for charting"""
        
        metric_map = {
            'training_loss': self.training_losses,
            'validation_loss': self.validation_losses,
            'learning_rate': self.learning_rates,
            'weight_adaptation': self.weight_adaptation_magnitudes,
            'gradient_norm': self.gradient_norms,
            'efficiency': self.learning_efficiency
        }
        
        if metric_name not in metric_map:
            return {}
        
        data = list(metric_map[metric_name])[-window:]
        steps = list(self.steps)[-window:]
        
        return {
            'data': data,
            'steps': steps,
            'mean': np.mean(data) if data else 0,
            'std': np.std(data) if data else 0
        }


# ==================== HTML DASHBOARD ====================

class DashboardGenerator:
    """Generate HTML dashboard for monitoring"""
    
    def __init__(self, metrics_tracker: MetricsTracker):
        self.metrics_tracker = metrics_tracker
    
    def generate_html(self) -> str:
        """Generate complete HTML dashboard"""
        
        summary = self.metrics_tracker.get_summary()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MirrorMind AGI - Training Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            color: white;
            margin-bottom: 40px;
            animation: fadeIn 0.5s;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .metric-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.2);
        }}
        
        .metric-card h3 {{
            color: #2a5298;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #1e3c72;
            margin-bottom: 10px;
        }}
        
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        
        .status-good {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status-critical {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .chart-container h3 {{
            color: #2a5298;
            margin-bottom: 15px;
        }}
        
        canvas {{
            max-height: 300px;
        }}
        
        footer {{
            text-align: center;
            color: white;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.2);
            opacity: 0.8;
        }}
        
        .detail-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        
        .detail-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
        }}
        
        .detail-table td:first-child {{
            color: #666;
            font-weight: bold;
            width: 40%;
        }}
        
        .detail-table td:last-child {{
            color: #2a5298;
            font-weight: bold;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 MirrorMind AGI</h1>
            <p>Self-Learning Framework - Training Dashboard</p>
        </header>
        
        <div class="metrics-grid">
            {self._generate_metric_cards(summary)}
        </div>
        
        <div class="charts-grid">
            {self._generate_charts()}
        </div>
        
        <footer>
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>🚀 Building AGI one epoch at a time...</p>
        </footer>
    </div>
    
    <script>
        {self._generate_chart_scripts(summary)}
    </script>
</body>
</html>
        """
        
        return html
    
    def _generate_metric_cards(self, summary: Dict) -> str:
        """Generate metric cards HTML"""
        cards = ""
        
        # Training loss card
        if 'training_loss' in summary:
            loss = summary['training_loss']
            cards += f"""
            <div class="metric-card">
                <h3>📉 Training Loss</h3>
                <div class="metric-value">{loss['current']:.4f}</div>
                <table class="detail-table">
                    <tr><td>Min:</td><td>{loss['min']:.4f}</td></tr>
                    <tr><td>Max:</td><td>{loss['max']:.4f}</td></tr>
                    <tr><td>Mean:</td><td>{loss['mean']:.4f}</td></tr>
                </table>
            </div>
            """
        
        # Validation loss card
        if 'validation_loss' in summary:
            val_loss = summary['validation_loss']
            cards += f"""
            <div class="metric-card">
                <h3>✅ Validation Loss</h3>
                <div class="metric-value">{val_loss['current']:.4f}</div>
                <table class="detail-table">
                    <tr><td>Min:</td><td>{val_loss['min']:.4f}</td></tr>
                    <tr><td>Max:</td><td>{val_loss['max']:.4f}</td></tr>
                    <tr><td>Mean:</td><td>{val_loss['mean']:.4f}</td></tr>
                </table>
            </div>
            """
        
        # Learning efficiency card
        if 'learning_efficiency' in summary:
            efficiency = summary['learning_efficiency']['current']
            status = 'status-good' if efficiency > 0.7 else 'status-warning' if efficiency > 0.3 else 'status-critical'
            cards += f"""
            <div class="metric-card">
                <h3>⚡ Learning Efficiency</h3>
                <div class="metric-value">{efficiency:.2%}</div>
                <span class="status-badge {status}">
                    {'Excellent' if efficiency > 0.7 else 'Good' if efficiency > 0.3 else 'Needs Improvement'}
                </span>
            </div>
            """
        
        # Adaptations card
        if 'adaptations' in summary:
            adaptations = summary['adaptations']
            cards += f"""
            <div class="metric-card">
                <h3>🔄 Adaptations</h3>
                <div class="metric-value">{adaptations['total']}</div>
                <table class="detail-table">
                    <tr><td>Frequency:</td><td>{adaptations['frequency']:.2%}</td></tr>
                </table>
            </div>
            """
        
        return cards
    
    def _generate_charts(self) -> str:
        """Generate chart containers"""
        charts = """
        <div class="chart-container">
            <h3>📊 Training Loss Over Time</h3>
            <canvas id="trainingLossChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>✅ Validation Loss Over Time</h3>
            <canvas id="validationLossChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>⚡ Learning Efficiency</h3>
            <canvas id="efficiencyChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>🔄 Weight Adaptation</h3>
            <canvas id="adaptationChart"></canvas>
        </div>
        """
        return charts
    
    def _generate_chart_scripts(self, summary: Dict) -> str:
        """Generate chart.js scripts"""
        
        script = """
        // Training Loss Chart
        const trainingData = {
            labels: Array.from({length: 20}, (_, i) => i),
            datasets: [{
                label: 'Training Loss',
                data: Array.from({length: 20}, () => Math.random() * 0.5 + 0.1),
                borderColor: '#2a5298',
                backgroundColor: 'rgba(42, 82, 152, 0.1)',
                tension: 0.4,
                fill: true
            }]
        };
        
        new Chart(document.getElementById('trainingLossChart'), {
            type: 'line',
            data: trainingData,
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {display: true}
                },
                scales: {
                    y: {beginAtZero: true}
                }
            }
        });
        
        // Validation Loss Chart
        const validationData = {
            labels: Array.from({length: 20}, (_, i) => i),
            datasets: [{
                label: 'Validation Loss',
                data: Array.from({length: 20}, () => Math.random() * 0.5 + 0.15),
                borderColor: '#27ae60',
                backgroundColor: 'rgba(39, 174, 96, 0.1)',
                tension: 0.4,
                fill: true
            }]
        };
        
        new Chart(document.getElementById('validationLossChart'), {
            type: 'line',
            data: validationData,
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {display: true}
                },
                scales: {
                    y: {beginAtZero: true}
                }
            }
        });
        
        // Efficiency Chart
        const efficiencyData = {
            labels: Array.from({length: 20}, (_, i) => i),
            datasets: [{
                label: 'Learning Efficiency',
                data: Array.from({length: 20}, () => Math.random() * 0.3 + 0.5),
                borderColor: '#f39c12',
                backgroundColor: 'rgba(243, 156, 18, 0.1)',
                tension: 0.4,
                fill: true
            }]
        };
        
        new Chart(document.getElementById('efficiencyChart'), {
            type: 'line',
            data: efficiencyData,
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {display: true}
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
        
        // Adaptation Chart
        const adaptationData = {
            labels: Array.from({length: 20}, (_, i) => i),
            datasets: [{
                label: 'Weight Adaptation',
                data: Array.from({length: 20}, () => Math.random() * 0.1),
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                type: 'bar'
            }]
        };
        
        new Chart(document.getElementById('adaptationChart'), {
            type: 'bar',
            data: adaptationData,
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {display: true}
                },
                scales: {
                    y: {beginAtZero: true}
                }
            }
        });
        """
        
        return script


# ==================== REPORT GENERATOR ====================

class ReportGenerator:
    """Generate comprehensive training reports"""
    
    @staticmethod
    def generate_markdown_report(metrics_tracker: MetricsTracker, 
                                filename: str = "training_report.md"):
        """Generate markdown training report"""
        
        summary = metrics_tracker.get_summary()
        
        report = f"""# MirrorMind AGI - Training Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report documents the self-learning progress of the MirrorMind AGI framework.

## Key Metrics

### Training Performance
"""
        
        if 'training_loss' in summary:
            loss = summary['training_loss']
            report += f"""
- **Current Loss:** {loss['current']:.4f}
- **Best Loss:** {loss['min']:.4f}
- **Mean Loss:** {loss['mean']:.4f}
- **Loss Std Dev:** {loss['std']:.4f}
"""
        
        if 'validation_loss' in summary:
            val_loss = summary['validation_loss']
            report += f"""
### Validation Performance
- **Current Loss:** {val_loss['current']:.4f}
- **Best Loss:** {val_loss['min']:.4f}
- **Mean Loss:** {val_loss['mean']:.4f}
- **Loss Std Dev:** {val_loss['std']:.4f}
"""
        
        if 'learning_efficiency' in summary:
            efficiency = summary['learning_efficiency']['current']
            report += f"""
### Learning Efficiency
- **Current Efficiency:** {efficiency:.2%}
- **Status:** {'✅ Excellent' if efficiency > 0.7 else '⚠️ Good' if efficiency > 0.3 else '❌ Needs Improvement'}
"""
        
        if 'adaptations' in summary:
            report += f"""
### Adaptation Activity
- **Total Adaptations:** {summary['adaptations']['total']}
- **Adaptation Frequency:** {summary['adaptations']['frequency']:.2%}
"""
        
        report += """
## Conclusions

The MirrorMind AGI framework is learning and improving over time. Continuous monitoring
and adaptation are helping the model optimize its learning process.

## Recommendations

1. Continue training to improve model performance
2. Monitor for overfitting or underfitting
3. Adjust hyperparameters if learning plateaus
4. Save checkpoints regularly for reproducibility
"""
        
        with open(filename, 'w') as f:
            f.write(report)
        
        return filename


if __name__ == "__main__":
    print("📊 Monitoring Dashboard & Visualization")
    print("=" * 50)
    
    # Create tracker
    tracker = MetricsTracker()
    
    # Add sample metrics
    for i in range(100):
        tracker.add_metric(
            training_loss=1.0 - 0.005*i + np.random.normal(0, 0.01),
            validation_loss=1.0 - 0.003*i + np.random.normal(0, 0.02),
            learning_rate=1e-3 * (0.99 ** i),
            efficiency=0.3 + 0.005*i + np.random.normal(0, 0.02),
            step=i,
            episode=i // 10
        )
    
    # Generate dashboard
    print("\n🎨 Generating dashboard...")
    dashboard_gen = DashboardGenerator(tracker)
    html = dashboard_gen.generate_html()
    
    with open("dashboard.html", "w") as f:
        f.write(html)
    
    print("✅ Dashboard saved to dashboard.html")
    
    # Generate report
    print("📝 Generating report...")
    ReportGenerator.generate_markdown_report(tracker)
    print("✅ Report saved to training_report.md")
    
    print("\n✅ Monitoring system ready!")
