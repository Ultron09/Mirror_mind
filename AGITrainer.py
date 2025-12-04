"""
🚀 AGI Trainer & Training Pipeline
===================================

Complete training orchestrator for the self-learning framework.
Handles the full lifecycle of learning, adaptation, and improvement.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional, Any, Callable
import numpy as np
from collections import deque
import json
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from abc import ABC

from SelfLearningFramework import (
    SelfLearningFramework, FrameworkConfig, FeedbackData, SelfLearningModel
)
from AdvancedAdaptation import (
    GradientAnalyzer, DynamicLearningRateScheduler, CurriculumLearningStrategy,
    PerformanceAnalyzer, AdaptiveTaskSelector
)


# ==================== ENVIRONMENT INTERFACE ====================

class Environment(ABC):
    """
    Abstract environment that the model learns from.
    Implement this for your specific problem domain.
    """
    
    def __init__(self):
        self.step_count = 0
        self.episode_count = 0
    
    def reset(self):
        """Reset environment for new episode"""
        self.episode_count += 1
        self.step_count = 0
    
    def step(self, action: Any) -> Tuple[Any, float, bool]:
        """
        Execute action in environment
        
        Returns:
            (observation, reward, done)
        """
        raise NotImplementedError
    
    def render(self):
        """Render current state (optional)"""
        pass


class SimpleRegressionEnvironment(Environment):
    """
    Simple regression task for testing.
    The model learns to map X -> Y.
    """
    
    def __init__(self, input_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Generate ground truth function
        self.weights = torch.randn(input_dim, output_dim)
        
        self.current_input = None
        self.current_target = None
    
    def reset(self):
        super().reset()
        self.step_count = 0
    
    def step(self, prediction: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """Execute step"""
        if self.current_input is None:
            # Generate new sample
            self.current_input = torch.randn(1, self.input_dim)
            self.current_target = torch.tanh(
                torch.matmul(self.current_input, self.weights)
            )
        
        # Compute reward (negative MSE)
        mse = torch.nn.functional.mse_loss(prediction, self.current_target)
        reward = -mse.item()
        
        self.step_count += 1
        done = self.step_count >= 100
        
        next_obs = torch.randn(1, self.input_dim)
        self.current_input = next_obs
        
        return next_obs, reward, done


# ==================== MAIN TRAINER ====================

class AGITrainer:
    """
    Complete training orchestrator for the self-learning AGI.
    
    This manages:
    - The learning loop
    - Adaptation strategies
    - Performance tracking
    - Model improvement
    """
    
    def __init__(self, 
                 config: FrameworkConfig,
                 environment: Optional[Environment] = None,
                 device=None,
                 checkpoint_dir: str = "./checkpoints"):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.config = config
        self.environment = environment or SimpleRegressionEnvironment()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Core framework
        self.framework = SelfLearningFramework(config, device)
        
        # Advanced components
        self.gradient_analyzer = GradientAnalyzer(self.framework.model, config, self.logger)
        self.lr_scheduler = DynamicLearningRateScheduler(
            self.framework.optimizer, config, config.learning_rate
        )
        self.curriculum = CurriculumLearningStrategy(config, self.logger)
        self.performance_analyzer = PerformanceAnalyzer(self.logger)
        self.task_selector = AdaptiveTaskSelector(config, self.logger)
        
        # Training state
        self.global_step = 0
        self.global_episode = 0
        self.best_loss = float('inf')
        
        # Metrics
        self.metrics_log = deque(maxlen=10000)
        
        self.logger.info("🚀 AGITrainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.framework.model.parameters()):,}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('AGITrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = self.checkpoint_dir / "training.log"
            file_handler = logging.FileHandler(log_file , encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Epoch metrics
        """
        self.framework.model.train()
        
        epoch_losses = []
        epoch_adaptations = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (input_data, target) in enumerate(pbar):
            # Training step
            metrics = self.framework.train_step(input_data, target)
            
            loss = metrics['loss']
            epoch_losses.append(loss)
            
            # Update learning rate scheduler
            lr_info = self.lr_scheduler.step(loss)
            
            # Log metrics
            self.metrics_log.append({
                'step': self.global_step,
                'epoch': self.global_episode,
                'loss': loss,
                'learning_rate': lr_info['learning_rate'],
                'timestamp': datetime.now().isoformat()
            })
            
            # Analyze performance
            self.performance_analyzer.add_metrics(loss=loss)
            
            # Periodic logging
            if self.global_step % self.config.log_frequency == 0:
                avg_loss = np.mean(epoch_losses[-self.config.log_frequency:])
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Check for learning issues
                if self.gradient_analyzer.detect_vanishing_gradients():
                    self.logger.warning("⚠️  Vanishing gradients detected!")
                if self.gradient_analyzer.detect_exploding_gradients():
                    self.logger.warning("⚠️  Exploding gradients detected!")
            
            self.global_step += 1
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        
        # Validation
        val_loss = None
        if val_loader:
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['val_loss']
            
            # Check for overfitting
            if self.performance_analyzer.detect_overfitting(avg_loss, val_loss):
                self.logger.warning("⚠️  Potential overfitting detected")
        
        return {
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'num_steps': len(train_loader)
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.framework.model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for input_data, target in val_loader:
                metrics = self.framework.evaluate(input_data, target)
                val_losses.append(metrics['eval_loss'])
        
        return {
            'val_loss': np.mean(val_losses),
            'val_loss_std': np.std(val_losses)
        }
    
    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             num_epochs: int = 10,
             save_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_checkpoint: Whether to save checkpoints
            
        Returns:
            Training summary
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        training_summary = {
            'epochs_completed': 0,
            'total_steps': 0,
            'best_train_loss': float('inf'),
            'best_val_loss': float('inf'),
            'epochs': []
        }
        
        for epoch in range(num_epochs):
            self.global_episode = epoch
            
            # Train epoch
            epoch_metrics = self.train_epoch(train_loader, val_loader)
            
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss = {epoch_metrics['train_loss']:.4f}"
                + (f", Val Loss = {epoch_metrics['val_loss']:.4f}" 
                   if epoch_metrics['val_loss'] else "")
            )
            
            # Update best losses
            if epoch_metrics['train_loss'] < training_summary['best_train_loss']:
                training_summary['best_train_loss'] = epoch_metrics['train_loss']
            
            if epoch_metrics['val_loss'] and epoch_metrics['val_loss'] < training_summary['best_val_loss']:
                training_summary['best_val_loss'] = epoch_metrics['val_loss']
                
                # Save best model
                if save_checkpoint:
                    self.framework.save_checkpoint(
                        str(self.checkpoint_dir / 'best_model.pt')
                    )
            
            # Save periodic checkpoint
            if save_checkpoint and (epoch + 1) % 5 == 0:
                self.framework.save_checkpoint(
                    str(self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
                )
            
            training_summary['epochs'].append(epoch_metrics)
            training_summary['epochs_completed'] = epoch + 1
            training_summary['total_steps'] = self.global_step
        
        self.logger.info("✅ Training completed!")
        
        return training_summary
    
    def get_training_report(self) -> Dict[str, Any]:
        """Generate training report"""
        diagnostics = self.performance_analyzer.get_diagnostics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_steps': self.global_step,
            'total_episodes': self.global_episode,
            'diagnostics': diagnostics,
            'best_loss': self.best_loss,
            'learning_efficiency': self.performance_analyzer.compute_learning_efficiency(),
            'recent_metrics': list(self.metrics_log)[-100:]
        }
    
    def save_training_report(self, path: str = None):
        """Save training report to JSON"""
        if path is None:
            path = self.checkpoint_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.get_training_report()
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Report saved to {path}")


# ==================== SIMPLE API ====================

class EasyTrainer:
    """
    Ultra-simple API for using the self-learning framework.
    
    Usage:
        trainer = EasyTrainer()
        trainer.train(X, y, epochs=10)
        predictions = trainer.predict(X_test)
    """
    
    def __init__(self, config: Optional[FrameworkConfig] = None, device=None):
        """Initialize trainer with default or custom config"""
        if config is None:
            config = FrameworkConfig(
                model_dim=128,
                num_layers=4,
                num_heads=4,
                learning_rate=1e-3,
                batch_size=32
            )
        
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create trainer
        self.trainer = AGITrainer(config, device=self.device)
        
        self.is_trained = False
        self.training_history = {}
    
    def train(self, 
             X: torch.Tensor, 
             y: torch.Tensor,
             val_split: float = 0.2,
             epochs: int = 10,
             batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X: Input data (N, seq_len, feature_dim)
            y: Target data (N, seq_len, feature_dim)
            val_split: Fraction of data to use for validation
            epochs: Number of epochs
            batch_size: Batch size (uses config default if None)
            
        Returns:
            Training summary
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Split into train/val
        n_samples = len(X)
        n_train = int(n_samples * (1 - val_split))
        
        train_dataset = TensorDataset(X[:n_train], y[:n_train])
        val_dataset = TensorDataset(X[n_train:], y[n_train:])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Train
        summary = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            save_checkpoint=True
        )
        
        self.is_trained = True
        self.training_history = summary
        
        return summary
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Make predictions
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        self.trainer.framework.model.eval()
        
        with torch.no_grad():
            X = X.to(self.device)
            output, uncertainty = self.trainer.framework.forward(X)
        
        return output.cpu()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return {
            'is_trained': self.is_trained,
            'total_steps': self.trainer.global_step,
            'best_loss': self.training_history.get('best_train_loss', None),
            'epochs_completed': self.training_history.get('epochs_completed', 0)
        }


# ==================== DEMO ====================

if __name__ == "__main__":
    print("🚀 AGI Trainer & Training Pipeline")
    print("=" * 50)
    
    # Create simple trainer
    print("\n📊 Creating EasyTrainer...")
    trainer = EasyTrainer()
    
    # Generate dummy data
    print("🔧 Generating dummy data...")
    X = torch.randn(100, 10, 128)  # 100 samples, seq_len=10, dim=128
    y = torch.randn(100, 10, 128)
    
    # Train
    print("\n⚡ Starting training...")
    summary = trainer.train(X, y, epochs=2, val_split=0.2)
    
    print(f"\n✅ Training completed!")
    print(f"  Best train loss: {summary['best_train_loss']:.4f}")
    print(f"  Best val loss: {summary['best_val_loss']:.4f}")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    X_test = torch.randn(10, 10, 128)
    predictions = trainer.predict(X_test)
    print(f"  Predictions shape: {predictions.shape}")
    
    print("\n✅ Demo complete!")
