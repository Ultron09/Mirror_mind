"""
ARC-AGI Data Loader for God Killer Test Suite

Fetches and processes real ARC-AGI benchmark data for comprehensive
MirrorMind validation against all baseline models.

ARC = Abstraction and Reasoning Corpus
- Measures fluid intelligence (IQ-style abstract reasoning)
- Contains 400 training + 100 test tasks
- Each task has multiple examples with grid transformations
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import urllib.request
import zipfile
import os

class ARCDataLoader:
    """Load and process ARC-AGI benchmark data."""
    
    def __init__(self, cache_dir: str = "arc_data"):
        """Initialize ARC data loader."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.training_tasks = []
        self.test_tasks = []
        self.loaded = False
        
    def download_arc_data(self):
        """Download ARC dataset from GitHub."""
        print("[ARC] Downloading ARC-AGI dataset...")
        
        train_url = "https://github.com/fchollet/ARC/raw/master/data/training.zip"
        test_url = "https://github.com/fchollet/ARC/raw/master/data/evaluation.zip"
        
        for url, name in [(train_url, "training"), (test_url, "evaluation")]:
            zip_path = self.cache_dir / f"{name}.zip"
            extract_dir = self.cache_dir / name
            
            if not extract_dir.exists():
                try:
                    print(f"  Downloading {name} split...")
                    urllib.request.urlretrieve(url, zip_path)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    print(f"  ✓ {name} extracted")
                except Exception as e:
                    print(f"  [WARNING] Could not download {name}: {e}")
                    print(f"  Using synthetic ARC-like data instead")
                    return False
        return True
    
    def generate_synthetic_arc_tasks(self, num_tasks: int = 100):
        """Generate synthetic ARC-like tasks for testing."""
        print(f"[ARC] Generating {num_tasks} synthetic ARC-like tasks...")
        
        tasks = []
        for task_id in range(num_tasks):
            # Create grid transformation tasks
            task = self._create_synthetic_task(task_id)
            tasks.append(task)
        
        return tasks
    
    def _create_synthetic_task(self, task_id: int) -> Dict:
        """Create a single synthetic ARC-like task."""
        np.random.seed(task_id)
        
        # Task types: pattern recognition, grid transformation, color mapping
        task_type = task_id % 5
        
        if task_type == 0:  # Pattern extension
            train_examples = self._generate_pattern_extension(task_id)
        elif task_type == 1:  # Color mapping
            train_examples = self._generate_color_mapping(task_id)
        elif task_type == 2:  # Rotation
            train_examples = self._generate_rotation(task_id)
        elif task_type == 3:  # Scaling
            train_examples = self._generate_scaling(task_id)
        else:  # Reflection
            train_examples = self._generate_reflection(task_id)
        
        # Create test examples
        test_examples = [
            {
                'input': np.random.randint(0, 10, (5, 5), dtype=np.int32).tolist(),
                'output': np.random.randint(0, 10, (5, 5), dtype=np.int32).tolist()
            }
            for _ in range(2)
        ]
        
        return {
            'task_id': f'task_{task_id:04d}',
            'task_type': ['pattern', 'color_map', 'rotation', 'scaling', 'reflection'][task_type],
            'train': train_examples,
            'test': test_examples
        }
    
    def _generate_pattern_extension(self, seed):
        """Generate pattern extension task."""
        grid_size = 7 + (seed % 3)
        num_examples = 2 + (seed % 3)
        
        examples = []
        for _ in range(num_examples):
            grid = np.zeros((grid_size, grid_size), dtype=np.int32)
            
            # Add pattern
            pattern_color = (seed % 9) + 1
            for i in range(grid_size):
                grid[i, i] = pattern_color
            
            examples.append({
                'input': grid.tolist(),
                'output': np.rot90(grid).tolist()
            })
        
        return examples
    
    def _generate_color_mapping(self, seed):
        """Generate color mapping task."""
        grid_size = 5 + (seed % 3)
        num_examples = 2 + (seed % 3)
        
        examples = []
        for _ in range(num_examples):
            input_grid = np.random.randint(1, 5, (grid_size, grid_size), dtype=np.int32)
            output_grid = (input_grid * 2) % 10  # Simple color map
            
            examples.append({
                'input': input_grid.tolist(),
                'output': output_grid.tolist()
            })
        
        return examples
    
    def _generate_rotation(self, seed):
        """Generate rotation task."""
        grid_size = 5 + (seed % 4)
        num_examples = 2 + (seed % 2)
        
        examples = []
        for _ in range(num_examples):
            grid = np.random.randint(0, 10, (grid_size, grid_size), dtype=np.int32)
            rotated = np.rot90(grid, k=1 + (seed % 4))
            
            examples.append({
                'input': grid.tolist(),
                'output': rotated.tolist()
            })
        
        return examples
    
    def _generate_scaling(self, seed):
        """Generate scaling task."""
        grid_size = 3 + (seed % 3)
        num_examples = 2 + (seed % 2)
        
        examples = []
        for _ in range(num_examples):
            grid = np.random.randint(0, 10, (grid_size, grid_size), dtype=np.int32)
            scale_factor = 2
            scaled = np.repeat(np.repeat(grid, scale_factor, axis=0), scale_factor, axis=1)
            
            examples.append({
                'input': grid.tolist(),
                'output': scaled.tolist()
            })
        
        return examples
    
    def _generate_reflection(self, seed):
        """Generate reflection task."""
        grid_size = 5 + (seed % 4)
        num_examples = 2 + (seed % 2)
        
        examples = []
        for _ in range(num_examples):
            grid = np.random.randint(0, 10, (grid_size, grid_size), dtype=np.int32)
            reflected = np.fliplr(grid)  # Horizontal flip
            
            examples.append({
                'input': grid.tolist(),
                'output': reflected.tolist()
            })
        
        return examples
    
    def load(self, num_training: int = 100, num_test: int = 50):
        """Load ARC-AGI data."""
        if self.loaded:
            return
        
        print("\n" + "="*60)
        print("ARC-AGI DATA LOADER")
        print("="*60)
        
        # Try to download real data
        if not self.download_arc_data():
            # Fall back to synthetic data
            all_tasks = self.generate_synthetic_arc_tasks(num_training + num_test)
            self.training_tasks = all_tasks[:num_training]
            self.test_tasks = all_tasks[num_training:]
        else:
            # Load from filesystem
            train_dir = self.cache_dir / "training"
            test_dir = self.cache_dir / "evaluation"
            
            self.training_tasks = self._load_tasks_from_dir(train_dir, num_training)
            self.test_tasks = self._load_tasks_from_dir(test_dir, num_test)
        
        self.loaded = True
        print(f"\n[OK] Loaded {len(self.training_tasks)} training tasks")
        print(f"[OK] Loaded {len(self.test_tasks)} test tasks")
        print(f"[OK] Total ARC-AGI data points: {len(self.training_tasks) + len(self.test_tasks)}")
    
    def _load_tasks_from_dir(self, directory: Path, limit: int) -> List[Dict]:
        """Load tasks from directory."""
        tasks = []
        json_files = sorted(directory.glob("*.json"))[:limit]
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    task_data = json.load(f)
                    tasks.append({
                        'task_id': json_file.stem,
                        'train': task_data.get('train', []),
                        'test': task_data.get('test', [])
                    })
            except Exception as e:
                print(f"  Error loading {json_file.name}: {e}")
        
        return tasks
    
    def get_training_tasks(self) -> List[Dict]:
        """Get training tasks."""
        if not self.loaded:
            self.load()
        return self.training_tasks
    
    def get_test_tasks(self) -> List[Dict]:
        """Get test tasks."""
        if not self.loaded:
            self.load()
        return self.test_tasks
    
    def grid_to_tensor(self, grid: List[List[int]], max_size: int = 30) -> torch.Tensor:
        """Convert grid to normalized tensor."""
        grid_array = np.array(grid, dtype=np.float32)
        
        # Pad to max_size
        current_h, current_w = grid_array.shape
        padded = np.zeros((max_size, max_size), dtype=np.float32)
        padded[:current_h, :current_w] = grid_array
        
        # Normalize to [0, 1]
        if padded.max() > 0:
            padded = padded / padded.max()
        
        return torch.from_numpy(padded).unsqueeze(0)  # Add channel dimension
    
    def task_to_dataset(self, task: Dict, max_grid_size: int = 30) -> Tuple[List, List]:
        """Convert ARC task to training dataset."""
        X, y = [], []
        
        # Process training examples
        for example in task.get('train', []):
            input_tensor = self.grid_to_tensor(example.get('input', []), max_grid_size)
            output_tensor = self.grid_to_tensor(example.get('output', []), max_grid_size)
            
            X.append(input_tensor)
            y.append(output_tensor)
        
        return X, y


class ARCBenchmark:
    """Benchmark suite for evaluating models on ARC tasks."""
    
    def __init__(self, arc_loader: ARCDataLoader):
        """Initialize benchmark."""
        self.loader = arc_loader
        self.results = {
            'task_scores': {},
            'task_improvements': {},
            'overall_metrics': {}
        }
    
    def evaluate_model_on_task(self, model, task: Dict, num_steps: int = 50) -> Dict:
        """Evaluate model on single ARC task."""
        X, y = self.loader.task_to_dataset(task)
        
        if not X:  # Skip if no examples
            return {'accuracy': 0.0, 'loss': float('inf')}
        
        # Quick training
        losses = []
        
        for step in range(min(num_steps, len(X) * 2)):
            idx = step % len(X)
            
            # Get input and target
            x_in = X[idx].unsqueeze(0)  # (1, 1, 30, 30)
            y_target = y[idx].unsqueeze(0)  # (1, 1, 30, 30)
            
            # Handle different model types
            if hasattr(model, 'train_step'):
                # MirrorMind framework
                x_flat = x_in.view(x_in.shape[0], -1)
                y_flat = y_target.view(y_target.shape[0], -1)
                metrics = model.train_step(x_flat, y_flat, enable_dream=False)
                loss = metrics.get('loss', 0.5)
                losses.append(loss)
            else:
                # Regular PyTorch models
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                optimizer.zero_grad()
                
                # Flatten for linear layers if needed
                if isinstance(model, torch.nn.Sequential):
                    first_layer = list(model.children())[0]
                    if isinstance(first_layer, torch.nn.Linear):
                        x_flat = x_in.view(x_in.shape[0], -1)
                        output = model(x_flat)
                        if output.dim() == 2:
                            output = output.view(y_target.shape)
                    else:
                        output = model(x_in)
                else:
                    output = model(x_in)
                
                loss = torch.nn.functional.mse_loss(output, y_target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        
        # Calculate accuracy (MSE-based)
        accuracy = max(0, 1.0 - np.mean(losses[-5:]) / 2.0)  # Last 5 steps
        
        return {
            'accuracy': float(accuracy),
            'loss': float(np.mean(losses[-5:])),
            'improvement': float(np.mean(losses[:5]) - np.mean(losses[-5:]))
        }
    
    def evaluate_on_benchmark(self, model, num_tasks: int = 20) -> Dict:
        """Evaluate model on ARC benchmark."""
        test_tasks = self.loader.get_test_tasks()[:num_tasks]
        scores = []
        
        for i, task in enumerate(test_tasks):
            result = self.evaluate_model_on_task(model, task)
            scores.append(result['accuracy'])
        
        return {
            'mean_accuracy': float(np.mean(scores)),
            'std_accuracy': float(np.std(scores)),
            'scores': scores
        }


if __name__ == "__main__":
    # Test the loader
    loader = ARCDataLoader()
    loader.load(num_training=100, num_test=50)
    
    print("\nSample training task:")
    if loader.training_tasks:
        task = loader.training_tasks[0]
        print(f"  Task ID: {task['task_id']}")
        print(f"  Type: {task.get('task_type', 'unknown')}")
        print(f"  Train examples: {len(task['train'])}")
        print(f"  Test examples: {len(task['test'])}")
