
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import sys
import os

# Ensure we can import airbornehrs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

def get_mnist_split(train=True, digits=[0, 1, 2, 3, 4]):
    """Get a subset of MNIST containing only specific digits."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    
    indices = [i for i, target in enumerate(dataset.targets) if target in digits]
    return Subset(dataset, indices)

def get_fashion_mnist(train=True):
    """Get FashionMNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    return torchvision.datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)

class SimpleCNN(nn.Module):
    """A simple standard CNN to be wrapped."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def run_gauntlet():
    print("\n⚔️  PROTOCOL V6: EXTREME ADAPTABILITY GAUNTLET ⚔️")
    print("===================================================")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")

    # 1. One-Shot Integration
    print("\n[Phase 0] One-Shot Integration...")
    base_model = SimpleCNN().to(DEVICE)
    
    config = AdaptiveFrameworkConfig(
        learning_rate=0.001,
        enable_consciousness=True,
        device=DEVICE,
        memory_type='si', # Synaptic Intelligence for forgetting resistance
        dream_interval=50, # Frequent dreaming
        consciousness_buffer_size=10000
    )
    
    try:
        framework = AdaptiveFramework(base_model, config)
        print("✅ Integration Successful: Model wrapped in AdaptiveFramework.")
    except Exception as e:
        print(f"❌ Integration Failed: {e}")
        return

    # 2. Phase 1: The Baseline (MNIST 0-4)
    print("\n[Phase 1] The Baseline: Learning Digits 0-4")
    train_loader_A = DataLoader(get_mnist_split(train=True, digits=[0,1,2,3,4]), batch_size=64, shuffle=True)
    test_loader_A = DataLoader(get_mnist_split(train=False, digits=[0,1,2,3,4]), batch_size=1000, shuffle=False)
    
    framework.train()
    for batch_idx, (data, target) in enumerate(train_loader_A):
        data, target = data.to(DEVICE), target.to(DEVICE)
        framework.train_step(data, target_data=target)
        if batch_idx >= 100: break # Short training for speed
        
    # Eval Phase 1
    acc_p1 = evaluate(framework, test_loader_A, DEVICE, "Phase 1 (Digits 0-4)")
    
    # 3. Phase 2: The Shift (MNIST 5-9)
    print("\n[Phase 2] The Shift: Sudden Switch to Digits 5-9")
    train_loader_B = DataLoader(get_mnist_split(train=True, digits=[5,6,7,8,9]), batch_size=64, shuffle=True)
    test_loader_B = DataLoader(get_mnist_split(train=False, digits=[5,6,7,8,9]), batch_size=1000, shuffle=False)
    
    # Force a "Task Boundary" in consciousness manually (simulating user signal or auto-detection)
    # In a real scenario, the "Surprise" spike should trigger this, but we help it for the protocol.
    if framework.consciousness:
        print("   -> Signaling Task Change to Consciousness...")
        # We can simulate a high surprise event or just let it adapt naturally.
        # Let's let it adapt naturally first!
    
    framework.train()
    for batch_idx, (data, target) in enumerate(train_loader_B):
        data, target = data.to(DEVICE), target.to(DEVICE)
        framework.train_step(data, target_data=target)
        if batch_idx >= 100: break
        
    # Eval Phase 2
    acc_p2 = evaluate(framework, test_loader_B, DEVICE, "Phase 2 (Digits 5-9)")
    
    # 4. Phase 3: The Alien World (FashionMNIST)
    print("\n[Phase 3] The Alien World: FashionMNIST")
    train_loader_C = DataLoader(get_fashion_mnist(train=True), batch_size=64, shuffle=True)
    test_loader_C = DataLoader(get_fashion_mnist(train=False), batch_size=1000, shuffle=False)
    
    framework.train()
    for batch_idx, (data, target) in enumerate(train_loader_C):
        data, target = data.to(DEVICE), target.to(DEVICE)
        framework.train_step(data, target_data=target)
        if batch_idx >= 100: break
        
    # Eval Phase 3
    acc_p3 = evaluate(framework, test_loader_C, DEVICE, "Phase 3 (FashionMNIST)")
    
    # 5. Phase 4: The Recall (Forgetting Test)
    print("\n[Phase 4] The Recall: Testing Forgetting on Phase 1 (Digits 0-4)")
    # No training here, just eval
    acc_p4 = evaluate(framework, test_loader_A, DEVICE, "Phase 4 (Recall Digits 0-4)")
    
    # Results Analysis
    print("\n=== GAUNTLET RESULTS ===")
    print(f"Phase 1 Accuracy (Baseline): {acc_p1:.2%}")
    print(f"Phase 2 Accuracy (Shift):    {acc_p2:.2%}")
    print(f"Phase 3 Accuracy (Alien):    {acc_p3:.2%}")
    print(f"Phase 4 Accuracy (Recall):   {acc_p4:.2%}")
    
    forgetting = acc_p1 - acc_p4
    print(f"\nForgetting Score: {forgetting:.2%} (Lower is better)")
    
    if forgetting < 0.20:
        print("✅ SUCCESS: System resisted catastrophic forgetting!")
    else:
        print("⚠️ WARNING: Significant forgetting detected.")
        
    if acc_p2 > 0.80 and acc_p3 > 0.70:
        print("✅ SUCCESS: System adapted to new domains effectively.")
    else:
        print("⚠️ WARNING: Adaptation to new domains was weak.")

def evaluate(framework, loader, device, name):
    framework.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output, _, _ = framework(data)
            if isinstance(output, tuple): output = output[0]
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    acc = correct / total
    print(f"   -> {name} Accuracy: {acc:.2%}")
    return acc

if __name__ == "__main__":
    run_gauntlet()
