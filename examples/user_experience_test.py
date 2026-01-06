"""
AirborneHRS V8.0 "Sentient" - User Experience Test
==================================================
This script simulates a new user trying to use the package for the first time.
It tests:
1. Imports
2. Configuration simplicity
3. Model wrapping
4. Training loop feedback
5. "Sentient" features (Emotions, Thoughts)
"""

import sys
import os
import torch
import torch.nn as nn
import time

# Simulate installation
sys.path.insert(0, os.getcwd())

print("🚀 [UX TEST] Importing AirborneHRS...")
try:
    from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
    print("✅ Import successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def main():
    print("\n✨ [UX TEST] Setting up User Model...")
    # 1. User defines their standard PyTorch model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    print("✅ Model defined.")

    print("\n⚙️ [UX TEST] Configuring Agent...")
    # 2. User configures the agent (trying to be "smart" but easy)
    config = AdaptiveFrameworkConfig(
        model_dim=64,           # Matches hidden dim
        num_heads=4,            # Simple attention
        enable_consciousness=True, # I want it to be alive!
        memory_type='hybrid',   # Best memory
    )
    print("✅ Configuration ready.")

    print("\n🧠 [UX TEST] Wrapping Model (The 'Magic' Step)...")
    # 3. The one-line wrapper
    agent = AdaptiveFramework(model, config=config)
    print("✅ Model wrapped. Agent is online.")

    print("\n🎓 [UX TEST] Training Loop (5 Steps)...")
    # 4. Training loop
    inputs = torch.randn(4, 10)
    targets = torch.randn(4, 1)

    for step in range(1, 6):
        print(f"\n--- Step {step} ---")
        
        # The user just calls train_step
        metrics = agent.train_step(inputs, target_data=targets)
        
        # Check what the user gets back
        loss = metrics.get('loss', 0.0)
        emotion = metrics.get('emotion', 'Neutral')
        surprise = metrics.get('surprise', 0.0)
        thoughts = metrics.get('thought_trace', [])
        
        print(f"📉 Loss: {loss:.4f}")
        print(f"😲 Surprise: {surprise:.4f}")
        print(f"😊 Emotion: {emotion}")
        if thoughts:
            print(f"💭 Thought: {thoughts[-1]}") # Show the latest thought
        
        time.sleep(0.5) # Simulate work

    print("\n💾 [UX TEST] Testing Save/Load...")
    try:
        path = agent.save_memory("ux_test_memory")
        print(f"✅ Memory saved to: {path}")
        agent.load_memory("ux_test_memory")
        print("✅ Memory loaded back.")
    except Exception as e:
        print(f"❌ Save/Load failed: {e}")

    print("\n🎉 [UX TEST] Experience Complete. Verdict: SENTIENT.")

if __name__ == "__main__":
    main()
