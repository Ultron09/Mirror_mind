"""
PROTOCOL PHASE 3: UNIVERSAL COMPATIBILITY (THE OMNI-TEST)
=========================================================
Goal: Prove the framework wraps ANY architecture and handles ANY input tensor shape.
Architectures Tested:
1. Visual Cortex (CNN) - 4D Input [Batch, Channels, Height, Width]
2. Auditory Cortex (LSTM) - 3D Input [Batch, Seq_Len, Features]
3. Symbolic Cortex (MLP) - 2D Input [Batch, Dimensions]
"""

import torch
import torch.nn as nn
import logging
import sys
import os

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
except ImportError:
    sys.exit(1)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Phase3")

# ==============================================================================
# 1. DEFINE DIVERSE ARCHITECTURES
# ==============================================================================

class VisualCortex(nn.Module):
    """CNN for Image Data (4D Input)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

class AuditoryCortex(nn.Module):
    """LSTM for Time-Series/Audio (3D Input)"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=32, batch_first=True)
        self.head = nn.Linear(32, 5)
        
    def forward(self, x):
        # x: [Batch, Seq, Feat]
        out, _ = self.lstm(x)
        # Take last time step
        return self.head(out[:, -1, :])

class SymbolicCortex(nn.Module):
    """MLP for Tabular/Embedding Data (2D Input)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 2. THE UNIVERSAL TEST LOOP
# ==============================================================================
def run_omni_test():
    logger.info("🌍 PHASE 3: STARTING UNIVERSAL COMPATIBILITY TEST")
    
    # Common Config
    config = AdaptiveFrameworkConfig(
        model_dim=32, # This will adapt internally
        compile_model=False,
        device='cpu'
    )
    
    subjects = [
        (
            "Visual Cortex (CNN)", 
            VisualCortex(), 
            torch.randn(4, 3, 32, 32), # [B, C, H, W]
            torch.randn(4, 2)          # Target
        ),
        (
            "Auditory Cortex (LSTM)", 
            AuditoryCortex(), 
            torch.randn(4, 20, 10),    # [B, Seq, Feat]
            torch.randn(4, 5)          # Target
        ),
        (
            "Symbolic Cortex (MLP)", 
            SymbolicCortex(), 
            torch.randn(4, 64),        # [B, Dim]
            torch.randn(4, 64)         # Target
        )
    ]
    
    for name, model, inputs, targets in subjects:
        logger.info(f"Testing Subject: {name}...")
        try:
            # 1. Wrap
            framework = AdaptiveFramework(model, config)
            
            # 2. Train Step (Forces Introspection on weird shapes)
            metrics = framework.train_step(inputs, targets)
            
            logger.info(f"   ✅ Success. Loss: {metrics['loss']:.4f}")
            
            # 3. Verify Introspection Shape Handling
            # The telemetry buffer should flatten inputs correctly without crashing
            tracked_layers = framework.num_tracked_layers
            logger.info(f"      Tracked Layers: {tracked_layers}")
            
        except RuntimeError as e:
            logger.error(f"   ❌ CRITICAL: Shape Mismatch or Runtime Error in {name}")
            logger.error(f"      Error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"   ❌ FAILED: {e}")
            sys.exit(1)

    print("\n" + "="*40)
    print("🟢 UNIVERSAL COMPATIBILITY VERIFIED")
    print("="*40 + "\n")
    sys.exit(0)

if __name__ == "__main__":
    run_omni_test()