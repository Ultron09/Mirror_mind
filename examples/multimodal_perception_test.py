import torch
import torch.nn as nn
import dataclasses
import inspect
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
import airbornehrs.core

def test_multimodal_perception():
    print("Testing Multi-Modal Perception Interface...")
    print(f"  Module file: {airbornehrs.core.__file__}")
    print(f"  AdaptiveFrameworkConfig fields: {[f.name for f in dataclasses.fields(AdaptiveFrameworkConfig)]}")
    print(f"  AdaptiveFrameworkConfig __init__ signature: {inspect.signature(AdaptiveFrameworkConfig.__init__)}")
    
    # 1. Define a simple base model that accepts the fused latent sequence
    # PerceptionGateway projects vision and audio into [B, Seq, model_dim]
    # ModalityFuser concatenates them.
    # For vision (patch_size=16, img=224x224) -> Seq = (224/16)^2 = 196
    # For audio (mel=80, time=100) -> Seq = 100
    # Total Seq = 196 + 100 = 296
    
    model_dim = 256
    class SimpleMind(nn.Module):
        def __init__(self):
            super().__init__()
            self.processor = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=model_dim, nhead=8, batch_first=True),
                num_layers=2
            )
            self.head = nn.Linear(model_dim, 10)
            
        def forward(self, x):
            # x is the fused latent [B, Seq, model_dim]
            x = self.processor(x)
            # Global average pooling over sequence
            x = x.mean(dim=1)
            return self.head(x)

    # 2. Configure Framework
    config = AdaptiveFrameworkConfig(
        model_dim=model_dim,
        enable_perception=True,
        vision_dim=3,
        audio_dim=80,
        enable_consciousness=True
    )
    
    base_model = SimpleMind()
    framework = AdaptiveFramework(base_model, config=config)
    
    # 3. Prepare Mock Multi-Modal Input
    batch_size = 4
    mock_inputs = {
        'image': torch.randn(batch_size, 3, 224, 224),
        'audio': torch.randn(batch_size, 80, 100)
    }
    mock_target = torch.randint(0, 10, (batch_size,))
    
    # 4. Test Predict
    print("  Testing Predict...")
    with torch.no_grad():
        output, log_var, affine = framework(mock_inputs)
        print(f"    Output shape: {output.shape}")
        assert output.shape == (batch_size, 10)
    
    # 5. Test Train Step
    print("  Testing Train Step...")
    metrics = framework.train_step(mock_inputs, target_data=mock_target)
    print(f"    Loss: {metrics['loss']:.4f}")
    print(f"    Surprise: {metrics.get('surprise', 0.0):.4f}")
    
    # 6. Verify Consciousness Observation
    if framework.consciousness:
        print("    Consciousness is active.")
        # Check if _last_fused_latent was stored
        if hasattr(framework, '_last_fused_latent') and framework._last_fused_latent is not None:
            print(f"    Fused Latent observed: {framework._last_fused_latent.shape}")
            assert framework._last_fused_latent.shape[0] == batch_size
            assert framework._last_fused_latent.shape[-1] == model_dim
        else:
            print("    Fused Latent NOT observed by framework!")
            
    print("Multi-Modal Perception Interface Verified!")

if __name__ == "__main__":
    test_multimodal_perception()
