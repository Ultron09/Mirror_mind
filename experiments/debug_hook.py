import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 32)
        
    def forward(self, x):
        return self.fc(x)

def hook(module, input, output):
    print(f"Hook: Input {input[0].shape}, Output {output.shape}")

model = SimpleModel()
model.fc.register_forward_hook(hook)

x = torch.randn(5, 10)
y = model(x)
print("Done")
