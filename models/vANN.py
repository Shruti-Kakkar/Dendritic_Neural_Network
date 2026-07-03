import torch
import torch.nn as nn

class VanillaANN(nn.Module):
    def __init__(self, input_size=1024, hidden1=256, 
                 hidden2=512, n_classes=10):
        super(VanillaANN, self).__init__()
        
        self.network = nn.Sequential(
            # input → hidden layer 1
            nn.Linear(input_size, hidden1),
            nn.LeakyReLU(negative_slope=0.1),
            
            # hidden layer 1 → hidden layer 2
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(negative_slope=0.1),
            
            # hidden layer 2 → output
            nn.Linear(hidden2, n_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten [batch, 1, 32, 32] → [batch, 1024]
        return self.network(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = VanillaANN()
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")
    
    # quick shape check
    dummy = torch.randn(8, 1, 32, 32)   # fake batch of 8 images
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")  # should be [8, 10]