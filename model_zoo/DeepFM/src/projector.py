import torch
import torch.nn as nn

class FeatureProjector(nn.Module):
    def __init__(self, feature_dim=16, llm_dim=1024, hidden_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
            nn.LayerNorm(llm_dim)
        )
    
    def forward(self, x):
        # x: [batch, num_fields, feature_dim]
        # returns: [batch, num_fields, llm_dim]
        return self.projection(x)
