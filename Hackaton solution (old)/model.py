import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, hidden_dim_2=50, dropout_rate=0.2):
        super(FeedForwardNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, 1) # Output x2
        )
        
    def forward(self, x):
        return self.network(x)
