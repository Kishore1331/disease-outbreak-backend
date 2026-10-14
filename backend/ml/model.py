import torch
import torch.nn as nn

class DengueOutbreakForecaster(nn.Module):
    """
    High-Precision Deep Epidemiological Neural Network for Municipal Ward-Level Disease Outbreak Prediction.
    
    Inputs (7 dimensions):
      1. cases_lag_1: Reported dengue cases 1 week prior (t-1)
      2. cases_lag_2: Reported dengue cases 2 weeks prior (t-2)
      3. cases_lag_3: Reported dengue cases 3 weeks prior (t-3)
      4. temperature_mean_c: Mean weekly ambient temperature in °C
      5. rainfall_sum_mm: Cumulative weekly precipitation in mm
      6. humidity_mean_pct: Mean weekly relative humidity percentage
      7. ward_population: Municipal ward population from Census 2011 PCA
      
    Outputs:
      Predicted weekly dengue case count (continuous non-negative scalar).
    """

    def __init__(self, input_dim: int = 7, hidden_dim1: int = 128, hidden_dim2: int = 64, hidden_dim3: int = 32, output_dim: int = 1):
        super().__init__()
        
        # Stored feature normalization buffers (mean & standard deviation)
        self.register_buffer("feature_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(input_dim, dtype=torch.float32))

        # Deep Residual Architecture
        self.proj_in = nn.Linear(input_dim, hidden_dim1)
        self.norm1 = nn.LayerNorm(hidden_dim1)
        self.act1 = nn.GELU()

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim1, hidden_dim1)
        )
        self.norm_res = nn.LayerNorm(hidden_dim1)
        self.act_res = nn.GELU()

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.norm2 = nn.LayerNorm(hidden_dim2)
        self.act2 = nn.GELU()

        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.norm3 = nn.LayerNorm(hidden_dim3)
        self.act3 = nn.GELU()

        self.head = nn.Linear(hidden_dim3, output_dim)
        self.relu = nn.ReLU()  # Physical constraint: disease case count >= 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Standardize features using registered model buffers
        norm_x = (x - self.feature_mean) / (self.feature_std + 1e-6)

        # Projection & Residual Block
        h = self.act1(self.norm1(self.proj_in(norm_x)))
        res = self.block1(h)
        h = self.act_res(self.norm_res(h + res))

        # Deep Bottleneck Layers
        h = self.act2(self.norm2(self.fc2(h)))
        h = self.act3(self.norm3(self.fc3(h)))
        
        # Positive scalar output
        out = self.relu(self.head(h))
        return out
