import torch
import torch.nn as nn
import math

class QJLTransform(nn.Module):
    def __init__(self, input_dim: int, sketch_dim: int, seed: int = 42):
        super().__init__()
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        self.seed = seed

        # Initialize the random projection matrix
        # Generate with a specific seed to ensure reproducibility
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Create a Gaussian random matrix
        # Usually standard normal distribution is used for random projection, scaled by 1/sqrt(sketch_dim) or not.
        # It won't affect the sign, but good practice.
        S = torch.randn(input_dim, sketch_dim, generator=gen)

        # Store as a persistent buffer
        self.register_buffer('S', S)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Projects the residual error `r` using the random projection matrix `S`
        and returns the 1-bit sketch as a sign vector {-1, 1} in torch.int8 format.

        Args:
            r: Residual error tensor of shape [..., input_dim]

        Returns:
            Sign vector tensor of shape [..., sketch_dim] in torch.int8 format.
        """
        # Ensure r is float for the projection
        r_float = r.to(self.S.dtype)

        # Project
        projected = torch.matmul(r_float, self.S)

        # 1-bit sketch: sign of the projected values
        sign = torch.sign(projected)

        # Map 0 to 1 to ensure output is strictly {-1, 1}
        sign = torch.where(sign == 0, torch.tensor(1.0, dtype=sign.dtype, device=sign.device), sign)

        return sign.to(torch.int8)
