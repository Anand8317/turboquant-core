import torch
import torch.nn as nn
from dataclasses import dataclass
import math

@dataclass
class TurboQuantOutput:
    radius: torch.Tensor
    angle_indices: torch.Tensor
    original_shape: torch.Size
    padding_length: int

class PolarQuantizer(nn.Module):
    def __init__(self, bits_radius: int = 16, bits_angle: int = 3, strategy: str = 'symmetric'):
        super().__init__()
        self.bits_radius = bits_radius
        self.bits_angle = bits_angle
        self.strategy = strategy

        # Determine radius dtype
        if self.bits_radius == 16:
            self.radius_dtype = torch.bfloat16
        elif self.bits_radius == 32:
            self.radius_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported bits_radius: {bits_radius}. Supported values are 16, 32.")

        self.num_angle_bins = 2 ** self.bits_angle

    def quantize(self, x: torch.Tensor) -> TurboQuantOutput:
        original_shape = x.shape
        x_flat = x.flatten()

        padding_length = 0
        if x_flat.numel() % 2 != 0:
            padding_length = 1
            x_flat = torch.cat([x_flat, torch.zeros(1, dtype=x_flat.dtype, device=x_flat.device)])

        x_pairs = x_flat.view(-1, 2)

        # Cartesian to Polar (compute using original high precision, e.g. float32)
        # Assuming input is float16/bfloat16/float32, converting to float32 for math.
        x_pairs_f32 = x_pairs.to(torch.float32)
        r = torch.norm(x_pairs_f32, dim=-1)
        theta = torch.atan2(x_pairs_f32[..., 1], x_pairs_f32[..., 0])

        # Quantize angle
        # theta is in [-pi, pi]
        # Normalize to [0, 1]
        theta_norm = (theta + math.pi) / (2 * math.pi)

        # Quantize to [0, num_bins - 1]
        theta_indices = torch.round(theta_norm * (self.num_angle_bins - 1))
        # Ensure indices are within bounds due to numerical issues
        theta_indices = torch.clamp(theta_indices, 0, self.num_angle_bins - 1).to(torch.int32)

        # Radius to desired dtype
        r_quant = r.to(self.radius_dtype)

        return TurboQuantOutput(
            radius=r_quant,
            angle_indices=theta_indices,
            original_shape=original_shape,
            padding_length=padding_length
        )

    def dequantize(self, q_out: TurboQuantOutput) -> torch.Tensor:
        r = q_out.radius.to(torch.float32)
        theta_indices = q_out.angle_indices.to(torch.float32)

        # Dequantize angle
        theta_norm = theta_indices / (self.num_angle_bins - 1)
        theta = theta_norm * 2 * math.pi - math.pi

        # Polar to Cartesian
        x_dim0 = r * torch.cos(theta)
        x_dim1 = r * torch.sin(theta)

        x_pairs = torch.stack([x_dim0, x_dim1], dim=-1)
        x_flat = x_pairs.flatten()

        if q_out.padding_length > 0:
            x_flat = x_flat[:-q_out.padding_length]

        return x_flat.view(q_out.original_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass with Straight-Through Estimator for quantization
        q_out = self.quantize(x)
        x_dequant = self.dequantize(q_out)

        # Straight-Through Estimator: Gradients flow through unmodified, but values are quantized
        x_dequant = x_dequant.to(x.dtype)
        return x + (x_dequant - x).detach()
