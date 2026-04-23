import torch
import torch.nn.functional as F
import math
import pytest
from src.core.polar import PolarQuantizer, TurboQuantOutput

def test_polar_quantizer_cosine_similarity():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Generate random weights, typical for a layer
    W = torch.randn(128, 128, dtype=torch.float32)

    # Initialize quantizer with enough angle bits for high fidelity (e.g. 8 bits for angle)
    # The requirement is > 0.999 cosine similarity
    quantizer = PolarQuantizer(bits_radius=16, bits_angle=8)

    q_out = quantizer.quantize(W)
    W_dequant = quantizer.dequantize(q_out)

    # Flatten for cosine similarity
    cos_sim = F.cosine_similarity(W.flatten().unsqueeze(0), W_dequant.flatten().unsqueeze(0)).item()

    assert cos_sim > 0.999, f"Cosine similarity is {cos_sim}, expected > 0.999"

def test_polar_quantizer_odd_padding():
    torch.manual_seed(42)

    # Odd number of elements
    W = torch.randn(127, 127, dtype=torch.float32)

    quantizer = PolarQuantizer(bits_radius=16, bits_angle=8)
    q_out = quantizer.quantize(W)
    W_dequant = quantizer.dequantize(q_out)

    assert q_out.padding_length == 1
    assert W.shape == W_dequant.shape

    cos_sim = F.cosine_similarity(W.flatten().unsqueeze(0), W_dequant.flatten().unsqueeze(0)).item()
    assert cos_sim > 0.999

def test_polar_quantizer_forward_ste():
    torch.manual_seed(42)
    W = torch.randn(10, 10, dtype=torch.float32, requires_grad=True)

    quantizer = PolarQuantizer(bits_radius=16, bits_angle=8)

    out = quantizer(W)

    # Compute dummy loss
    loss = out.sum()
    loss.backward()

    # Check if gradients flow through
    assert W.grad is not None
    # With STE, W.grad should be a matrix of ones
    assert torch.allclose(W.grad, torch.ones_like(W))

def test_polar_quantizer_dtypes():
    W = torch.randn(10, 10, dtype=torch.float32)

    quantizer_16 = PolarQuantizer(bits_radius=16, bits_angle=4)
    q_out_16 = quantizer_16.quantize(W)
    assert q_out_16.radius.dtype == torch.bfloat16

    quantizer_32 = PolarQuantizer(bits_radius=32, bits_angle=4)
    q_out_32 = quantizer_32.quantize(W)
    assert q_out_32.radius.dtype == torch.float32
