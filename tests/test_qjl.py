import torch
import math
import pytest
from src.core.qjl import QJLTransform

def test_qjl_transform_output():
    input_dim = 128
    sketch_dim = 16
    qjl = QJLTransform(input_dim=input_dim, sketch_dim=sketch_dim, seed=42)

    r = torch.randn(10, input_dim)
    q = qjl(r)

    # Check shape
    assert q.shape == (10, sketch_dim)

    # Check dtype
    assert q.dtype == torch.int8

    # Check values are strictly {-1, 1}
    unique_vals = torch.unique(q)
    for val in unique_vals:
        assert val.item() in [-1, 1]

def test_qjl_inner_product_unbiased():
    input_dim = 10000
    sketch_dim = 4000
    qjl = QJLTransform(input_dim=input_dim, sketch_dim=sketch_dim, seed=123)

    # Generate two random residual vectors
    torch.manual_seed(456)
    r1 = torch.randn(input_dim)
    r2 = torch.randn(input_dim)

    # Correlate r2 with r1 so they are not orthogonal
    r2 = 0.5 * r1 + 0.866 * r2

    q1 = qjl(r1)
    q2 = qjl(r2)

    # True inner product
    true_ip = torch.dot(r1, r2).item()

    # Norms of the original vectors (this would be transmitted separately or preserved)
    norm1 = torch.norm(r1).item()
    norm2 = torch.norm(r2).item()

    # Inner product of 1-bit sketches
    # Cast to float to avoid overflow during dot product and division
    q1_float = q1.float()
    q2_float = q2.float()
    dot_q = torch.dot(q1_float, q2_float).item()

    # Cosine estimator formula
    # ||r1|| ||r2|| cos(pi * (1 - (q1 · q2) / d) / 2)
    estimated_ip = norm1 * norm2 * math.cos(math.pi * (1 - dot_q / sketch_dim) / 2)

    # Check relative error
    rel_error = abs(true_ip - estimated_ip) / abs(true_ip)

    # Since this is a randomized estimator, we check if it is reasonably close
    # With d=4000, the error should be quite small
    assert rel_error < 0.1, f"Estimated inner product {estimated_ip} is too far from true {true_ip}. Rel error: {rel_error}"
