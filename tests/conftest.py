"""Shared pytest fixtures for DCAF test suite."""

from dataclasses import dataclass
from typing import Dict, List

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Optional dependency markers
# ---------------------------------------------------------------------------

try:
    import scipy  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import pacmap  # noqa: F401
    _HAS_PACMAP = True
except ImportError:
    _HAS_PACMAP = False

try:
    import sklearn  # noqa: F401
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

requires_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")
requires_pacmap = pytest.mark.skipif(not _HAS_PACMAP, reason="pacmap not installed")
requires_sklearn = pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")


# ---------------------------------------------------------------------------
# Mock signal object
# ---------------------------------------------------------------------------

@dataclass
class MockSignal:
    """Minimal signal object with .id and .cluster for testing."""
    id: str
    cluster: str  # '+', '-', or '0'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_signals_plus_minus_zero() -> List[MockSignal]:
    """Two T+ signals, two T- signals, one T0 signal."""
    return [
        MockSignal(id="t1_pref_target", cluster="+"),
        MockSignal(id="t2_sft_target", cluster="+"),
        MockSignal(id="t6_pref_opposite", cluster="-"),
        MockSignal(id="t7_sft_opposite", cluster="-"),
        MockSignal(id="t11_language", cluster="0"),
    ]


@pytest.fixture
def weight_deltas_3signal() -> Dict[str, torch.Tensor]:
    """Weight deltas for T+, T-, T0 with known properties."""
    torch.manual_seed(42)
    n = 100
    return {
        "t1_pref_target": torch.randn(n) * 0.5 + 0.3,
        "t2_sft_target": torch.randn(n) * 0.5 + 0.2,
        "t6_pref_opposite": torch.randn(n) * 0.5 - 0.3,
        "t7_sft_opposite": torch.randn(n) * 0.5 - 0.2,
        "t11_language": torch.randn(n) * 0.01,
    }


@pytest.fixture
def mock_activations() -> Dict[str, torch.Tensor]:
    """Mock activation tensors [n_samples, dim]."""
    torch.manual_seed(42)
    return {
        "plus": torch.randn(20, 64),
        "minus": torch.randn(20, 64) * -1 + 0.1,
    }


@pytest.fixture
def uniform_effectiveness() -> Dict[str, float]:
    """Equal effectiveness for all signals."""
    return {
        "t1_pref_target": 1.0,
        "t2_sft_target": 1.0,
        "t6_pref_opposite": 1.0,
        "t7_sft_opposite": 1.0,
        "t11_language": 0.5,
    }


# ---------------------------------------------------------------------------
# Tiny neural network for e2e tests
# ---------------------------------------------------------------------------

class TinyTransformer(nn.Module):
    """Minimal 2-layer transformer for testing. No tokenizer needed."""

    def __init__(self, d_model=32, n_heads=4, n_layers=2, vocab_size=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=64, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.encoder(x)
        return self.lm_head(x)


@pytest.fixture
def tiny_model():
    """A tiny 2-layer transformer for e2e tests (CPU, ~50K params)."""
    torch.manual_seed(0)
    model = TinyTransformer()
    model.eval()
    return model


@pytest.fixture
def tiny_model_pair():
    """Two copies of the tiny model — base and 'trained' with perturbed weights."""
    torch.manual_seed(0)
    base = TinyTransformer()
    trained = TinyTransformer()
    trained.load_state_dict(base.state_dict())

    with torch.no_grad():
        for name, param in trained.named_parameters():
            if "weight" in name and param.dim() >= 2:
                param.add_(torch.randn_like(param) * 0.05)

    base.eval()
    trained.eval()
    return base, trained
