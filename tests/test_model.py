"""Unit tests for src/model.py.

Run with:
    cd src && python -m pytest ../tests -v
(or add `src` to PYTHONPATH; see .github/workflows/ci.yml for the CI setup)
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import SimpleCNN, get_model  # noqa: E402


@pytest.mark.parametrize("architecture", ["simple_cnn", "resnet18"])
def test_get_model_returns_correct_output_shape(architecture):
    model = get_model(architecture=architecture, num_classes=10)
    model.eval()
    batch = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (4, 10)


def test_get_model_respects_num_classes():
    model = get_model(architecture="simple_cnn", num_classes=5)
    batch = torch.randn(2, 3, 32, 32)
    output = model(batch)
    assert output.shape == (2, 5)


def test_unknown_architecture_raises():
    with pytest.raises(ValueError):
        get_model(architecture="not_a_real_model", num_classes=10)


def test_simple_cnn_is_trainable():
    """One optimizer step should change at least one weight tensor."""
    model = SimpleCNN(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    before = model.classifier[-1].weight.clone()

    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))

    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()

    after = model.classifier[-1].weight
    assert not torch.equal(before, after)
