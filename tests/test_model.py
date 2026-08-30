import pytest
import torch

from model import SimpleCNN, get_model


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=10)
    batch = torch.randn(4, 1, 28, 28)
    output = model(batch)
    assert output.shape == (4, 10)


def test_simple_cnn_default_num_classes():
    model = SimpleCNN()
    batch = torch.randn(2, 1, 28, 28)
    output = model(batch)
    assert output.shape == (2, 10)


def test_get_model_returns_simple_cnn():
    model = get_model(architecture="simple_cnn", num_classes=5)
    assert isinstance(model, SimpleCNN)
    output = model(torch.randn(1, 1, 28, 28))
    assert output.shape == (1, 5)


def test_get_model_unsupported_architecture_raises():
    with pytest.raises(ValueError):
        get_model(architecture="resnet18", num_classes=10)
