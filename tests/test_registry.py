"""
Test registry functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.registry import MODEL_REGISTRY, register_model
from model.lstm import LSTMClassifier


def test_model_registration():
    """Test that models are registered correctly."""
    assert "lstm" in MODEL_REGISTRY
    assert MODEL_REGISTRY["lstm"] is LSTMClassifier
    print("Model registry test passed!")


if __name__ == "__main__":
    test_model_registration()


