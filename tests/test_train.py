import pytest
import torch
import os
from my_project.train import train
from my_project.model import MyAwesomeModel
from my_project.data import corrupt_mnist
from torch.utils.data import TensorDataset



def test_training_runs_without_errors(tmp_path, monkeypatch):
    """Test that training completes without crashing."""
    # Change to temporary directory to avoid polluting the workspace
    monkeypatch.chdir(tmp_path)
    
    # Create required directories
    (tmp_path / "models").mkdir()
    (tmp_path / "reports" / "figures").mkdir(parents=True)

    def dummy_corrupt_mnist():
        # Small synthetic dataset to avoid filesystem dependency
        x = torch.zeros((64, 1, 28, 28))
        y = torch.zeros((64,), dtype=torch.long)
        return TensorDataset(x, y), TensorDataset(x, y)

    # Patch both references so train uses the dummy data
    monkeypatch.setattr("my_project.train.corrupt_mnist", dummy_corrupt_mnist)
    monkeypatch.setattr("my_project.data.corrupt_mnist", dummy_corrupt_mnist)
    
    # Run training with minimal configuration
    train(lr=0.001, batch_size=32, epochs=1)
    
    # Verify model checkpoint was created
    assert (tmp_path / "models" / "model.pth").exists(), "Model checkpoint was not saved"

