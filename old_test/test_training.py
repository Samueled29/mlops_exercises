# tests/test_training.py
import pytest
import torch
from my_project.train import train
from my_project.model import MyAwesomeModel
from my_project.data import corrupt_mnist

def test_training_saves_model_and_figure(tmp_path, monkeypatch):
    """
    Test that training saves model and figure.
    """
    # Change working directory to temporary path
    monkeypatch.chdir(tmp_path)

    # Create necessary directories
    (tmp_path / "models").mkdir()
    (tmp_path / "reports" / "figures").mkdir(parents=True)

    # Run training with reduced parameters for speed
    train(epochs=1, batch_size=16)

    # Check that files were created
    assert (tmp_path / "models" / "model.pth").exists()
    assert (tmp_path / "reports" / "figures" / "training_statistics.png").exists()


def test_model_can_learn_on_small_dataset():
    """
    Test that model can learn on a small dataset.
    """
    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()

    # Take only 32 examples to easily overfit
    small_dataset = torch.utils.data.Subset(train_set, range(32))
    dataloader = torch.utils.data.DataLoader(small_dataset, batch_size=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # initial loss
    img, target = next(iter(dataloader))
    model.eval()
    with torch.no_grad():
        initial_loss = loss_fn(model(img), target).item()

    # Quick training on a few batches
    model.train()
    for _ in range(3):
        for img, target in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(model(img), target)
            loss.backward()
            optimizer.step()

    # final loss
    model.eval()
    with torch.no_grad():
        final_loss = loss_fn(model(img), target).item()

    assert final_loss < initial_loss, "Il modello non è riuscito a imparare sul piccolo dataset"




