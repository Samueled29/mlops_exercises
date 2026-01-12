import torch
import pytest
import os


from my_project.data import corrupt_mnist

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "Input data has incorrect shape"
            assert y in range(10), "Target label out of range"
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all(), "Not all classes present in training data"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all(), "Not all classes present in test data"