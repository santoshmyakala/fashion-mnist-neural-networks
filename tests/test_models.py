import pytest
import torch

from fmnist.models import MODELS
from fmnist.train import augment


@pytest.mark.parametrize("name", sorted(MODELS))
def test_output_shape(name):
    model = MODELS[name]().eval()
    assert model(torch.randn(4, 1, 28, 28)).shape == (4, 10)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_can_overfit_a_tiny_batch(name):
    """A sanity check that gradients flow end to end: memorise 32 random images."""
    torch.manual_seed(0)
    model = MODELS[name]()
    x, y = torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    assert (model(x).argmax(1) == y).float().mean() > 0.9


def test_augment_preserves_shape():
    x = torch.randn(8, 1, 28, 28)
    assert augment(x, torch.Generator().manual_seed(0)).shape == x.shape
