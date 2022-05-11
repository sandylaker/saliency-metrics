import pytest
import torch
import torch.nn as nn

from saliency_metrics.models import freeze_module, get_module


def test_get_module():
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(2, 2, bias=False),
                nn.ReLU(),
                nn.Linear(2, 3, bias=False),
                nn.ReLU(),
                nn.Linear(3, 2, bias=False),
            )

            self.head = nn.Linear(2, 2, bias=False)

            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.Linear):
                    layer.weight.data.fill_(i)
            self.head.weight.data.fill_(5)

            for p in self.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.head(self.layers(x))

    model = SimpleModel()
    model.eval()

    # test with valid module name
    layer_4 = get_module(model, "layers.4")
    # layer_4 is filled with 4.0
    layer_4_out = layer_4(torch.ones((2, 3)))
    torch.testing.assert_allclose(layer_4_out, torch.full((2, 2), 12.0))

    head = get_module(model, "head")
    # head is filled with 5.0
    head_out = head(torch.ones((2, 2)))
    torch.testing.assert_allclose(head_out, torch.full((2, 2), 10.0))

    # test with empty module name
    full_model = get_module(model, "")
    assert isinstance(full_model.layers, nn.Sequential)
    assert isinstance(full_model.head, nn.Linear)

    # test with non-existing module name
    assert get_module(model, "backbone") is None

    # test with invalid module type
    with pytest.raises(TypeError, match="module can only be a str"):
        _ = get_module(model, module=0)  # type: ignore


def test_freeze_model():
    model_1 = nn.Sequential(nn.Linear(3, 2, bias=False), nn.ReLU(), nn.Linear(2, 1, bias=False))

    freeze_module(model_1, "0", eval_mode=False)
    assert model_1.training
    assert not model_1[0].weight.requires_grad
    assert model_1[2].weight.requires_grad

    model_2 = nn.Sequential(nn.Conv2d(3, 2, 2), nn.BatchNorm2d(2), nn.ReLU())

    freeze_module(model_2, module=None, eval_mode=True)
    assert not model_2.training
    for p in model_2.parameters():
        assert not p.requires_grad
