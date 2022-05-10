import os

import pytest
import torch
import torch.nn as nn
from timm.models.efficientnet import EfficientNet
from torchvision.models import ResNet

from saliency_metrics.models.build_classifier import CUSTOM_CLASSIFIERS, _preprocess_cfg, build_classifier, get_module


def _set_torch_home(target_path):
    os.environ["TORCH_HOME"] = target_path


def test_preprocess_cfg():
    cfg_original = {
        "type": "timm.resnet50",
        "num_classes": 1000,
    }
    default_args = {
        "pretrained": True,
        "num_classes": 10,
    }
    cfg_output = _preprocess_cfg(cfg_original, default_args=default_args)
    assert "pretrained" in cfg_output and cfg_output["pretrained"]
    assert "num_classes" in cfg_output and cfg_output["num_classes"] == 1000
    # test if deepcopy works
    assert "pretrained" not in cfg_original


def test_timm_classifier(tmp_path):
    _set_torch_home(target_path=str(tmp_path))
    # without pretrained
    cfg = {
        "type": "timm.efficientnet_b0",
        "pretrained": False,
    }
    classifier = build_classifier(cfg)
    assert isinstance(classifier, EfficientNet)

    # with pretrained
    cfg.update({"pretrained": True})
    classifier = build_classifier(cfg)
    assert isinstance(classifier, EfficientNet)

    # save and then load from checkpoint
    classifier.classifier = nn.Linear(1280, 2)
    ckpt_path = tmp_path.joinpath("hub/checkpoints/b0_2_classes.pth")
    torch.save(classifier.state_dict(), str(ckpt_path))
    assert ckpt_path.exists()

    cfg.update({"pretrained": False, "num_classes": 2, "checkpoint_path": str(ckpt_path)})
    classifier = build_classifier(cfg)
    assert isinstance(classifier, EfficientNet)
    assert classifier.classifier.weight.shape == torch.Size([2, 1280])


def test_torchvision_classifier(tmp_path):
    _set_torch_home(target_path=str(tmp_path))
    # without pretrained
    cfg = {"type": "torchvision.resnet18", "pretrained": False}
    classifier = build_classifier(cfg)
    assert isinstance(classifier, ResNet)

    # with pretrained
    cfg.update({"pretrained": True})
    classifier = build_classifier(cfg)
    assert isinstance(classifier, ResNet)

    classifier.fc = nn.Linear(512, 2)
    ckpt_path = tmp_path.joinpath("hub/checkpoints/resnet18_2_classes.pth")
    torch.save(classifier.state_dict(), str(ckpt_path))
    assert ckpt_path.exists()

    cfg.update({"pretrained": False, "num_classes": 2, "checkpoint_path": str(ckpt_path)})
    classifier = build_classifier(cfg)
    assert isinstance(classifier, ResNet)
    assert classifier.fc.weight.shape == torch.Size([2, 512])


def test_custom_classifier(tmp_path):
    @CUSTOM_CLASSIFIERS.register_module()
    class DummyModel(nn.Module):
        def __init__(self, num_hidden=5, checkpoint_path=None):
            super(DummyModel, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, num_hidden, bias=False), nn.ReLU(), nn.Linear(num_hidden, 2, bias=False)
            )
            self.init_weights(checkpoint_path)

        def init_weights(self, checkpoint_path=None):
            if checkpoint_path is not None:
                ckpt = torch.load(checkpoint_path)
                self.load_state_dict(ckpt)

        def forward(self, x):
            return self.layers(x)

    assert "DummyModel" in CUSTOM_CLASSIFIERS

    # without pretrained
    cfg = {
        "type": "custom.DummyModel",
        "num_hidden": 5,
    }
    classifier = build_classifier(cfg)
    assert isinstance(classifier, DummyModel)

    for layer in classifier.layers:
        if isinstance(layer, nn.Linear):
            layer.weight.data.fill_(0)

    # save and load checkpoint
    tmp_path.joinpath("hub/checkpoints/").mkdir(parents=True)
    ckpt_path = tmp_path.joinpath("hub/checkpoints/dummy_model.pth")
    torch.save(classifier.state_dict(), str(ckpt_path))
    assert ckpt_path.exists()

    cfg.update({"checkpoint_path": ckpt_path})
    classifier = build_classifier(cfg)
    out = classifier(torch.ones(2, 3))
    torch.testing.assert_allclose(out, torch.zeros(2, 2))


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


def test_invalid_type_or_scope():
    cfg = {"pretrained": False}
    with pytest.raises(ValueError, match="Key 'type' must be contained in the config"):
        _ = build_classifier(cfg)

    cfg.update({"type": "resnet50"})
    with pytest.raises(ValueError, match="type must be in format"):
        _ = build_classifier(cfg)

    cfg.update({"type": ".resnet50"})
    with pytest.raises(ValueError, match="scope must not be an empty string"):
        _ = build_classifier(cfg)

    cfg.update({"type": "invalid.resnet50"})
    with pytest.raises(ValueError, match="Invalid scope name"):
        _ = build_classifier(cfg)
