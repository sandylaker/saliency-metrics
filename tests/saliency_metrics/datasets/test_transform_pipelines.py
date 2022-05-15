import numpy as np
import torch

from saliency_metrics.datasets import build_pipeline


def test_build_pipeline():
    cfg = [
        dict(type="Resize", height=5, width=5),
        dict(type="Normalize", mean=(0.2, 0.3, 0.4), std=(0.2, 0.3, 0.4)),
        dict(type="ToTensorV2"),
    ]
    pipeline = build_pipeline(cfg)
    img = np.ones((8, 8, 3)) * np.array([120, 140, 160])
    img = img.astype(np.uint8)
    img_transformed = pipeline(image=img)["image"]
    expected_values = (torch.tensor([120, 140, 160]) - torch.tensor([0.2, 0.3, 0.4]) * 255) / (
        torch.tensor([0.2, 0.3, 0.4]) * 255
    )
    expected = torch.ones((3, 5, 5), dtype=torch.float32) * expected_values.reshape(3, 1, 1)
    torch.testing.assert_allclose(img_transformed, expected)
