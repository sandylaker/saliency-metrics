import os
import os.path as osp
from functools import partial
from typing import Dict, List

import mmcv
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from saliency_metrics.datasets import ImageFolder, build_dataset, image_folder_collate_fn


@pytest.fixture(scope="module")
def class_names():
    yield ["cat", "dog", "plane"]


@pytest.fixture(scope="module")
def image_folder_factory(tmp_path_factory, class_names):
    def image_folder(same_shape: bool):
        """ImageFolder contains 3 classes, and each class contains 1 image."""
        tmp_path_suffix = "same_shape" if same_shape else "diff_shape"
        tmp_path = tmp_path_factory.mktemp(f"image_folder_{tmp_path_suffix}")
        img_root = tmp_path / "images"
        img_root.mkdir()
        smap_root = tmp_path / "smaps"
        smap_root.mkdir()

        for class_name in class_names:
            img_path = str(img_root / class_name / f"{class_name}_img.jpg")
            img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            mmcv.imwrite(img, file_path=img_path)

            smap_path = str(smap_root / class_name / f"{class_name}_img.png")
            smap_shape = (10, 10) if same_shape else (8, 8)
            smap = np.random.randint(0, 225, smap_shape, dtype=np.uint8)
            mmcv.imwrite(smap, file_path=smap_path)

        return tmp_path

    yield image_folder


@pytest.fixture(scope="module")
def cls_to_ind_file(tmp_path_factory, class_names):
    """A json file.

    The mapping is ``{"plane": 0, "dog": 1, "cat": 2}``
    """
    tmp_path = tmp_path_factory.mktemp("json")
    # deliberately sort class names in reversed order
    class_names = sorted(class_names, reverse=True)
    cls_to_ind = {v: k for k, v in enumerate(class_names)}
    mmcv.dump(cls_to_ind, tmp_path / "cls_to_ind.json")
    yield tmp_path / "cls_to_ind.json"


@pytest.fixture(scope="module")
def image_folder_cfg_factory(image_folder_factory, cls_to_ind_file):
    def image_folder_cfg(with_smap: bool, to_tensor: bool, same_shape: bool, with_cls_to_ind_file: bool) -> Dict:
        """Get a config for ImageFolder.

        Args:
            with_smap: if True, set the ``smap_root`` in the config.
            to_tensor: if True, append ``ToTensorV2`` to the transform pipeline.
            same_shape: if True, the synthesized images will have the same shape as the saliency maps.
            with_cls_to_ind_file: if True, use the ``cls_to_ind_file`` (a JSON file).

        Returns:
            A config dict
        """
        image_folder = image_folder_factory(same_shape)
        img_root = str(image_folder / "images")
        smap_root = str(image_folder / "smaps")

        pipeline = [
            dict(type="Resize", height=5, width=5),
            dict(type="Normalize", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]

        if to_tensor:
            pipeline.append(dict(type="ToTensorV2"))

        cfg = dict(
            type="ImageFolder",
            img_root=img_root,
            smap_root=smap_root if with_smap else None,
            pipeline=pipeline,
            cls_to_ind_file=str(cls_to_ind_file) if with_cls_to_ind_file else None,
        )

        return cfg

    yield image_folder_cfg


def test_image_folder_invalid_smap_extension(image_folder_cfg_factory):
    cfg = image_folder_cfg_factory(with_smap=True, to_tensor=True, same_shape=True, with_cls_to_ind_file=False)
    cfg.update({"smap_extension": ".abc"})

    with pytest.raises(ValueError, match="smap_extension"):
        _ = build_dataset(cfg)


def _test_image_folder(
    dataset_cfg: Dict,
    class_names_list: List[str],
    with_smap: bool,
    with_json: bool,
    to_tensor: bool,
    same_shape: bool,
):
    """Test ``ImageFolder`` class.

    Args:
        dataset_cfg: config dictionary.
        class_names_list: list of class names in the dataset.
        with_smap: if True, assume the dataset also includes saliency maps.
        with_json: if True, use the ``cls_to_ind_file`` JSON file.
        to_tensor: if True, append ``ToTensorV2`` to the transform pipeline.
        same_shape: if True, the synthesized images will have different shapes as the saliency maps.
    """
    dataset = build_dataset(dataset_cfg)
    assert isinstance(dataset, ImageFolder)
    assert len(dataset) == 3
    assert len(dataset.img_paths) == 3

    # the mapping in json file is specially designed,
    # such that class names are mapped to the reversed sorted indices.
    expected_cls_to_ind = {v: k for k, v in enumerate(sorted(class_names_list, reverse=with_json))}
    expected_ind_to_cls = {v: k for k, v in expected_cls_to_ind.items()}
    assert dataset.get_cls_to_ind() == expected_cls_to_ind
    assert dataset.get_ind_to_cls() == expected_ind_to_cls

    if same_shape:
        sample = dataset[0]

        if to_tensor:
            assert isinstance(sample["img"], torch.Tensor)
            assert sample["img"].shape == (3, 5, 5)

            if with_smap:
                assert isinstance(sample["smap"], torch.Tensor)
                assert sample["smap"].shape == (5, 5)
                assert sample["smap"].dtype == torch.uint8

            else:
                assert "smap" not in sample
        else:
            assert isinstance(sample["img"], np.ndarray)
            assert sample["img"].shape == (5, 5, 3)

            if with_smap:
                assert isinstance(sample["smap"], np.ndarray)
                assert sample["smap"].shape == (5, 5)
                assert sample["smap"].dtype == np.uint8
            else:
                assert "smap" not in sample

        img_path = sample["meta"]["img_path"]
        ori_size = sample["meta"]["ori_size"]
        assert sample["target"] == expected_cls_to_ind[osp.basename(osp.dirname(img_path))]
        assert ori_size == (10, 10)
    else:
        if with_smap:
            # images have different shapes as saliency maps. An error should be thrown.
            with pytest.raises(ValueError, match="Saliency map"):
                _ = dataset[0]


@pytest.mark.parametrize("with_smap", [True, False])
@pytest.mark.parametrize("with_json", [True, False])
@pytest.mark.parametrize("to_tensor", [True, False])
@pytest.mark.parametrize("same_shape", [True, False])
def test_image_folder(image_folder_cfg_factory, class_names, with_smap, with_json, to_tensor, same_shape):
    cfg = image_folder_cfg_factory(with_smap, to_tensor, same_shape, with_json)
    _test_image_folder(
        dataset_cfg=cfg,
        class_names_list=class_names,
        with_smap=with_smap,
        with_json=with_json,
        to_tensor=to_tensor,
        same_shape=same_shape,
    )


def test_image_folder_smap_does_not_exist(
    image_folder_cfg_factory,
):
    """Test the case where an image exists but the corresponding saliency map does not."""
    cfg = image_folder_cfg_factory(with_smap=True, to_tensor=True, same_shape=True, with_cls_to_ind_file=True)
    dataset = build_dataset(cfg)

    # add a new image to the dataset but keep the saliency maps the same.
    new_img_path = osp.join(dataset.img_root, dataset.get_ind_to_cls()[0], "extra_img.jpg")
    mmcv.imwrite(np.ones((5, 5, 3), dtype=np.uint8), file_path=new_img_path)
    # update the dataset.img_paths, note that this list stores the relative paths
    dataset.img_paths.append(osp.relpath(new_img_path, dataset.img_root))
    assert len(dataset) == 4
    with pytest.raises(FileNotFoundError, match="Saliency map"):
        _ = dataset[3]

    # clean up
    os.remove(new_img_path)


@pytest.mark.parametrize("smap_as_tensor", [True, False])
def test_image_folder_collate_fn(image_folder_cfg_factory, smap_as_tensor):
    cfg = image_folder_cfg_factory(with_smap=True, to_tensor=True, same_shape=True, with_cls_to_ind_file=True)
    dataset = build_dataset(cfg)
    collate_fn = partial(image_folder_collate_fn, smap_as_tensor=smap_as_tensor)
    data_loader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)
    assert len(data_loader) == 1
    sample = next(iter(data_loader))
    img = sample["img"]
    smap = sample["smap"]
    target = sample["target"]
    meta = sample["meta"]

    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 3, 5, 5)
    if smap_as_tensor:
        assert isinstance(smap, torch.Tensor)
    else:
        assert isinstance(smap, np.ndarray)
    assert smap.shape == (3, 5, 5)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (3,)

    assert isinstance(meta["img_path"], list)
    assert len(meta["img_path"]) == 3
    assert isinstance(meta["img_path"][0], str)
    assert isinstance(meta["ori_size"], list)
    assert len(meta["ori_size"]) == 3
    assert isinstance(meta["ori_size"][0], tuple)
    assert meta["ori_size"][0] == (10, 10)
