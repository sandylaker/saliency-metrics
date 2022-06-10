import os
import os.path as osp
from typing import Callable, Dict, List, Optional

import cv2
import mmcv
from torch.utils.data.dataloader import default_collate

from .base_dataset import BaseDataset
from .transform_pipelines import build_pipeline

__all__ = ["ImageFolder", "image_folder_collate_fn"]


IMG_EXTENSIONS = (
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
)


class ImageFolder(BaseDataset):
    """Dataset in the `ImageFolder`_ style.

    Compared to the ``torchvision.datasets.ImageFolder``, this class can load
    an image and its corresponding saliency map (abbreviated as *"smap"*) simultaneously. It is assumed that the
    dataset folder has the following hierarchy:

    .. code-block::

        # images
        root/images/dog/dog_0.jpg
        root/images/dog/dog_1.jpg
        ...

        root/images/cat/cat_0.jpg
        root/images/cat/cat_1.jpg
        ...

        # saliency maps
        root/smaps/dog/dog_0.png
        root/smaps/dog/dog_1.png
        ...

        root/smaps/cat/cat_0.png
        root/smaps/cat/cat_1.png
        ...

    .. note::
        #. An image and its corresponding saliency map must have **the same spatial size**. Please pre-process the
           images and saliency maps in advance.
        #. The file names (without extensions) of an image and its corresponding saliency map must be consistent, e.g.
           ``"dog_0.jpg"`` and ``"dog_0.png"``.

    .. _ImageFolder: https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder     # noqa

    Each sample is a ``dict`` containing following fields:

    - ``"img"``: (``Union[torch.Tensor, numpy.ndarray]``) Transformed image. The image is converted to ``torch.Tensor``
      with shape (num_channels, height, width) if ``ToTensorV2`` (or ``ToTensor``) is in the transform pipeline.
      Otherwise, it is a ``numpy.ndarray`` with shape (height, width, num_channels).
    - ``"smap"``: (``numpy.ndarray``) Saliency map with shape (height, width). This field exists only when ``smap_root``
      is not None.
    - ``"target"``: (``int``) Ground truth label.
    - ``"meta"``: (``dict``) A dictionary containing meta information like image path (with key ``"img_path"``) and
      original size (with key ``"ori_size"``) of the image.

    Args:
        img_root: Root of the image folders.
        pipeline: Config of transform pipeline.
        smap_root: Root of the saliency map folders. If None, no saliency maps will be loaded.
        smap_extension: File extension of the saliency maps. This argument only has influence when ``smap_root`` is not
            None. If ``smap_extension`` is None, then the extension of the images will be used, this assumes that all
            the images have the same extension.
        cls_to_ind_file: Path of a file (json, yaml etc.) that can be de-serialized to a dictionary, which maps
            class names to indices. If None, the class names (folder names under ``img_root``) will be sorted and
            mapped to the sorted indices. For example, ``["a", "b"]`` will be mapped to ``[0, 1]``, respectively.

    Examples:
        .. code-block:: python

            from saliency_metrics.datasets import build_dataset

            pipeline = [
                dict(type="Resize", height=5, width=5),
                dict(type="Normalize", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                dict(type="ToTensorV2",)
            ]

            cfg = dict(
                type="ImageFolder",
                img_root="path/to/data/images/",
                pipeline=pipeline,
                smap_root="path/to/data/smaps/",
                smap_extension=".png",
                cls_to_ind_file="path/to/data/cls_to_ind_file.json",
            )

            dataset = build_dataset(cfg)
            assert isinstance(dataset, ImageFolder)
    """

    def __init__(
        self,
        img_root: str,
        pipeline: List[Dict],
        smap_root: Optional[str] = None,
        smap_extension: Optional[str] = ".png",
        cls_to_ind_file: Optional[str] = None,
    ) -> None:
        super().__init__()

        if smap_root is not None:
            if (smap_extension is not None) and (smap_extension not in IMG_EXTENSIONS):
                raise ValueError(f"smap_extension should be one of {IMG_EXTENSIONS}, but got {smap_extension}")

        self.img_root = img_root
        self.smap_root = smap_root
        self.pipeline: Callable = build_pipeline(pipeline)  # type: ignore

        # each path is relative to self.img_root
        self.img_paths = [v for v in mmcv.scandir(self.img_root, suffix=IMG_EXTENSIONS, recursive=True)]
        img_extension = osp.splitext(self.img_paths[0])[1]
        self.smap_extension = img_extension if smap_extension is None else smap_extension

        if cls_to_ind_file is None:
            cls_names = sorted(os.listdir(img_root))
            self._cls_to_ind = {c: i for i, c in enumerate(cls_names)}
        else:
            cls_to_ind = mmcv.load(cls_to_ind_file)
            # in case of indices are str when being loaded from the file
            self._cls_to_ind = {k: int(v) for k, v in cls_to_ind.items()}
        self._ind_to_cls = {v: k for k, v in self._cls_to_ind.items()}

    def __getitem__(self, index: int) -> Dict:
        rel_img_path = self.img_paths[index]
        full_img_path = osp.join(self.img_root, rel_img_path)
        img = cv2.imread(full_img_path)
        ori_size = img.shape[:2]
        meta = dict(img_path=full_img_path, ori_size=ori_size)
        if img.ndim == 3:
            # 3-channel image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.smap_root is not None:
            # the saliency maps might have different extension as the images
            rel_smap_path = osp.splitext(rel_img_path)[0] + self.smap_extension
            full_smap_path = osp.join(self.smap_root, rel_smap_path)
            if not osp.exists(full_smap_path):
                raise FileNotFoundError(
                    f"Saliency map: {full_smap_path} does not exist for the image: {full_img_path}."
                )
            smap = cv2.imread(full_smap_path, cv2.IMREAD_UNCHANGED)
            if smap.shape[:2] != ori_size:
                raise ValueError(
                    f"Saliency map:{full_smap_path} has shape: {smap.shape[:2]}, "
                    f"but the image: {full_img_path} has shape: {ori_size}."
                )
        else:
            smap = None

        transformed = self.pipeline(image=img) if smap is None else self.pipeline(image=img, mask=smap)

        img_folder, img_name_with_ext = osp.split(full_img_path)
        cls_name = osp.basename(img_folder)
        target = self._cls_to_ind[cls_name]
        result = {"target": target, "meta": meta, "img": transformed["image"]}
        if smap is not None:
            result.update({"smap": transformed["mask"]})

        return result

    def __len__(self) -> int:
        return len(self.img_paths)

    def get_ind_to_cls(self) -> Dict[int, str]:
        return self._ind_to_cls

    def get_cls_to_ind(self) -> Dict[str, int]:
        return self._cls_to_ind


def image_folder_collate_fn(batch: List[Dict], smap_as_tensor: bool = True) -> Dict:
    """Collate function for :class:`saliency_metrics.datasets.image_folder.ImageFolder`.

    The collated batch is a dict that contains:

    * ``"img"``: (``Tensor``) images with shape (batch_size, num_channels, height, width).
    * ``"target"``: (``Tensor``) targets with shape (batch_size,).
    * ``"smap"``: (``Optional[Union[Tensor, ndarray]]``) saliency maps with shape (batch_size, height, width).
    * ``"meta"``: A dict that contains:

      * ``"img_path"``: (``List[str]``) image paths with the length of batch_size.
      * ``"ori_size"``: (``List[Tuple[int, int]]``) list of original spatial sizes of images.

    Args:
        batch: A batch of data with length of batch_size.
        smap_as_tensor: If True, batch the saliency maps to a ``torch.Tensor``. Otherwise, batch them to
            a ``numpy.ndarray``.

    Returns:
        Collated batch.
    """
    has_smap = "smap" in batch[0]

    collated_batch = default_collate(batch)
    if has_smap and not smap_as_tensor:
        collated_batch.update({"smap": collated_batch["smap"].numpy()})

    ori_height, ori_width = collated_batch["meta"]["ori_size"]
    ori_size = list(zip(ori_height.tolist(), ori_width.tolist()))
    collated_batch["meta"].update(ori_size=ori_size)
    return collated_batch
