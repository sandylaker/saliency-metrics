data_root = "data/food101/"
dataset_type = "ImageFolder"

img_norm_cfg = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

train_pipeline = [
    dict(type="RandomResizedCrop", height=224, width=224),
    dict(type="Flip", p=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ToTensorV2"),
]

test_pipeline = [
    dict(type="LongestMaxSize", max_size=256),
    dict(type="PadIfNeeded", min_height=224, min_width=224),
    dict(type="CenterCrop", height=224, width=224),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ToTensorV2"),
]

data = dict(
    data_loader=dict(batch_size=256, num_workers=8, shuffle=True),
    train=dict(
        type="ImageFolder",
        img_root=data_root + "train/",
        pipeline=train_pipeline,
        smap_root=data_root + "smaps/train/",
        cls_to_ind_file=data_root + "meta/cls_to_ind.json",
    ),
    val=dict(
        type="ImageFolder",
        img_root=data_root + "val/",
        pipeline=test_pipeline,
        smap_root=data_root + "smaps/val/",
        cls_to_ind_file=data_root + "meta/cls_to_ind.json",
    ),
    test=dict(
        type="ImageFolder",
        img_root=data_root + "test/",
        pipeline=test_pipeline,
        smap_root=data_root + "smaps/test/",
        cls_to_ind_file=data_root + "meta/cls_to_ind.json",
    ),
)
