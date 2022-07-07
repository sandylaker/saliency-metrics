from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from saliency_metrics.metrics.insertion_deletion.insertion_deletion_metric import InsertionDeletion


def run_insertion_deletion(work_dir: str) -> None:
    training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())

    # Build dataloader
    batch_size = 16
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    batch: Dict[str, torch.tensor] = {}
    for data in train_dataloader:
        batch["img"] = data[0]
        batch["targets"] = data[1]
        smaps: List[torch.tensor] = []
        for img in data[0]:
            smap = torch.mean(img, axis=0)
            smaps.append(smap)
        batch["smaps"] = torch.stack(smaps)
        # preparing data for one batch
        break

    # Classifier Config
    classifier_config = dict(type="torchvision.resnet18", num_classes=10, pretrained=False)

    # Instantiate Class
    ins_del = InsertionDeletion(classifier_config, forward_batch_size=32, perturb_step_size=10, summarized=False)
    file_path = work_dir + r"\results.json"
    for i in range(1):
        img = torch.unsqueeze(batch["img"][i], dim=0)
        smap = batch["smaps"][i]
        target = batch["targets"][i].item()
        target = 3
        img_path = r"user\somepath"
        single_result = ins_del.evaluate(img, smap, target, img_path=img_path)
        ins_del.update(single_result)
    result = ins_del.get_result
    result.dump(file_path)
