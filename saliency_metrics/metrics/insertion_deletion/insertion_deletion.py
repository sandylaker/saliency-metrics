from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from saliency_metrics.metrics.insertion_deletion import InsertionDeletion


def run_insertion_deletion() -> None:
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
        break

    # Classifier Config
    classifier_config = dict(type="torchvision.resnet18", num_classes=10, pretrained=False)

    # Instantiate Class
    ins_del = InsertionDeletion(classifier_config, forward_batch_size=32, perturb_step_size=10, summarized=False)
    # TODO - change file path
    file_path = r"saliency_metrics\metrics\insertion_deletion\results.json"
    for i in range(1):
        # img = batch["img"][i]
        # smap = batch["smaps"][i]
        # target = batch["targets"][i].item()
        img = torch.linspace(0, 1, 3072, dtype=torch.float32).reshape(3, 32, 32)
        smap = torch.linspace(0, 1, 1024, dtype=torch.float32).reshape(32, 32)
        target = 3
        img_path = r"user\somepath"
        single_result = ins_del.evaluate(img, smap, target, img_path)
        ins_del.update(single_result)
    result = ins_del.get_result
    result.dump(file_path)


if __name__ == "__main__":
    run_insertion_deletion()
