# Build Dataset and Load Smap
import torch
from insertion_deletion_metric import InsertionDeletion
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# TODO Discuss config,datasets and dataloader
training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())

# Build dataloader
batch_size = 16
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
batch = {}
for data in train_dataloader:
    batch["img"] = data[0]
    batch["targets"] = data[1]
    smaps = []
    for img in data[0]:
        smap = torch.mean(img, axis=0)
        smaps.append(smap)
    batch["smaps"] = torch.stack(smaps)
    break

# Classifier Config
classifier_config = dict(type="torchvision.resnet18", num_classes=10, pretrained=False)

# Instantiate Class
ins_del = InsertionDeletion(classifier_config, forward_batch_size=32, perturb_step_size=10, summarized=False)
file_path = r"saliency_metrics\metrics\insertion_deletion\results.json"
images = []
smaps = []
for i in range(2):
    img = batch["img"][i]
    smap = batch["smaps"][i]
    target = batch["targets"][i].item()
    single_result = ins_del.evaluate(img, smap, target)
    ins_del.update(single_result)
# TODO Discuss dump
result = ins_del.get_result
result.dump(file_path)
