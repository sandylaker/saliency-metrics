import torch
from sensn_reinference import SensitivityN
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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
sensn = SensitivityN(classifier_config, 1, 0.01, summarized=True, num_masks=100)
file_path = r"sensitivity_n\results.json"
images = []
smaps = []
for i in range(5):
    img = batch["img"][i]
    smap = batch["smaps"][i]
    target = batch["targets"][i].item()
    single_result = sensn.evaluate(img, smap, target)
    sensn.update(single_result)
    sensn.increment_n()
    sensn.get_result().dump(file_path)

# sensn.dump(file_path)
