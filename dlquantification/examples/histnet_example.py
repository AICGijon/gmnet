import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from dlquantification.featureextraction.cnn import CNNFeatureExtractionModule
from dlquantification.histnet import HistNet
import os


from dlquantification.utils.utils import APPBagGenerator


trainset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

device = torch.device("cuda:1")
torch.manual_seed(2032)

fe = CNNFeatureExtractionModule(output_size=2)

histnet = HistNet(
    train_epochs=300,
    test_epochs=1,
    n_classes=10,
    start_lr=0.0001,
    end_lr=0.000001,
    n_bags=2,
    bag_size=5,
    n_bins=8,
    random_seed=2032,
    linear_sizes=[256],
    feature_extraction_module=fe,
    bag_generator=APPBagGenerator(device=torch.device("cpu")),
    val_bag_generator=APPBagGenerator(device=torch.device("cpu")),
    batch_size=20,
    quant_loss=torch.nn.L1Loss(),
    lr_factor=0.2,
    patience=20,
    dropout=0.05,
    epsilon=0,
    weight_decay=0,
    histogram="softrbf",
    use_labels=False,
    val_split=0.2,
    device=device,
    verbose=1,
    use_multiple_devices=False,
    batch_normalization=True,
    dataset_name="test_mnist",
)
histnet.fit(trainset)
