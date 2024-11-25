import torch
import torchvision
import torchvision.transforms as transforms
from dlquantification.featureextraction.cnn import CNNFeatureExtractionModule
from dlquantification.deepsets import DeepSets

from dlquantification.utils.utils import APPBagGenerator


trainset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

device = torch.device("cuda:0")
torch.manual_seed(2032)

fe = CNNFeatureExtractionModule(output_size=256)
deepsets = DeepSets(
    train_epochs=300,
    test_epochs=1,
    n_classes=10,
    start_lr=0.0001,
    end_lr=0.000001,
    n_bags=100,
    bag_size=500,
    random_seed=2032,
    linear_sizes=[256],
    feature_extraction_module=fe,
    bag_generator=APPBagGenerator(device=device),
    val_bag_generator=APPBagGenerator(device=device),
    batch_size=20,
    quant_loss=torch.nn.L1Loss(),
    lr_factor=0.2,
    patience=20,
    dropout=0.05,
    epsilon=0,
    weight_decay=0,
    use_labels=False,
    val_split=0.2,
    device=device,
    verbose=1,
    dataset_name="test_mnist",
)
deepsets.fit(trainset)
