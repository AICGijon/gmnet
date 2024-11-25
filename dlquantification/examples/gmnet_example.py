import numpy as np
from torch.utils.data import TensorDataset
import torch
from dlquantification.featureextraction.nofe import NoFeatureExtractionModule
from dlquantification.histnet import HistNet
from dlquantification.gmnet import GMNet
from dlquantification.utils.utils import APPBagGenerator
from dlquantification.featureextraction.fullyconnected import FCFeatureExtractionModule
from dlquantification.featureextraction.cnn import CNNFeatureExtractionModule
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision import datasets, transforms

np.random.seed(2032)
torch.manual_seed(2032)

# num_samples = 5000
n_classes = 10
n_features = 100
n_gaussians = [50,50]
n_dimensions = [3,4]
n_gm_layers = 2
n_bags=200
bag_size=300
device = torch.device("cuda:0")


# fe = NoFeatureExtractionModule(input_size=n_features)
# fe = FCFeatureExtractionModule(input_size=n_features, output_size=n_features // 2, hidden_sizes=[1024, 1024])
fe = CNNFeatureExtractionModule(output_size=n_features)

train = datasets.FashionMNIST(
    "data/", download=True, train=True, transform=transforms.ToTensor()
)
test = datasets.FashionMNIST(
    "data/", download=True, train=False, transform=transforms.ToTensor()
)

x_train = np.zeros((60000, 28, 28))
x_test = np.zeros((10000, 28, 28))
y_train = np.zeros((60000), dtype=np.int32)
y_test = np.zeros((10000), dtype=np.int32)

for i in range(len(train)):
    x_train[i, :, :] = train[i][0]
    y_train[i] = train[i][1]

for i in range(len(test)):
    x_test[i, :, :] = test[i][0]
    y_test[i] = test[i][1]

x_train = np.expand_dims(x_train, axis=1)  # add the channel dimension
x_test = np.expand_dims(x_test, axis=1)

trainset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
testset = TensorDataset(torch.Tensor(x_test))

gmnet = GMNet(
    train_epochs=1000,
    test_epochs=1,
    n_classes=n_classes,
    start_lr=0.0001,
    end_lr=0.000001,
    n_bags=n_bags,
    bag_size=bag_size,
    num_gaussians=n_gaussians,
    n_gm_layers= n_gm_layers,
    gaussian_dimensions=n_dimensions,
    random_seed=2032,
    linear_sizes=[1024], 
    feature_extraction_module=fe,
    bag_generator=APPBagGenerator(device=device),
    val_bag_generator=APPBagGenerator(device=device),
    batch_size=50,
    cka_regularization=1,
    quant_loss=torch.nn.L1Loss(),
    lr_factor=0.2,
    patience=20,
    dropout=0,
    epsilon=0,
    weight_decay=0,
    use_labels=False,
    val_split=0.2,
    device=device,
    verbose=10,
    dataset_name="test_fashionmnist",
    wandb_experiment_name=f"gmnet",
    use_wandb=False,
)

#gmnet.fit(trainset)


testSampleGenerator = APPBagGenerator(device='cpu')
samples, prevalences = testSampleGenerator.compute_bags(
    n_bags=n_bags, bag_size=bag_size, y=y_test
)
#Create a dataset with all the test bags
tensor = torch.empty(n_bags, bag_size, 1, 28, 28)
for i, sample in enumerate(samples):
    tensor[i,:] = torch.from_numpy(x_test[sample, :])

dataset = TensorDataset(tensor)
print(gmnet.predict(dataset,process_in_batches=20))