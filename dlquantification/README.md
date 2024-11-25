# GMNet
GMNet is a deep neural network for quantification.

### Installation steps
First, clone the project:
```
git clone https://github.com/AICGijon/gmnet
```

The best way to install GMNet and the dependencies is using an Anaconda python distribution and a conda virtual enviroment.

**---> Option 1**: Create a conda enviroment and activate it:
```sh
conda create --name gmnet
conda activate gmnet
```
Install the dependencies, that means, installing pytorch (there is no other extra dependency):

```bash
conda install pytorch torchvision cudatoolkit=10.2 pandas tensorboard -c pytorch
```
Actual version of cuda depends on your machine.

**---> End Option 1**

**---> Option 2**: Another alternative is to install all the exact dependencies that we have used for coding gmnet. (replace by the full path of the environment in your machine):
```
conda env create -f environment.yml
```
**---> End Option 2**

Add DLQuantification project to the conda enviroment, it is required to install conda-build. (replace by the full path of the project in your machine):
```
conda develop /media/nas/pgonzalez/DLquantification #replace as neccesary
```

Once you have pytorch installed, you are ready to go. You can check that GMNet is working by executing the example script in gmnet_example.py. That will train the network against the MNIST dataset.

```
python dlquantification/examples/gmnet_example.py
```

## Getting started

You can also test other Deep Learning Models for Quantification such as Histnet. Here there a basic example of how to use HistNet.
```python
import torch
import torchvision
import torchvision.transforms as transforms
from dlquantification.featureextraction.cnn import CNNFeatureExtractionModule
from dlquantification.histnet import HistNet
from dlquantification.utils.utils import AppBagGenerator

trainset = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = transforms.ToTensor())

device = torch.device('cuda:1')

fe = CNNFeatureExtractionModule(output_size=256)
histnet = HistNet(train_epochs=5000,test_epochs=1,start_lr=0.001,end_lr=0.000001,n_bags=500,bag_size=16,n_bins=8,random_seed = 2032,linear_sizes=[256],
                    feature_extraction_module=fe,bag_generator=APPBagGenerator(device=device),batch_size=16,quant_loss=torch.nn.L1Loss(),lr_factor=0.2, patience=20, dropout=0.05,epsilon=0,weight_decay=0,histogram='softrbf',use_labels=False,val_split=0.2,device=device,verbose=1,dataset_name="test_mnist")
histnet.fit(trainset)
```
This example will fit the network to the MNIST dataset. Check histnet/histnet_example.py.

## Sample generation
GMNet needs samples for training and validating, and also for testing. Samples are generated differently depending on the type of problem. 

### Training and validation samples
In the case of training and validation samples we can have two different scenarios:
1. Samples are not provided by the problem (we only have a training set with all the examples together). In this case we need to artificially generate the samples. For generating the samples we generate a random vector of prevalences (that add up to one) and then we take for training (or validation) dataset, the number of examples required to match these prevalences. Note that this sampling is done with replacement to make sure that we do not get problems with classes with few examples. 
2. Samples are provided by the problem (E.g. the plankton problem). This approach takes into account that some samples are more probable than others. For instance in the plankton problem, the mix class is the prevalent one, so it is not likely that this class has a low prevalence in any sample in the test set. Taking this into account, samples are generated making a variation of a real sample, modifying the prevalence of one random class and adjusting the prevalence of the rest of the classes to add up to one. Once we have the dessired prevalences we draw the examples from the whole training set (this is an important decission because we are creating new samples with examples of different real samples).

### Test samples
For test samples we have two possible scenarios as well:
1. Samples are not provided. In this case we also need the generate the test samples artificially. Normally we will use the same sample size that for training. The process will be equivalent to the one described before for training samples but obviously using test data.
2. We have test samples (E.g. the plankton problem). In this case we will use the test samples to test the algorithm. The problem is that test samples will have different sizes. The easiest idea is to generate random subsamples for each sample with the training size and then compute the mean of the predicted prevalences. That would be a nice estimation of the test sample prevalence.
