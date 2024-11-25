import torch.nn as nn


class CNNFeatureExtractionModule(nn.Module):
    """This is good for datasets like mnist (28x28x1)"""

    def __init__(self, output_size):
        super(CNNFeatureExtractionModule, self).__init__()

        self.output_size = output_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        bag_size = x.shape[1]
        # We process all the examples in all the samples at the same time
        x = x.view(bag_size * batch_size, *x.shape[2:])
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        x = x.view(batch_size, bag_size, -1)
        output = self.out(x)
        return output


class CNNFeatureExtractionModuleCifar10(nn.Module):
    """This is good for datasets like cifar10"""

    def __init__(self, output_size):
        super(CNNFeatureExtractionModuleCifar10, self).__init__()

        self.output_size = output_size

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.out = nn.Linear(4096, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        bag_size = x.shape[1]
        # We process all the examples in all the samples at the same time
        x = x.view(bag_size * batch_size, *x.shape[2:])
        x = self.conv_layer(x)
        # flatten the output
        x = x.view(x.size(0), -1)
        x = x.view(batch_size, bag_size, -1)
        output = self.out(x)
        return output
