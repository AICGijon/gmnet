import torch.nn as nn
import torchvision
import torch


class CNNResnetFeatureExtractionModule(nn.Module):
    def __init__(self, output_size, pretrained=True, model_path=None, train_resnet=True):
        super(CNNResnetFeatureExtractionModule, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=pretrained)

        if model_path is not None:
            state_dict = torch.load(model_path)
            # We need to remove the prefix because it was saved with dataparallel
            state_dict = {k.partition("module.")[2]: v for k, v in state_dict.items()}
            # Create the original structure of the saved model
            last_layer_size = len(state_dict["fc.bias"])
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, last_layer_size)
            self.resnet.load_state_dict(state_dict)

        if model_path is None or last_layer_size != output_size:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)

        self.set_train_resnet(train_resnet)

        self.output_size = output_size

    def set_train_resnet(self, train_resnet):
        self.train_resnet = train_resnet
        for _, param in self.resnet.named_parameters():
            param.requires_grad = train_resnet

    def forward(self, x):
        batch_size = x.shape[0]
        bag_size = x.shape[1]
        # We process all the examples in all the samples at the same time
        x = x.reshape(bag_size * batch_size, *x.shape[2:])
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = x.view(batch_size, bag_size, -1)
        return x
