import torch
import torch.nn

class MeanLayer(torch.nn.Module):
    def __init__(self):
        super(MeanLayer, self).__init__()

    def forward(self, x):
       return torch.mean(x, dim = 1)
