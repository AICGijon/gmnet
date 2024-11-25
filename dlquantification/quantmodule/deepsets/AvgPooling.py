import torch


class AvgPooling(torch.nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, input):
        return torch.mean(input, dim=1)
