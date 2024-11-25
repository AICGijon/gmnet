import torch


class MaxPooling(torch.nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, input):
        return torch.max(input, dim=1)[0]
