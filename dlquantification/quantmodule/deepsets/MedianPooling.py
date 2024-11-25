import torch


class MedianPooling(torch.nn.Module):
    def __init__(self):
        super(MedianPooling, self).__init__()

    def forward(self, input):
        return torch.median(input, dim=1)[0]
