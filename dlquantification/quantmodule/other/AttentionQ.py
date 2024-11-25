import torch.nn as nn
import torch
from dlquantification.quantmodule.histograms.HardHistogram import HardHistogram


class AttentionQ(nn.Module):
    def __init__(self, dim_in, num_inds, dim_hidden, n_bins):
        super(AttentionQ, self).__init__()
        # self.I = nn.Parameter(torch.Tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]).unsqueeze(0))
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_in))
        # nn.init.xavier_uniform_(self.I)
        nn.init.zeros_(self.I)
        # The idea is to get a matrix with size num_inds x bag_size in which each cell is the distance
        # between the ind_point i and the example j. Then we build a histogram with the distances for each of the
        # ind_points
        self.output_size = num_inds * n_bins
        self.histogram = HardHistogram(n_features=num_inds, num_bins=n_bins, quantiles=False)
        self.sigmoid = torch.nn.Sigmoid()
        # self.fc1 = nn.Linear(dim_in, dim_hidden)
        # self.fc2 = nn.Linear(dim_in, dim_hidden)

    def forward(self, X):
        # X dimension -> (batch_size, bag_size, dim_in)
        # I dimension -> (batch_size, num_index, dim_in)
        # First idea, build histograms over the distances
        return self.histogram(self.sigmoid(X.bmm(self.I.repeat(X.size(0), 1, 1).transpose(1, 2))))
        # Second idea, forget about histograms and pass directly the attentions to next layer
        # return X.bmm(self.I.repeat(X.size(0), 1, 1)).transpose(1, 2).flatten(1)
