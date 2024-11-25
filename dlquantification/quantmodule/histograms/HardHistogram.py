# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class HardHistogram(nn.Module):

    """This class computes a histogram based in convolutional layers. I have followed the paper.
    Yusuf, Ibrahim, George Igwegbe, and Oluwafemi Azeez. "Differentiable Histogram with Hard-Binning."
    arXiv preprint arXiv:2012.06311 (2020).
    """

    def __init__(self, n_features, num_bins, quantiles=False):

        # inherit nn.module
        super(HardHistogram, self).__init__()

        # The number of features will be the number of channels to use in the convolution
        self.in_channels = n_features
        self.num_bins = num_bins
        self.quantiles = quantiles

        # in_channels: number of features that we have
        # out_channels: number of bins*number of features
        # kernel_size: 1
        # groups: we group by channel, each input channel is convolved with its own set of filters
        # in_channels: correspond with the number of input features
        # out_channels: n_features*n_bins
        # bias terms: we will have as many bias terms as the out channels. This is ok because these are the bin centers
        self.bin_centers_conv = nn.Conv1d(
            self.in_channels, self.num_bins * self.in_channels, kernel_size=1, groups=self.in_channels, bias=True
        )
        # All weights to one and we do not want to learn these.
        # We are modeling the centering operation, we only work with bias
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False

        self.bin_widths_conv = nn.Conv1d(
            self.num_bins * self.in_channels,
            self.num_bins * self.in_channels,
            kernel_size=1,
            groups=self.num_bins * self.in_channels,
            bias=True,
        )
        # In this case, bias fixed with ones and we do not learn it
        self.bin_widths_conv.weight.data.fill_(-1)
        self.bin_widths_conv.weight.requires_grad = False

        # Threshold layer
        self.threshold = torch.nn.Threshold(1, 0)

        # self.hist_pool = nn.AvgPool1d(n_examples, stride=1, padding=0)  # average by row
        self.centers = self.bin_centers_conv.bias
        self.widths = self.bin_widths_conv.weight

        # Compute initial bin centers and bin sizes, this can change during backpropagation
        bin_centers = -1 / self.num_bins * (torch.arange(self.num_bins).float() + 0.5)
        self.bin_centers_conv.bias = torch.nn.Parameter(torch.cat(self.in_channels * [bin_centers]), requires_grad=True)
        bin_width = (1 / (2 * self.num_bins)) + 0.001
        self.bin_widths_conv.bias.data.fill_(bin_width)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        result = torch.empty((input.shape[0], self.num_bins * self.in_channels), device=input.device)
        # Histograms do not work with batches of samples, process sample by sample
        for i, sample in enumerate(input):
            # Pass through first convolution to learn bin centers: |x-center|
            sample = self.bin_centers_conv(sample.transpose(0, 1).unsqueeze(0))

            # Absolute value
            sample = torch.abs(sample)

            # Second convolution
            sample = self.bin_widths_conv(sample)

            # Exponentiation
            sample = torch.pow(1.01, sample)

            # Thresholding
            sample = self.threshold(sample)

            # Gloval average pooling
            # input = self.hist_pool(input)
            sample = torch.mean(sample, dim=2)

            if self.quantiles:
                sample = sample.view(-1, self.num_bins).cumsum(dim=1)

            # The output is a Tensor with size n_bins*n_features
            result[i, :] = sample.flatten()
        return result
