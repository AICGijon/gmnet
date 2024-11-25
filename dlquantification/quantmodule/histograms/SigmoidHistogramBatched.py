import torch


class SigmoidHistogramBatched(torch.nn.Module):
    """
    This class allows to build a histogram that is differentiable. This histogram is an aproximation, not the real one
    Following: https://discuss.pytorch.org/t/differentiable-torch-histc/25865
    """

    def __init__(self, num_bins, min, max, sigma, quantiles=False):
        super(SigmoidHistogramBatched, self).__init__()
        self.num_bins = num_bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.quantiles = quantiles
        self.delta = float(max - min) / float(num_bins)  # Compute the width of each bin
        self.register_buffer(
            "centers", float(min) + self.delta * (torch.arange(num_bins).float() + 0.5)
        )  # Compute the center of each bin
        # TODO: check this. In theory histograms are more exact but it works worse for lequa
        # self.delta = self.delta + 0.001

    def forward(self, input):
        """Function that computes the histogram. It is prepared to compute multiple histograms at the same time,
        saving a lot of time.
        Args:
            x Tensor. Two dimension tensor. Each row should be all possible values for a single feature (as many values
            as columns).
        Returns
            A vector with size n_features*n_bins, containing all the histograms
        """
        if len(input.shape) == 2:
            input = input.unsqueeze(0)

        result = torch.empty((input.shape[0], self.num_bins * input.shape[2]), device=input.device)
        n_examples_bag = input.shape[1]
        tens = torch.unsqueeze(input.mT, 2) - torch.unsqueeze(
            self.centers, 1
        )  # Compute distance to the center (mT transpose just the last 2 dim of the tensor)
        tens = torch.sigmoid(self.sigma * (tens + self.delta / 2)) - torch.sigmoid(
            self.sigma * (tens - self.delta / 2)
        )  # Compute the 2 sigmoids
        tens = tens.mT.sum(dim=2) / n_examples_bag
        result = tens.flatten(start_dim=1)
        return result
