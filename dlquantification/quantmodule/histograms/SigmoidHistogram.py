import torch


class SigmoidHistogram(torch.nn.Module):
    """
    This class allows to build a histogram that is differentiable. This histogram is an aproximation, not the real one
    Following: https://discuss.pytorch.org/t/differentiable-torch-histc/25865
    """

    def __init__(self, num_bins, min, max, sigma, quantiles=False):
        super(SigmoidHistogram, self).__init__()
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
        for i, sample in enumerate(input):
            sample = torch.unsqueeze(sample.transpose(0, 1), 1) - torch.unsqueeze(
                self.centers, 1
            )  # compute the distance to the center
            sample = torch.sigmoid(self.sigma * (sample + self.delta / 2)) - torch.sigmoid(
                self.sigma * (sample - self.delta / 2)
            )
            sample = sample.flatten(end_dim=1).sum(dim=1) / n_examples_bag

            if self.quantiles:
                sample = input.view(-1, self.num_bins).cumsum(dim=1)

            result[i, :] = sample.flatten()
        return result
