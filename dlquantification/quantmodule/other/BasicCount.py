import torch


class BasicCount(torch.nn.Module):
    """
    This performs a basic count suposing that we have a classifier in the feature extraction layer.
    """

    def __init__(self, n_classes):
        super(BasicCount, self).__init__()
        self.n_classes = n_classes
        self.num_bins = 1

    def forward(self, input):
        if self.n_classes != input.shape[2]:
            raise ValueError("Basic count expects a classifier in the feature extraction layer.")
        batch_size = input.shape[0]
        n_examples = input.shape[1]
        max = torch.argmax(input, dim=2)
        freqs = torch.zeros((batch_size, self.n_classes), device=input.device)
        for i, sample in enumerate(max):
            counts = torch.bincount(sample, minlength=self.n_classes)
            freqs[i, :] = counts / n_examples
        return freqs
