import torch


class NoFeatureExtractionModule(torch.nn.Module):
    """This module is the feature extraction part of the network."""

    def __init__(self, input_size: int):
        """No feature extraction.

        Input will be returned without change

        Args:
            input_size (int): output size will be equal to input size
        """
        super(NoFeatureExtractionModule, self).__init__()
        self.output_size = input_size

    def forward(self, input):
        """Forward pass.

        Args:
            input (Tensor): input data

        Returns:
            Tensor: same as input
        """
        return input
