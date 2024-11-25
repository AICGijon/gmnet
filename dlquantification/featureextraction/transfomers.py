import torch
from dlquantification.quantmodule.transformers.modules import ISAB


class ISABExtractionModule(torch.nn.Module):
    """This module is the feature extraction part of the network."""

    def __init__(self, input_size: int, dim_hidden: int, num_heads: int, num_inds: int, ln: bool, rFF=None):

        super(ISABExtractionModule, self).__init__()
        self.fe = torch.nn.Sequential(
            ISAB(input_size, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        if rFF is not None:
            self.fe.add_module("linear_fe", rFF)
            self.output_size = rFF.output_size
        else:
            self.output_size = dim_hidden

    def forward(self, input):
        return self.fe(input)
