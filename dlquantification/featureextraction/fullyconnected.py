import torch


class FCFeatureExtractionModule(torch.nn.Module):
    """This module is the feature extraction part of the network.
    Could be anything, a CNN, LSTM, depending on the application"""

    def __init__(self, input_size, output_size, hidden_sizes, dropout=0, activation="leakyrelu", flatten=False):
        super(FCFeatureExtractionModule, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        self.layers = torch.nn.Sequential()

        prev_size = input_size
        if flatten:
            self.layers.add_module("fe_flatten", torch.nn.Flatten(start_dim=2))
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.add_module("fe_linear_%d" % i, torch.nn.Linear(prev_size, hidden_size))
            if activation == "leakyrelu":
                self.layers.add_module("fe_leaky_relu_%d" % i, torch.nn.LeakyReLU())
            elif activation == "relu":
                self.layers.add_module("fe_relu_%d" % i, torch.nn.ReLU())
            self.layers.add_module("fe_dropout_%d", torch.nn.Dropout(dropout))
            prev_size = hidden_size

        self.layers.add_module("fe_lastlinear", torch.nn.Linear(prev_size, output_size))

    def forward(self, input):
        return self.layers(input)
