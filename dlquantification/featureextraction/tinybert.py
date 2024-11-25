import torch
from transformers import AutoModel, AutoModelForSequenceClassification


class TinyBertFeatureExtractionModule(torch.nn.Module):
    """This module is the feature extraction part of the network.
    Could be anything, a CNN, LSTM, depending on the application"""

    def __init__(
        self,
        train_bert: bool = False,
        from_pretrained="google/bert_uncased_L-2_H-128_A-2",
        classifier_layer: bool = False,
        n_labels: int = None,
        linear_size=None,
        dropout=0,
    ):
        super(TinyBertFeatureExtractionModule, self).__init__()
        self.classifier_layer = classifier_layer

        if classifier_layer:
            self.model = AutoModelForSequenceClassification.from_pretrained(from_pretrained)
        else:
            self.model = AutoModel.from_pretrained(from_pretrained)

        # self.model.gradient_checkpointing_enable()
        self.set_train_bert(train_bert)
        self.linear_size = linear_size
        if linear_size is not None:
            if classifier_layer:
                self.out = torch.nn.Linear(n_labels, linear_size)
            else:
                self.out = torch.nn.Linear(128, linear_size)
            self.dropout = torch.nn.Dropout(p=dropout)
            self.output_size = linear_size
        else:
            if classifier_layer:
                self.output_size = n_labels
            else:
                self.output_size = 128

    def set_train_bert(self, train_bert):
        self.train_bert = train_bert
        for _, param in self.model.named_parameters():
            param.requires_grad = train_bert

    def cat_batches(input):
        """Bert only accepts tensors in the form batch_size x seq_lenght. We need to join all the examples of all the samples."""

    def forward(self, input):
        batch_size = input[next(iter(input))].shape[0]
        bag_size = input[next(iter(input))].shape[1]
        for key in input.keys():
            input[key] = input[key].view(batch_size * bag_size, -1)
        bert_output = self.model(**input)
        if self.classifier_layer:
            features = bert_output.logits
        else:
            features = bert_output[0][:, 0, :]

        features = features.view(batch_size, bag_size, -1)

        if self.linear_size is not None:
            return self.out(self.dropout(features))
        else:
            return features
