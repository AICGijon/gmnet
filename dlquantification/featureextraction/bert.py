import torch
from transformers import BertModel


class BertFeatureExtractionModule(torch.nn.Module):
    """This module is the feature extraction part of the network.
    Could be anything, a CNN, LSTM, depending on the application"""

    def __init__(
        self,
        train_bert: bool = False,
        from_pretrained="bert-base-uncased",
        linear_size=None,
        dropout=0,
        pooler_output=False,
    ):
        super(BertFeatureExtractionModule, self).__init__()
        self.model = BertModel.from_pretrained(from_pretrained)
        self.model.gradient_checkpointing_enable()
        self.set_train_bert(train_bert)
        self.linear_size = linear_size
        self.pooler_output = pooler_output
        if linear_size is not None:
            self.out = torch.nn.Linear(768, linear_size)
            self.dropout = torch.nn.Dropout(p=dropout)
            self.output_size = linear_size
        else:
            self.output_size = 768

    def set_train_bert(self, train_bert):
        self.train_bert = train_bert
        for _, param in self.model.named_parameters():
            param.requires_grad = train_bert

    def forward(self, input):
        if self.pooler_output:
            features = self.model(**input, output_hidden_states=False, output_attentions=False)["pooler_output"]
        else:
            bert_output = self.model(**input)
            features = bert_output[0][:, 0, :]
        if self.linear_size is not None:
            return self.out(self.dropout(features))
        else:
            return features
