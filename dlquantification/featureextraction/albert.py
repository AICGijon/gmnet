import torch
from transformers import AlbertModel

class AlbertFeatureExtractionModule(torch.nn.Module):
    """This module is the feature extraction part of the network. Could be anything, a CNN, LSTM, depending on the application"""
    def __init__(self,train_bert:bool=False):
        super(AlbertFeatureExtractionModule, self).__init__()
        self.model = AlbertModel.from_pretrained("albert-base-v2")
        self.set_train_bert(train_bert)
        self.output_size = 768

    def set_train_bert(self,train_bert):
        self.train_bert = train_bert
        for _, param in self.model.named_parameters():                
            param.requires_grad = train_bert

    def forward(self,input):
        bert_output = self.model(**input)
        features = bert_output[0][:,0,:]
        return features