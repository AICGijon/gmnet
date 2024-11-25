import torch
from transformers import DistilBertModel

class DistilBertFeatureExtractionModule(torch.nn.Module):
    """This module is the feature extraction part of the network. Could be anything, a CNN, LSTM, depending on the application"""
    def __init__(self,output_size:int, dropout:float,train_bert:bool=False):
        super(DistilBertFeatureExtractionModule, self).__init__()
        self.output_size = output_size

        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.model.gradient_checkpointing_enable()
        self.dropout = torch.nn.Dropout(dropout)
        #self.output_layer = torch.nn.Linear(768, output_size)

        self.set_train_bert(train_bert)

    def set_train_bert(self,train_bert):
        self.train_bert = train_bert
        for _, param in self.model.named_parameters():                
            param.requires_grad = train_bert

    def forward(self,input):
        bert_output = self.model(**input)
        features = bert_output[0][:,0,:]
        #return self.output_layer(self.dropout(features))
        return features