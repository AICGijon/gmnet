import torch

class BertTextDataset(torch.utils.data.Dataset):
    def __init__(self,inputs_ids=None,attention_masks=None, targets=None):
        self.inputs_ids = inputs_ids
        self.attention_masks = attention_masks
        self.targets = targets

    def __getitem__(self, idx):
        item = {'input_ids': self.inputs_ids[idx],'attention_mask':self.attention_masks[idx]}
        if self.targets is None:
            return (item,)
        else:
            return item,self.targets[idx]

    def __len__(self):
        return len(self.inputs_ids)

    def load(self,file,device=torch.device('cpu')):
        data = torch.load(file,map_location=device)
        self.inputs_ids = data['inputs_ids']
        self.attention_masks = data['attention_masks']
        self.targets = data['targets']
    
    def save(self,file):
        torch.save({"inputs_ids": self.inputs_ids, "attention_masks":self.attention_masks, "targets":self.targets}, file)