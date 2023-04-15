import torch 
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils import parameters_to_vector
class HingeLoss(nn.Module):
    def __init__(self,model):
        super(HingeLoss, self).__init__()
        self.model=model
    
    def forward(self, output , y,c=0.5):
        loss=torch.mean(torch.clamp(1-y*output, min=0))
        weight=parameters_to_vector(self.model.parameters()).squeeze()
        loss+=c*(weight.t()@weight)
        return loss

        

        



        