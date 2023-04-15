import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.init as init
class LinearModel(nn.Module):
    def __init__(self, input_shape, zero_init=True):
        super(LinearModel,self).__init__()
        # if zero_init:
        #     w_init=torch.zeros((1,))
            
       
        self.linear=nn.Linear(input_shape,1,bias=True)
        # self.linear.weight.data.copy_(w_init)
        init.zeros_(self.linear.weight)
        init.zeros_(self.linear.bias)
    def forward(self,x):
        # x=x.view(x.shape[0],-1)
        x=self.linear(x)
        return x