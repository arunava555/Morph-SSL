#!/usr/bin/env python
# coding: utf-8


import torch.nn as nn
import torch


# In[2]:


class S3DConv(nn.Module):
    def __init__(self, in_chnls, out_chnls, stride=1,p=0.25, fltr_sz=7):
        super(S3DConv, self).__init__()
        
        ###########################################################################
        # The ordering of the spatial dimensions is Height X Width X Depth 
        # Option 1: use 1X3X3,  3X1X3   and   3X3X1 convolutions (fltr_sz=7)
        # Option 2: use 1X3X3,  7X1X3   and   7X3X1 convolutions (fltr_sz=3)
        ###########################################################################
        pout1=int(round(p*out_chnls))
        pout2=int(round(p*out_chnls))
        pout3=out_chnls-(pout1+pout2)
                
        self.conv_1by3by3=nn.Sequential(
                                nn.GroupNorm(1, in_chnls),
                                nn.ELU(),
                                nn.Conv3d(in_chnls, pout1, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,1,1),
                                          groups=1, bias=False, padding_mode='replicate'))
        
        self.conv_3by1by3=nn.Sequential(
                                nn.GroupNorm(1, in_chnls),
                                nn.ELU(),
                                nn.Conv3d(in_chnls, pout2, kernel_size=(fltr_sz,1,3), stride=(stride,1,stride), padding=(fltr_sz//2,0,1),
                                          groups=1, bias=False, padding_mode='replicate'))
        
        self.conv_3by3by1=nn.Sequential(
                                nn.GroupNorm(1, in_chnls),
                                nn.ELU(),
                                nn.Conv3d(in_chnls, pout3, kernel_size=(fltr_sz,3,1), stride=(stride,stride,1), padding=(fltr_sz//2,1,0),
                                          groups=1, bias=False, padding_mode='replicate'))
        
    def forward(self, x):
        
        x1=self.conv_1by3by3(x)
        x2=self.conv_3by1by3(x)
        x3=self.conv_3by3by1(x)
        
        out=torch.cat((x1, x2, x3),dim=1)
        return out

