#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
from S3DConv_model import *
from Encoder_model import Encoder_Block, Basic_Encoder_Block


# In[ ]:


####################### The Comparator network (A copy of first 3 layers of the Encoder) ############
'''
The Comparator Network is a copy of the first few layers of the Encoder architecture.
It is used to compute perceptual loss (MSE in a feature space rather than using the voxel intensities directly)
Weights of the Comparator Network are updated with an exponential moving average of the Encoder weights
(and not through backprop to prevent feature collapse)
'''

class Comparator_Architecture(nn.Module):
    def __init__(self, base_chnls, out_ftr_dim):
        super(Comparator_Architecture, self).__init__()
        # This first layer helps with the pre-activation thing as the next block starts with normalization.
        self.first_layer=nn.Sequential(nn.Conv3d(1, base_chnls, kernel_size=(1,1,1), stride=(1,1,1), 
                                       padding=(0,0,0), groups=1, bias=False)) # B,32,192,192,32
        s1=base_chnls    # 16
        s2=base_chnls*2  # 32
        
        ###################
        self.enc1=Basic_Encoder_Block(s1)    # Bsz,16,192,192,32 ----> Bsz,32,192,192,32
        self.enc2=Encoder_Block(s2,1)         # Bsz,32,192,192,32 ----> Bsz,64,96, 96,32
    
    def forward(self, x):
        x=self.first_layer(x)
        
        x1 = self.enc1(x)        
        x2 = self.enc2(x1)
        # return three feature spaces x,x1,x2 to average of the MSE in these three feature spaces.
        return (x,x1,x2)

