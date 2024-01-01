#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch.nn as nn
from S3DConv_model import *


# In[13]:


############################### Basic Encoder Block ############################
class Basic_Encoder_Block(nn.Module):
    def __init__(self, in_chnls, stride=1):
        super(Basic_Encoder_Block, self).__init__()
        
        ########### BASIC ENCODER BLOCK #############
        self.conv1=S3DConv(in_chnls, in_chnls, stride)
        self.conv2=S3DConv(in_chnls, in_chnls, stride)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        out=torch.cat((x,x1), dim=1) # concatenation based skip connection
        return out


# In[10]:


################################## Downsampling Block ###########################
class Downsampling_Block(nn.Module):
    def __init__(self, in_chnls, strd_slc):
        super(Downsampling_Block, self).__init__()
        
        ########### DOWNSAMPLING BLOCK #############
        self.conv=nn.Sequential(nn.GroupNorm(1, in_chnls),
                                nn.ELU(),
                                nn.Conv3d(in_chnls, in_chnls, kernel_size=(3,3,3), stride=(2,2,strd_slc), 
                                            padding=(1,1,1),groups=in_chnls, bias=False))
        
    def forward(self, x):       
        x = self.conv(x)
        return x


# In[11]:


############## Encoder Block: Combine the Basic Block with Downsampling ##########
class Encoder_Block(nn.Module):
    def __init__(self, in_chnls, strd_slc):
        super(Encoder_Block, self).__init__()

        # while the Downsampling always occurs across the H, W dimensions, it may or maynot occur across slices
        # this is controlled my assigning strd_slc=1 (no downsampling) or 2.
        self.dwn_smpl=Downsampling_Block(in_chnls, strd_slc)
        self.conv_block=Basic_Encoder_Block(in_chnls) 
        
    def forward(self, x):
        x=self.dwn_smpl(x)            
        x=self.conv_block(x)
        return x


# In[12]:


########################### The Complete Encoder Architecture ###############
class Encoder_Architecture(nn.Module):
    def __init__(self, base_chnls, out_ftr_dim):
        super(Encoder_Architecture, self).__init__()
        
        # This first layer helps with the pre-activation thing as the next block starts with normalization.
        self.first_layer=nn.Sequential(nn.Conv3d(1, base_chnls, kernel_size=(1,1,1), stride=(1,1,1), 
                                       padding=(0,0,0), groups=1, bias=False)) # B,32,192,192,32
        s1=base_chnls    # 16
        s2=base_chnls*2  # 32
        s3=base_chnls*4  # 64
        s4=base_chnls*8  # 128
        s5=base_chnls*16 # 256
        
        ###################
        self.enc1=Basic_Encoder_Block(s1)    # Bsz,16,192,192,32 ----> Bsz,32,192,192,32
        self.enc2=Encoder_Block(s2,1)         # Bsz,32,192,192,32 ----> Bsz,64,96, 96,32
        self.enc3=Encoder_Block(s3,1)         # Bsz,64,96, 96,32 ----> Bsz,128,48, 48,32
        self.enc4=Encoder_Block(s4,1)         # Bsz,128,48, 48,32 ----> Bsz,256,24, 24,32
        self.enc5=Encoder_Block(s5,2)         # Bsz,256,24, 24,32 ----> Bsz,512,12, 12,16 
        s5=s5*2
        ###################
        self.out_deform_lyr=nn.Sequential(
                                nn.GroupNorm(1, s5),
                                nn.ELU(),
                                nn.Conv3d(s5, s5, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0),
                                          groups=1, bias=True),
        
                                nn.GroupNorm(1, s5),
                                nn.ELU(),
                                nn.Conv3d(s5, out_ftr_dim//2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0),
                                          groups=1, bias=True)
        
                                )
        
        self.out_intens_lyr=nn.Sequential(
                                nn.GroupNorm(1, s5),
                                nn.ELU(),
                                nn.Conv3d(s5, s5, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0),
                                          groups=1, bias=True),
                                
                                nn.GroupNorm(1, s5),
                                nn.ELU(),
                                nn.Conv3d(s5, out_ftr_dim//2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0),
                                          groups=1, bias=True)
        
                                )
        
    
    def forward(self, x):
        
        x=self.first_layer(x)
        
        ############ Encoder Blocks #############
        x = self.enc1(x)        
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        
        #########################################
        ftr_deform=self.out_deform_lyr(x)
        ftr_intens=self.out_intens_lyr(x)
        ftr=torch.cat((ftr_deform, ftr_intens), dim=1)
        
        return ftr

