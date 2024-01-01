#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
from S3DConv_model import *
import torch


# In[ ]:


############ DECODER BUILDING BLOCK #############
class Decoder_Block(nn.Module):
    def __init__(self, in_chnls, strd_slc):
        super(Decoder_Block, self).__init__()
        ############ DECODER BUILDING BLOCK #############
        self.up_smpl=nn.Sequential(nn.GroupNorm(1, in_chnls),
                                nn.ELU(),
                                nn.Upsample(scale_factor=(2,2,strd_slc), mode='trilinear'),
                                nn.Conv3d(in_chnls, in_chnls//4, kernel_size=(3,3,3), stride=(1,1,1), 
                                          padding=(1,1,1),groups=in_chnls//4, bias=False))
        
        self.conv_block=S3DConv(in_chnls//4, in_chnls//4)
            
    def forward(self, x):
        x=self.up_smpl(x)     # Upsample the H,W,D and halve channels      
        x1=self.conv_block(x)
        out=torch.cat((x,x1), dim=1)       
        return out


# In[ ]:


##################### The Complete Encoder Architecture ################
class Decoder_Model(nn.Module):
    def __init__(self, in_dim,first_dim, out_chnls):
        super(Decoder_Model, self).__init__()
        ###### The Complete Encoder Architecture ######
        self.first_layer=nn.Sequential(nn.Conv3d(in_dim, first_dim, kernel_size=(1,1,1), stride=(1,1,1), 
                                       padding=(0,0,0), groups=1, bias=False))
        s1=first_dim//2 # 256
        s2=s1//2 # 128
        s3=s2//2 # 64
        
        ###################
        self.dec1=Decoder_Block(first_dim,2)   # Bsz,512,12,12,16 ----> Bsz,256,24,24,32
        self.dec2=Decoder_Block(s1,1)          # Bsz,256,24,24,32 ----> Bsz, 128,48,48,32
        self.dec3=Decoder_Block(s2,1)          # Bsz,128,48,48,32 ----> Bsz,  64,96,96,32
        self.dec4=Decoder_Block(s3,1)          # Bsz, 64,96,96,32 ----> Bsz,  32,192,192,32 
        ###################
        s3=s3//2
        
        self.out_lyr=nn.Sequential(
                                nn.GroupNorm(1, s3),
                                nn.ELU(),
                                nn.Conv3d(s3, out_chnls, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0),
                                          groups=1, bias=True))
        
    
    def forward(self, x):
        x=self.first_layer(x) 
        x = self.dec1(x)        
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        out=self.out_lyr(x)
        return out


# In[ ]:


class my_scalar_model(nn.Module):
    def __init__(self):
        super(my_scalar_model, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1.0) # exp of 0 is 1

    def forward(self, x):
        return (self.weight*x)


# In[ ]:


'''
class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        #nn.init.xavier_uniform_(self.log_weight)
        nn.init.constant_(self.log_weight, 0.0) # exp of 0 is 1

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())
'''


# In[ ]:


#######################  2 arms, mag scaling ############
class Decoder_Architecture(nn.Module):
    def __init__(self, in_dim, first_dim):
        super(Decoder_Architecture, self).__init__()
        
        # ftr_dims: total dimensionality of the encoder feature (including both additive & deform)
        self.deform_decoder=Decoder_Model(in_dim,first_dim, 3) # 
        self.additive_decoder=Decoder_Model(in_dim,first_dim, 1)
        
        self.scl=my_scalar_model()
        
        self.scl_dfrm1=nn.Linear(1,1, bias=False)#PositiveLinear(1,1)
        self.scl_add1=nn.Linear(1,1, bias=False)#PositiveLinear(1,1)
        
        self.scl_dfrm2=nn.Linear(1,1, bias=False)#PositiveLinear(1,1)
        self.scl_add2=nn.Linear(1,1, bias=False)#PositiveLinear(1,1)
        
        
    def forward(self, ftr_A, ftr_B):
        
        # Scaling the difference as similar values in ftr_A and ftr_B may tend to cancel each other out
        df=self.scl((ftr_B-ftr_A))
        
        # Since 2 arms, divide df into 2 halves
        dir_dfrm=df[:,0:df.shape[1]//2]
        dir_add=df[:,df.shape[1]//2: df.shape[1]]
        
        ######## convert to unit norm  ###############
        mag_dfrm=torch.norm(dir_dfrm.view(dir_dfrm.shape[0],-1), keepdim=True, dim=1)
        mag_add=torch.norm(dir_add.view(dir_add.shape[0],-1), keepdim=True, dim=1)
        
        # Scale the magnitude by a constant value
        mag_dfrm1=self.scl_dfrm1(mag_dfrm)
        mag_add1=self.scl_add1(mag_add)
        
        mag_dfrm2=self.scl_dfrm2(mag_dfrm)
        mag_add2=self.scl_add2(mag_add)
        
        ############ compute unit norm directions ###################  
        dir_dfrm=dir_dfrm/mag_dfrm2
        dir_add=dir_add/mag_add2
        
        ####### Compute the deformation #####
        deform=self.deform_decoder(dir_dfrm)
        additive=self.additive_decoder(dir_add)
        
        ##### Scaling the output of decoders by magnitude of the feature
        mag_out_dfrm=torch.norm(deform.reshape(deform.shape[0],-1), keepdim=True,dim=1)
        scl_mag_dfrm=mag_dfrm1/mag_out_dfrm
        
        mag_out_add=torch.norm(additive.reshape(additive.shape[0],-1), keepdim=True,dim=1)
        scl_add_dfrm=mag_add1/mag_out_add
        
        deform=deform*scl_mag_dfrm        
        additive=additive*scl_add_dfrm
        
        return deform, additive#, deform_inv, additive_inv

