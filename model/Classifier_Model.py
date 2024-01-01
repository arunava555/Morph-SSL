#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch.nn as nn
import torch


# In[3]:


def compute_entropy(x):
    ### Compute spatial Entropy of the "Saliency Map" M 
    B,C,H,W,D=x.shape
    x=x.view(B,-1) # B,d   where d=H*W*D (C=1)
    
    epsilon=10**(-9)
    p=x/(torch.sum(x, dim=1,keepdim=True)+epsilon)   #B,d
    logp=torch.log(p+epsilon) # B,d
    
    h = p*logp # B,d
    h = -1.0 * torch.sum(h, dim=1,keepdim=True) # B,1
    return h
    
    

    
class my_scalar_model(nn.Module):
    def __init__(self):
        super(my_scalar_model, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0) # exp of 0 is 1

    def forward(self, x):
        return (x*self.weight.exp())
    
    
    

##################################################################################################
class Classification_Network(nn.Module):
    def __init__(self):
        super(Classification_Network, self).__init__()
        
        self.conv1=nn.Sequential(
                            nn.Conv3d(128, 1024, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0),
                                  groups=1, bias=False),
                            nn.BatchNorm3d(1024),
                            nn.ELU()
                            )
        
        self.conv2=nn.Sequential(
                            nn.Conv3d(1024, 1024, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0),
                                  groups=1, bias=False),
                            nn.BatchNorm3d(1024),
                            nn.ELU()
                            )
        
        self.conv3=nn.Sequential(
                            nn.Conv3d(1024, 1, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0),
                                  groups=1, bias=False),
                            nn.BatchNorm3d(1),
                            nn.Softplus()
                            )
        
        self.scalar_wt=my_scalar_model()#nn.Linear(1, 1, bias=True); sometimes replacing it with Linear can improve performance
        self.scalar_wt2=my_scalar_model()
        
       
        
           
        
    def forward(self, x):
        x=self.conv1(x) # change C from 128 to 1024
        x=self.conv2(x)
        x=self.conv3(x) # 1 channel
        
        # GAP
        b=torch.mean(x.view(x.shape[0],x.shape[1],-1), dim=2)# GAP B,1024
        b=self.scalar_wt2(b)
        b=1/b # inverse: larger t_cnv=> smaller b=> less saliency map activation
        
        a=compute_entropy(x)
        a=torch.sigmoid(self.scalar_wt(a)) 
        return a,b,x
        
#################################################################################################


# In[ ]:




