#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random 
from skimage import filters
import torch
import torch.nn.functional as F



class train_dataset(Dataset):
    def __init__(self, img_pth, img_pairs):
        '''
        img_pth: Directory containing preprocessed scans. Each file has the name: eye_id__visitdate.npz.
        The npz file contains: 
        a) img:preprocessed OCT vol of size Height(192+margin of 32), Width(192+margin of 32), Depth(32+margin of 5).
               img is H X W X D
        b) roi_mask: binary ROI mask of the region in img containing the retinal tissue
        c) ll, ul: axial(Height) bounds (lower and upper bound) of the ROI mask 
        
        img_pairs: an npz file which contains the pre-computed random pair of Scans from different visits 
        of the same eye from the training dataset. It contains a dictionary containing
        'trn_eyes': the eye_id,  'trn_visitA': the visit-date of the first visit
        'trn_visitB': the visit-date of the second visit
        
        '''
        
        self.pth=img_pth
        
        a=np.load(img_pairs)
        self.eye_id=a['trn_eyes']
        self.visitA=a['trn_visitA']
        self.visitB=a['trn_visitB']
        del a
        
        # params for random noise for Data Aug
        self.std=0.001
        self.mean=0
        
        
    def read_image(self, eye_id, visit):
        a=np.load(self.pth+eye_id+'__'+visit+'.npz')
        I=a['img']
        ll=a['ll']
        ul=a['ul']
        roi_mask=a['roi_mask']
        del a
        
        ll=int(ll)
        ul=int(ul)
        return I, ll,ul, roi_mask
    

    
    def process_image(self, I, msk, mn_slc, mn_col, mn_ht):
        
        I=I[mn_ht:mn_ht+192, mn_col:mn_col+192,mn_slc:mn_slc+32] 
        msk=msk[mn_ht:mn_ht+192, mn_col:mn_col+192,mn_slc:mn_slc+32] 
        
        I=I.astype(np.float)
        msk=msk.astype(np.float)
        ####I=I/256
        if np.max(I)>300:
            I=I/65535  # then most probably it is uint16
        else:
            I=I/255 # uint8
        #######
        I=(I*2)-1
        
        # random blurring of image for data aug
        r=random.random()*0.90 # between [0,1]
        I = filters.gaussian(I, sigma=(r, r,r), truncate=3)
        
        # Add channel dimension
        I=np.expand_dims(I, axis=0) # 1,H,W,D
        I=torch.FloatTensor(I)
        
        msk=np.expand_dims(msk, axis=0)
        msk=torch.FloatTensor(msk)
        
        # Add random noise for data aug
        I=I + torch.randn(I.size()) * self.std + self.mean
        return I, msk
        
        
        
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""        
        # Read images
        I_A, ll_A,ul_A, IA_msk=self.read_image(self.eye_id[index], self.visitA[index])
        I_B, ll_B,ul_B, IB_msk=self.read_image(self.eye_id[index], self.visitB[index])
        
        ## Determine the crop upper limit based on ll_A, ll_B, ul_A, ul_B
        mn_ht=random.randint(0,32)
        mn_col=random.randint(0,32)
        mn_slc=random.randint(0,5)
        
        ll=min(ll_A, ll_B)
        if ((mn_ht>ll) and (ll>0)):
            mn_ht=ll
        
        I_A, IA_msk=self.process_image(I_A, IA_msk, mn_slc, mn_col, mn_ht)
        I_B, IB_msk=self.process_image(I_B, IB_msk, mn_slc, mn_col, mn_ht)
        
        nm_A=self.eye_id[index]+'_'+self.visitA[index]
        nm_B=self.eye_id[index]+'_'+self.visitB[index]
        
        # Random Flipping
        if random.random()<=0.5:
            I_A=torch.flip(I_A,dims=[2])
            I_B=torch.flip(I_B,dims=[2])
            IA_msk=torch.flip(IA_msk,dims=[2])
            IB_msk=torch.flip(IB_msk,dims=[2])
        
        
        sample={'I_A': I_A, 'I_B': I_B, 'IA_msk': IA_msk, 'IB_msk': IB_msk, 'nm_A': nm_A, 'nm_B': nm_B}
        return sample 

    def __len__(self):
        return self.eye_id.shape[0]


# In[ ]:


class val_dataset(Dataset):
    def __init__(self, img_pth, img_pairs):
        
        '''
        img_pth: Directory containing preprocessed scans. Each file has the name: eye_id__visitdate.npz.
        The npz file contains: 
        a) img:preprocessed OCT vol of size Height(192+margin of 32), Width(192+margin of 32), Depth(32+margin of 5).
               img is H X W X D
        b) roi_mask: binary ROI mask of the region in img containing the retinal tissue
        c) ll, ul: axial(Height) bounds (lower and upper bound) of the ROI mask 
        
        img_pairs: an npz file which contains the pre-computed random pair of Scans from different visits 
        of the same eye from the validation dataset. It contains a dictionary containing
        'val_eyes': the eye_id,  'val_visitA': the visit-date of the first visit
        'val_visitB': the visit-date of the second visit
        
        '''
        
        self.pth=img_pth
        
        a=np.load(img_pairs)
        self.eye_id=a['val_eyes']
        self.visitA=a['val_visitA']
        self.visitB=a['val_visitB']
        self.lbl=a['val_lbl']
        del a
        
        
    def read_image(self, eye_id, visit):
        a=np.load(self.pth+eye_id+'__'+visit+'.npz')
        I=a['img']
        ll=a['ll']
        ul=a['ul']
        roi_mask=a['roi_mask']
        del a
        
        ll=int(ll)
        ul=int(ul)
        return I, ll,ul, roi_mask
    
    
    def process_image(self, I,msk, mn_slc, mn_col, mn_ht):
        
        I=I[mn_ht:mn_ht+192, mn_col:mn_col+192,mn_slc:mn_slc+32]
        msk=msk[mn_ht:mn_ht+192, mn_col:mn_col+192,mn_slc:mn_slc+32]
        
        I=I.astype(np.float)
        msk=msk.astype(np.float)
        ####I=I/256
        if np.max(I)>300:
            I=I/65535  # then most probably it is uint16
        else:
            I=I/255 # uint8
        #######
        I=(I*2)-1
        
        # Add channel dimension
        I=np.expand_dims(I, axis=0) # 1,H,W,D
        I=torch.FloatTensor(I)
        
        msk=np.expand_dims(msk, axis=0)
        msk=torch.FloatTensor(msk)
        return I, msk
        
        
        
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        # For the training set, the segmentation masks are not reqd. They are only used for validation.
        # Read images
        I_A, ll_A,ul_A, IA_msk=self.read_image(self.eye_id[index], self.visitA[index])
        I_B, ll_B,ul_B, IB_msk=self.read_image(self.eye_id[index], self.visitB[index])
        lbl=self.lbl[index]
        
        mn_ht=16#random.randint(0,32)
        mn_col=16#random.randint(0,32)
        mn_slc=2#random.randint(0,5)
        
        ll=min(ll_A, ll_B)
        if ((mn_ht>ll) and (ll>0)):
            mn_ht=ll
            
        I_A, IA_msk=self.process_image(I_A, IA_msk,mn_slc, mn_col, mn_ht)
        I_B, IB_msk=self.process_image(I_B, IB_msk,mn_slc, mn_col, mn_ht)
        
        nm_A=self.eye_id[index]+'_'+self.visitA[index]
        nm_B=self.eye_id[index]+'_'+self.visitB[index]
                
        sample={'I_A': I_A, 'IA_msk':IA_msk , 'IB_msk': IB_msk,'I_B': I_B, 'nm_A': nm_A, 'nm_B': nm_B,'lbl':lbl}
        return sample 

    def __len__(self):
        return self.eye_id.shape[0]

