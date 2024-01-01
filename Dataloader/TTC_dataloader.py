#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random 
from skimage import filters
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F


# In[ ]:





# In[ ]:


class train_dataset(Dataset):
    def __init__(self, fold, img_pth):
        
        ##### read the training set ######
        '''
        fold is an npz file containing list of scans in the current fold
        # trn_eye_lst[index]+'__'+trn_visit_lst[index] is the name of the OCT scan uniquely identified by the 
        EyeID and the visit date.
        '''
        a=np.load(fold)
        self.eye_lst=a['trn_eye_lst']
        self.visit_lst=a['trn_visit_lst']
        self.t_cnv_lst=a['trn_t_cnv']
        self.t_bfr_lst=a['trn_t_bfr']
        del a
        
        # params for random noise for Data Aug
        self.std=0.001
        self.mean=0
        
        ############################################################################################
        """
         pth is the directory containing the preprocessed Scans for TTC task. 
         Each preprocessed scan is saved with a name "eyeID__visitdate.npz".
         Each npz file contains :
         a) img:preprocessed OCT vol of size Height(192+margin of 32), Width(192+margin of 32), Depth(32+margin of 5).
               img is H X W X D
        b) roi_mask: binary ROI mask of the region in img containing the retinal tissue
        c) ll, ul: axial(Height) bounds (lower and upper bound) of the ROI mask          
        """
        self.pth=img_pth
        ############################################################################################
        
    def find_time_boundaries(self, t_cnv, t_bfr):
        
        if t_cnv>(18*30):# 18 month
            # treat as if it will not convert
            # The GT will be 0 all the way through from 0-18 months
            t1=0
            t2=(18*30)
            gt1=0
            gt2=0
        elif (t_cnv==0) and (t_bfr<0):
            # Already converted (t_cnv==0 implies just converted)
            # GT will be 1 all the way through from 0-18 months
            t1=0
            t2=(18*30) # days
            gt1=1
            gt2=1
        elif (t_cnv==0) and (t_bfr>0):
            # not possible
            pass
        elif (t_cnv>0) and (t_bfr<0):
            # not possible
            pass
        elif (t_cnv>0) and (t_bfr>0):
            # most samples will be this
            # 0-t_bfr will be 0(not converted), t_cnv-18 will be 1(converted). In between t_bfr-t_cnv is unknown.
            # We only consider the boundary positions of conversion t_bfr and t_cnv 
            t1=t_bfr # t_bfr=0 implies all the ones before it will also be 0
            t2=t_cnv # t_cnv=1 implies everything after it will also be 1.
            gt1=0
            gt2=1
        elif (t_cnv==0) and (t_bfr==0):
            # also "already converted"
            t1=0
            t2=(18*30) # days
            gt1=1
            gt2=1
        elif (t_cnv>0) and (t_bfr==0):
            # this is the last visit before the conversion visit
            t1=t_bfr
            t2=t_cnv
            gt1=0
            gt2=1
        
        # Normalize [0-18] months to [0-1] range
        t1=t1/(18*30)
        t2=t2/(18*30)
        t=np.array([t1,t2])
        gt=np.array([gt1, gt2])
        return t,gt
    
        
        
        
    def read_image(self, eye_id, visit):
        # f'{visit:04d}'
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
        
        # random blurring of image for data aug
        r=random.random()*0.90 # between [0,1]
        I = filters.gaussian(I, sigma=(r, r,r), truncate=3)
        
        # Add channel dimension
        I=np.expand_dims(I, axis=0) # 1,H,W,D
        I=torch.FloatTensor(I)
        
        msk=zoom(msk, (1/16, 1/16, 1/2), order=0)
        msk=np.expand_dims(msk, axis=0)
        msk=torch.FloatTensor(msk)
        
        # Add random noise for data aug
        I=I + torch.randn(I.size()) * self.std + self.mean
        
        # Random Flipping
        if random.random()<=0.5:
            I=torch.flip(I,dims=[2])
            msk=torch.flip(msk,dims=[2])
        
        return I, msk
        
        
        
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        # For the training set, the segmentation masks are not reqd. They are only used for validation.
        # Read images
        I, ll,ul, roi_msk=self.read_image(self.eye_lst[index], self.visit_lst[index])
        
        mn_ht=random.randint(0,32)
        mn_col=random.randint(0,32)
        mn_slc=random.randint(0,5)
        
        if ((mn_ht>ll) and (ll>0)):
            mn_ht=ll
            
        I, roi_msk=self.process_image(I, roi_msk,mn_slc, mn_col, mn_ht)
        nm=self.eye_lst[index]+'__'+self.visit_lst[index]
        
        t_cnv=self.t_cnv_lst[index]
        t_bfr=self.t_bfr_lst[index]
        t,gt=self.find_time_boundaries(t_cnv, t_bfr)
        
        # Convert to pytorch tensor
        t=torch.FloatTensor(t)
        gt=torch.FloatTensor(gt)
        
        
        t_cnv=t_cnv/(18*30)
        t_bfr=t_bfr/(18*30)
        
        sample={'I': I, 'gt': gt, 't': t, 't_cnv':t_cnv, 't_bfr':t_bfr, 'roi_msk':roi_msk, 'nm': nm}
        return sample 

    def __len__(self):
        return self.eye_lst.shape[0]


# In[ ]:





# In[ ]:


class val_dataset(Dataset):
    def __init__(self, fold, img_pth):
        
        # read the training set
        a=np.load('/msc/home/achakr83/PINNACLE/SSL_training/June30/cross_validation_splits_new/fold'+str(fold)+'.npz')
        self.eye_lst=a['val_eye_lst']
        self.visit_lst=a['val_visit_lst']
        self.t_cnv_lst=a['val_t_cnv']
        self.t_bfr_lst=a['val_t_bfr']
        del a
        self.pth=img_pth
        
        
    def compute_gt(self, t_cnv, t_bfr, thresh):
        if t_cnv<=thresh:
            # already converted
            gt=1
        else:
            # t_cnv occurs afterwards, 
            # check the t_bfr. If it occurs after as well then we are sure that the gt=0
            if t_bfr>=thresh:
                gt=0
            else: 
                # the t_bfr occurs before thresh but t_cnv occurs after it
                # So, actual time-point of conversion could strictly occur either < or > thresh as it falls in the uncertain region
                gt=-1       
        return gt
    
    
    
    def find_time_boundaries(self, t_cnv, t_bfr):
        # t is 3 time-points 6 months, 12 months and 18 months (consider 30 days per month)
        t0=0
        t1=(6*30)/(18*30) # 6 month in days
        t2=(12*30)/(18*30)
        t3=(18*30)/(18*30)
        
        if t_cnv<=0:
            # already converted at current time-point
            gt0=1
        else:
            gt0=0
            
        gt1=self.compute_gt(t_cnv, t_bfr,(6*30))
        gt2=self.compute_gt(t_cnv, t_bfr,(12*30))
        gt3=self.compute_gt(t_cnv, t_bfr,(18*30))
        
        gt=np.array([gt0,gt1,gt2,gt3])
        t=np.array([t0,t1,t2,t3])
        return t,gt
    
        
        
        
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
        
        msk=zoom(msk, (1/16, 1/16, 1/2), order=0)
        msk=np.expand_dims(msk, axis=0)
        msk=torch.FloatTensor(msk)
        
        return I, msk
        
        
        
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        # For the training set, the segmentation masks are not reqd. They are only used for validation.
        # Read images
        I, ll,ul, roi_msk=self.read_image(self.eye_lst[index], self.visit_lst[index])
        
        mn_ht=16#random.randint(0,32)
        mn_col=16#random.randint(0,32)
        mn_slc=2#random.randint(0,5)
        
        if ((mn_ht>ll) and (ll>0)):
            mn_ht=ll
            
        I, roi_msk=self.process_image(I, roi_msk,mn_slc, mn_col, mn_ht)
        nm=self.eye_lst[index]+'__'+self.visit_lst[index]
        
        t_cnv=self.t_cnv_lst[index]
        t_bfr=self.t_bfr_lst[index]
        t,gt=self.find_time_boundaries(t_cnv, t_bfr)
        
        # Convert to pytorch tensor
        t=torch.FloatTensor(t)
        gt=torch.FloatTensor(gt)
        
        
        t_cnv=t_cnv/(18*30)
        t_bfr=t_bfr/(18*30)
        
        sample={'I': I, 'gt': gt, 't': t, 't_cnv':t_cnv, 't_bfr':t_bfr, 'roi_msk':roi_msk, 'nm': nm}
        return sample 

    def __len__(self):
        return self.eye_lst.shape[0]

