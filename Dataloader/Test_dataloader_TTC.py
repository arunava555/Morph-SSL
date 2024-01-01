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


class test_dataset(Dataset):
    def __init__(self, img_pth, fold):
        
        '''
        fold is an npz file containing list of scans in the current fold
        # tst_eye_lst[index]+'__'+tst_visit_lst[index] is the name of the OCT scan uniquely identified by the 
        EyeID and the visit date.
        '''
        # read the training set
        # '/msc/home/achakr83/PINNACLE/SSL_training/June30/cross_validation_splits_new/fold'+str(fold)+'.npz'
        a=np.load(fold)
        self.eye_lst=a['tst_eye_lst']
        self.visit_lst=a['tst_visit_lst']
        self.t_cnv_lst=a['tst_t_cnv']
        self.t_bfr_lst=a['tst_t_bfr']
        del a
        
        # 
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


# In[ ]:




