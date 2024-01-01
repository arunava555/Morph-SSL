#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
1. The code to create a meshgrid of spatial dimensions. 
2. Applying spatial deformation field in a differentiable manner similar to the Spatial Transformer Networks
3. The regularization losses on the Deformation Field to ensure smoothness and discourage folding.
'''
################# smoothness loss for the deformation field #################
def comp_smooth_loss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0

#################### compute_fold_loss(grid) #######################
def compute_fold_loss(grid):
    #eps=10**(-9)
    eps=0.0
    # for no fold all the values will be negative and relu will make them 0
    dx=F.relu(grid[:,0,:-1,:,:]-grid[:,0,1:,:,:]+eps)
    dy=F.relu(grid[:,1,:,:-1,:]-grid[:,1,:,1:,:]+eps)
    dz=F.relu(grid[:,2,:,:,:-1]-grid[:,2,:,:,1:]+eps)
    
    loss=F.mse_loss(dx, torch.zeros_like(dx),reduction='mean')+\
    F.mse_loss(dy, torch.zeros_like(dy),reduction='mean')+F.mse_loss(dz, torch.zeros_like(dz),reduction='mean')
    
    return loss


####################  Spatial Transformer Network ####################

def create_mesh_grid(H,W,D):
    vectors = [torch.arange(0, s) for s in [H,W,D]]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0) # 1,3,H,W,D  : 3 channels give position of x,y & z co-ordinates
    grid = grid.to(device=device, dtype=torch.float)
    
    # Normalize grid positions to [0,1]
    grid[:,0,:,:,:]=grid[:,0,:,:,:]/(H-1)# for height H, range is [0,H-1], so max=H-1
    grid[:,1,:,:,:]=grid[:,1,:,:,:]/(W-1)
    grid[:,2,:,:,:]=grid[:,2,:,:,:]/(D-1)
    # Now normalize to [-1,1]
    grid=2*grid-1
    return grid


def spatial_transform(img, deform_fld, grid,intrp):
    
    B,C,H,W,D=img.shape
    grid=grid+deform_fld
    fold_loss=compute_fold_loss(grid)
    
    ############ From the Voxel Morph Code ###
    grid=grid.permute(0, 2, 3, 4, 1) # move channels to last dim N,H,W,D,3
    grid=grid[..., [2, 1, 0]] # reverse channels 
    out=F.grid_sample(img, grid, align_corners=True, mode=intrp)
    return out, fold_loss

