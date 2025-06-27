'''
Description: 
Author: 
Date: 2023-04-17 20:20:32
LastEditTime: 2023-04-18 20:54:07
LastEditors: Jingyi Wan
Reference: 
'''
import torch
from torch import nn
import torch.nn.functional as F
from utilities.tools.calculate import cal_gradient_torch
from utilities.tools.grid_sample_gradfix import grid_sample

class Grid3D(nn.Module):
    def __init__(
            self, 
            resolution, 
            init_weight = None,
            decoder = None
        ):
        super(Grid3D, self).__init__()

        self.grid = nn.Parameter(torch.zeros([1, 1, *resolution]))
        init_weight(self.grid)

        self.decoder = decoder

    @torch.no_grad()
    def get_grid(self):
        return self.grid
    
    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        features = F.grid_sample(self.grid, xyz.flip((-1,)), mode='bilinear', align_corners=True)
        features = features.reshape(1,-1).T.reshape(*shape,1) 
    
        if self.decoder is not None:
            out = self.decoder(features)
        else:
            out = features.squeeze(-1)
            
        return out
    

class Grid2D(nn.Module):
    def __init__(self, world_size, xy_min, xy_max, **kwargs):
        super(Grid2D, self).__init__()
        self.world_size = world_size
        self.register_buffer('xy_min', torch.Tensor(xy_min))
        self.register_buffer('xy_max', torch.Tensor(xy_max))
        self.grid = nn.Parameter(torch.zeros([1, 1, *world_size]))

    def forward(self, xy, do_grad=False):
        '''
        xy: global coordinates to query
        '''
        shape = xy.shape[:-1]
        xy = xy.reshape(1,1,-1,2)
        ind_norm = ((xy - self.xy_min) / (self.xy_max - self.xy_min)) * 2 - 1
        out = grid_sample(self.grid, ind_norm)
        out = out.reshape(1,-1).T.reshape(*shape,1)
        out = out.squeeze(-1)
        
        grad = cal_gradient_torch(xy, out) if do_grad else None
        
        return out, grad