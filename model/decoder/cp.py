'''
Description: the encoding of the tensorf: tensorcp; rewrite from nerfstudio
Author: 
Date: 2022-11-22 10:38:59
LastEditTime: 2023-04-24 04:08:21
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Reference: 
'''
"""
Implements image encoders
"""
import torch
from torch import nn
import torchvision
import numpy as np
from torchtyping import TensorType
import torch.nn.functional as F
from utilities.tools.grid_sample_gradfix import grid_sample


class CPDecoding(nn.Module):

    def __init__(self, 
                 resolution: list = [256, 256, 256], 
                 num_components: int = 96, 
                 init_weight = None,
                 decoder = None):
        super().__init__()

        # init grid
        line_coef_z = nn.Parameter(torch.empty((1, num_components, resolution[2], 1)))
        init_weight(line_coef_z)

        line_coef_y = nn.Parameter(torch.empty((1, num_components, resolution[1], 1)))
        init_weight(line_coef_y)

        line_coef_x = nn.Parameter(torch.empty((1, num_components, resolution[0], 1)))
        init_weight(line_coef_x)

        self.line_coef = torch.nn.ParameterList([line_coef_z, line_coef_y, line_coef_x])

        self.decoder = decoder
    
    @torch.no_grad()
    def get_grid(self):
        features = torch.matmul(torch.matmul(self.line_coef[0].squeeze()[:,:, None], self.line_coef[1].squeeze()[:,None,:])[:,:,:,None], self.line_coef[2].squeeze()[:,None,None,:])
        grid = torch.sum(features, dim=0)
        return grid
    
    
    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"], do_grad=False) -> TensorType["bs":..., "output_dim"]:
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...] zyx
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        line_coord = line_coord.view(3, -1, 1, 2)

        line_features_z = grid_sample(self.line_coef[0], line_coord[[0]]) 
        line_features_y = grid_sample(self.line_coef[1], line_coord[[1]])
        line_features_x = grid_sample(self.line_coef[2], line_coord[[2]])
        line_features = torch.cat([line_features_z, line_features_y, line_features_x], dim=0)
    
        features = torch.prod(line_features, dim=0)
        features = torch.moveaxis(features.view(features.shape[0], *in_tensor.shape[:-1]), 0, -1)
        
        if self.decoder is not None:
            out_tensor = self.decoder(features)
        else:
            out_tensor = torch.sum(features, dim=-1)
        
        return out_tensor 
