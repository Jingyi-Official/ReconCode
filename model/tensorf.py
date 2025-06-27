'''
Description: 
Author: 
Date: 2022-09-19 21:50:03
LastEditTime: 2023-04-24 04:04:55
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Reference: 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities.transforms.point_transforms import transform_points, scale_points
from utilities.tools.calculate import cal_gradient_torch

class TensoRF(nn.Module):
    def __init__(
        self,
        xyz_min,
        xyz_max,
        positional_encoder,
        decoder,
        **kwargs,
    ) -> None:
        super().__init__()

        self.xyz_min = torch.FloatTensor(xyz_min)
        self.xyz_max = torch.FloatTensor(xyz_max)

        self.positional_encoder = positional_encoder
        self.decoder = decoder

    
    def forward(self, x, noise_std=None, do_grad=False):
        
        x_ts = ((x - self.xyz_min.to(x.device)) / (self.xyz_max.to(x.device) - self.xyz_min.to(x.device))) * 2 - 1 
        
        x_pe = self.positional_encoder(x_ts)
        
        y = self.decoder(x_pe)

        
        if noise_std is not None: 
            noise = torch.randn(y.shape, device=x.device) * noise_std
            y = y + noise
            
        # y = y * self.scale_output 
        y = y.squeeze(-1)

        grad = cal_gradient_torch(x, y) if do_grad else None
        # grad = None
        
        return y, grad
    
    @torch.no_grad()
    def reg_values(self):
        if hasattr(self.decoder, 'plane_coef'):
            total = 0
            for idx in range(len(self.decoder.line_coef)):
                total = total + torch.mean(torch.abs(self.decoder.plane_coef[idx])) + torch.mean(torch.abs(self.decoder.line_coef[idx]))
            return total
        else:
            total = 0
            for idx in range(len(self.decoder.line_coef)):
                total = total + torch.mean(torch.abs(self.decoder.line_coef[idx]))
            return total
        
        


