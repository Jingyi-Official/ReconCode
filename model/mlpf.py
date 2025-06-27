'''
Description: 
Author: 
Date: 2022-09-19 21:50:03
LastEditTime: 2023-05-02 00:49:07
LastEditors: Jingyi Wan
Reference: 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities.transforms.point_transforms import transform_points, scale_points
from utilities.tools.calculate import cal_gradient_torch

class MLPF(nn.Module):
    def __init__(
        self,
        positional_encoder,
        decoder,
        scale_input=1.,
        transform_input=None,
        scale_output=1.,
        **kwargs,
    ) -> None:
        super().__init__()

        self.scale_input = scale_input
        self.transform_input = torch.FloatTensor(transform_input)
        self.positional_encoder = positional_encoder
        self.decoder = decoder
        self.scale_output = scale_output
    
    def forward(self, x, noise_std=None, do_grad=False):
        
        x_ts = scale_points(transform_points(x, transform=(self.transform_input).to(x.device)), scale=self.scale_input).squeeze()

        x_pe = self.positional_encoder(x_ts) # torch.Size([183, 27, 255])

        y = self.decoder(x_pe)

        
        if noise_std is not None: 
            noise = torch.randn(y.shape, device=x.device) * noise_std
            y = y + noise
            
        y = y * self.scale_output
        y = y.squeeze(-1)

        grad = cal_gradient_torch(x, y) if do_grad else None

        return y, grad

    


