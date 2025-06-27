'''
Description: the revised version inspried by tensorf: to increase the inference ability, the encoding of the tensorf but replace the spatials into the mlp 
Author: 
Date: 2022-11-22 10:38:59
LastEditTime: 2023-05-19 20:19:10
LastEditors: Jingyi Wan
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
from model.decoder.mlp import MLP


class CMLPDecoding(nn.Module):
    """Learned CANDECOMP/PARFAC (CP) decomposition encoding used in TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    def __init__(self, 
                 in_dim,
                 num_layers,
                 layer_width,
                 out_dim,
                 skip_connections,
                 activation,
                 out_activation,
                 decoder,
                 include_input=False,
                 ) -> None:
        super().__init__()

        self.x_decoder = MLP(in_dim=in_dim, num_layers=num_layers[0],layer_width=layer_width[0], out_dim=out_dim, skip_connections=skip_connections, activation=activation, out_activation=out_activation)
        self.y_decoder = MLP(in_dim=in_dim, num_layers=num_layers[1],layer_width=layer_width[0], out_dim=out_dim, skip_connections=skip_connections, activation=activation, out_activation=out_activation)
        self.z_decoder = MLP(in_dim=in_dim, num_layers=num_layers[2],layer_width=layer_width[0], out_dim=out_dim, skip_connections=skip_connections, activation=activation, out_activation=out_activation)
        
        self.include_input = include_input
        if decoder:
            if self.include_input:
                self.decoder = decoder(in_dim=out_dim*3 + in_dim*3)
            else:
                self.decoder = decoder(in_dim=out_dim*3)
        else: 
            self.decoder = None

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"], do_grad=False) -> TensorType["bs":..., "output_dim"]:
        if in_tensor.shape[-1] > 3:
            in_tensor = in_tensor.reshape(in_tensor.shape[0], 2, -1)
            in_tensor_x = in_tensor[:,:,0:int(in_tensor.shape[-1]/3)]
            in_tensor_y = in_tensor[:,:,int(in_tensor.shape[-1]/3):int(in_tensor.shape[-1]/3*2)]
            in_tensor_z = in_tensor[:,:,int(in_tensor.shape[-1]/3*2):in_tensor.shape[-1]]
            in_tensor = torch.cat([in_tensor_x.reshape(in_tensor_x.shape[0], 1, -1), in_tensor_y.reshape(in_tensor_y.shape[0], 1, -1), in_tensor_z.reshape(in_tensor_z.shape[0], 1, -1)], dim=1)
        elif in_tensor.shape[-1]==3:
            in_tensor = in_tensor.unsqueeze(dim=-1)
            
        x_features = self.x_decoder(in_tensor[:, 0].view(1,-1,self.x_decoder.in_dim))
        y_features = self.y_decoder(in_tensor[:, 1].view(1,-1,self.y_decoder.in_dim))
        z_features = self.z_decoder(in_tensor[:, 2].view(1,-1,self.z_decoder.in_dim))


        if self.decoder is not None:
            if self.include_input:
                features = torch.cat([x_features, y_features, z_features, in_tensor.view(1,in_tensor.shape[0],-1)], dim=-1)
            else:
                features = torch.cat([x_features, y_features, z_features], dim=-1)
            out_tensor = self.decoder(features)
        else:
            line_features = torch.cat([x_features, y_features, z_features], dim=0)
            features = torch.prod(line_features, dim=0)
            out_tensor = torch.sum(features, dim=-1)
        
        return out_tensor 
    

