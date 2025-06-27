'''
Description: the revised version inspried by tensorf: to increase the inference ability, the encoding of the tensorf but replace the spatials into the mlp 
Author: 
Date: 2022-11-22 10:38:59
LastEditTime: 2023-05-19 20:42:15
LastEditors: Jingyi Wan
Reference: 
'''
"""
Implements image encoders
"""
import torch
from torch import nn
from torchtyping import TensorType
from model.decoder.mlp import MLP


class KMLPDecoding(nn.Module):

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


        self.yz_decoder = MLP(in_dim=in_dim, num_layers=num_layers[0],layer_width=layer_width[0], out_dim=out_dim, skip_connections=skip_connections, activation=activation, out_activation=out_activation)
        self.xz_decoder = MLP(in_dim=in_dim, num_layers=num_layers[1],layer_width=layer_width[1], out_dim=out_dim, skip_connections=skip_connections, activation=activation, out_activation=out_activation)
        self.xy_decoder = MLP(in_dim=in_dim, num_layers=num_layers[2],layer_width=layer_width[2], out_dim=out_dim, skip_connections=skip_connections, activation=activation, out_activation=out_activation)
    
        self.include_input = include_input
        if decoder:
            if self.include_input:
                self.decoder = decoder(in_dim=out_dim*3 + int(in_dim/2)*3)
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
        
        
        yz_features = self.yz_decoder(in_tensor[:, [1,2]].view(1,-1,self.yz_decoder.in_dim))
        xz_features = self.xz_decoder(in_tensor[:, [0,2]].view(1,-1,self.xz_decoder.in_dim))
        xy_features = self.xy_decoder(in_tensor[:, [0,1]].view(1,-1,self.xy_decoder.in_dim))

        

        if self.decoder is not None:
            if self.include_input:
                features = torch.cat([yz_features, xz_features, xy_features, in_tensor.view(1,in_tensor.shape[0],-1)], dim=-1)
            else:
                features = torch.cat([yz_features, xz_features, xy_features], dim=-1)
            out_tensor = self.decoder(features)
        else:
            features = (yz_features * xz_features * xy_features).squeeze()
            out_tensor = torch.sum(features, dim=-1)
        
        return out_tensor 
    

