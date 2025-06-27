'''
Description: the encoding of the tensorf: tensorvm
Author: 
Date: 2023-02-27 09:35:48
LastEditTime: 2023-05-19 15:50:52
LastEditors: Jingyi Wan
Reference: 
'''
import torch
from torch import nn
import torchvision
import numpy as np
from torchtyping import TensorType
import torch.nn.functional as F
from utilities.tools.grid_sample_gradfix import grid_sample


class VMDecoding(nn.Module):
    """Learned vector-matrix encoding proposed by TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    plane_coef: TensorType[3, "num_components", "resolution", "resolution"]
    line_coef: TensorType[3, "num_components", "resolution", 1]

    def __init__(
        self,
        resolution: list = [128, 128, 128], #[x,y,z]
        num_components: list = [24, 24, 24], #[z,y,x]
        init_weight = None,
        decoder = None,
    ) -> None:
        super().__init__()

        plane_coef_yx = nn.Parameter(torch.empty((1, num_components[0], resolution[1], resolution[0])))
        init_weight(plane_coef_yx)
        line_coef_z = nn.Parameter(torch.empty((1, num_components[0], resolution[2], 1)))
        init_weight(line_coef_z)

        plane_coef_zx = nn.Parameter(torch.empty((1, num_components[1], resolution[2], resolution[0])))
        init_weight(plane_coef_zx)
        line_coef_y = nn.Parameter(torch.empty((1, num_components[1], resolution[1], 1)))
        init_weight(line_coef_y)

        plane_coef_zy = nn.Parameter(torch.empty((1, num_components[2], resolution[2], resolution[1])))
        init_weight(plane_coef_zy)
        line_coef_x = nn.Parameter(torch.empty((1, num_components[2], resolution[0], 1)))
        init_weight(line_coef_x)

        self.plane_coef = torch.nn.ParameterList([plane_coef_yx, plane_coef_zx, plane_coef_zy])
        self.line_coef = torch.nn.ParameterList([line_coef_z, line_coef_y, line_coef_x])

        self.decoder = decoder

    def get_out_dim(self) -> int:
        return self.num_components * 3

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"], do_grad=False) -> TensorType["bs":..., "output_dim"]:
        """Compute encoding for each position in in_positions

        Args:
            in_tensor: position inside bounds in range [-1,1], # torch.Size([1000, 27, 3])

        Returns: Encoded position
        """
        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]])  # [3,...,2]
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        plane_coord = plane_coord.view(3, -1, 1, 2)#.detach()
        line_coord = line_coord.view(3, -1, 1, 2)#.detach()
        
        plane_features_yx = grid_sample(self.plane_coef[0], plane_coord[[0]]) # torch.Size([1, 32, 27000, 1])
        line_features_z = grid_sample(self.line_coef[0], line_coord[[0]]) # torch.Size([1, 32, 27000, 1])

        plane_features_zx = grid_sample(self.plane_coef[1], plane_coord[[1]])
        line_features_y = grid_sample(self.line_coef[1], line_coord[[1]])

        plane_features_zy = grid_sample(self.plane_coef[2], plane_coord[[2]])
        line_features_x = grid_sample(self.line_coef[2], line_coord[[2]])

        features_1 = torch.sum((plane_features_yx*line_features_z).squeeze(), dim=0) # torch.Size([27000])
        features_2 = torch.sum((plane_features_zx*line_features_y).squeeze(), dim=0)
        features_3 = torch.sum((plane_features_zy*line_features_x).squeeze(), dim=0)

        out_tensor = features_1 + features_2 + features_3
        out_tensor = out_tensor.reshape(*in_tensor.shape[:-1],1)

        if self.decoder is not None:
            out_tensor = self.decoder(out_tensor)
        else:
            out_tensor = torch.sum(out_tensor, dim=-1)

        
        return out_tensor # [..., 3 * Components]
