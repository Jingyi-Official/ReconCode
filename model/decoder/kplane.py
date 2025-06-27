"""
Density proposal field
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from utilities.tools.grid_sample_gradfix import grid_sample


class KPlane(nn.Module):
    def __init__(self,
                 resolution = [727, 314, 1300],
                 num_components = 8,
                 init_weight = None,
                 decoder = None):
        super().__init__()

        # init the grid
        yx_plane_coef = nn.Parameter(torch.empty((1, num_components, resolution[1], resolution[0])))
        init_weight(yx_plane_coef)

        zx_plane_coef = nn.Parameter(torch.empty((1, num_components, resolution[2], resolution[0])))
        init_weight(zx_plane_coef)

        zy_plane_coef = nn.Parameter(torch.empty((1, num_components, resolution[2], resolution[1])))
        init_weight(zy_plane_coef)
        
        self.plane_coef = torch.nn.ParameterList([yx_plane_coef, zx_plane_coef, zy_plane_coef])

        self.decoder = decoder(in_dim=num_components)
 
    @torch.no_grad()
    def get_grid(self):
        return self.plane_coef[0] * self.plane_coef[1] * self.plane_coef[2]
    
    def forward(self, pts: torch.Tensor):

        feature_yz = grid_sample(self.plane_coef[0], einops.rearrange(pts[..., [0,1]], 'N xy -> 1 1 N xy')).view(self.plane_coef[0].shape[0], self.plane_coef[0].shape[1], pts.shape[0]).transpose(-1, -2).squeeze()

        feature_zx = grid_sample(self.plane_coef[1], einops.rearrange(pts[..., [0,2]], 'N xy -> 1 1 N xy')).view(self.plane_coef[1].shape[0], self.plane_coef[1].shape[1], pts.shape[0]).transpose(-1, -2).squeeze()
        
        feature_xy = grid_sample(self.plane_coef[2], einops.rearrange(pts[..., [1,2]], 'N xy -> 1 1 N xy')).view(self.plane_coef[2].shape[0], self.plane_coef[2].shape[1], pts.shape[0]).transpose(-1, -2).squeeze()


        features = feature_yz * feature_zx * feature_xy

        if self.decoder is not None:
            out = self.decoder(features)
        else:
            out = torch.sum(features, dim=-1)

        return out

    