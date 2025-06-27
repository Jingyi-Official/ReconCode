'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-04-20 19:35:29
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-04-20 20:07:33
FilePath: /ReconstructCode/model/mlp2d.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch import nn
import torch.nn.functional as F
from utilities.tools.calculate import cal_gradient_torch
from utilities.tools.grid_sample_gradfix import grid_sample
from model.decoder.mlp import MLP

class MLP2D(nn.Module):
    def __init__(
        self,
        decoder,
    ) -> None:
        super(MLP2D, self).__init__()
        
        self.decoder = decoder

    def forward(self, xyz, do_grad=False):
        '''
        xyz: global coordinates to query
        '''
        
        out = self.decoder(xyz).squeeze()
        
        grad = cal_gradient_torch(xyz, out) if do_grad else None
        # grad=None

        
        return out, grad