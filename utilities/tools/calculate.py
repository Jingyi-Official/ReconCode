'''
Description: 
Author: 
Date: 2022-09-19 21:49:24
LastEditTime: 2023-03-07 15:50:39
LastEditors: Jingyi Wan
Reference: 
'''
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
# from functorch import vmap, jacrev # , grad

def cal_gradient_torch(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


'''
description: 
param {*} inputs have to be (-1,3)
param {*} model
return {*}
'''
# def cal_gradient_functorch(inputs, model):
#     points_grad = vmap(jacrev(model))(inputs)
#     return points_grad

def get_sdf_pred_chunks(pc, fc_sdf_map, chunk_size = 100000, to_cpu=False):
    n_pts = pc.shape[0]
    n_chunks = int(np.ceil(n_pts / chunk_size))
    alphas = []
    for n in range(n_chunks):
        start = n * chunk_size
        end = start + chunk_size
        chunk = pc[start:end, :]

        alpha, _ = fc_sdf_map(chunk, do_grad=False) 

        alpha = alpha.squeeze(dim=-1)
        if to_cpu:
            alpha = alpha.cpu()
        alphas.append(alpha)

    alphas = torch.cat(alphas, dim=-1)

    return alphas
