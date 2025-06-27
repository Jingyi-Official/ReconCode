'''
Description: 
Author: 
Date: 2023-01-16 17:03:54
LastEditTime: 2023-04-21 23:53:07
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Reference: 
'''
from typing import Any, List
import json
import hydra
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from utilities.metrics.sdf_error import binned_errors
from utilities.tools.vis import plot_3D
from utilities.transforms.grid_transforms import get_eval_mesh_grid
from utilities.tools.calculate import get_sdf_pred_chunks
from utilities.transforms.sdf_transforms import sdf_render_mesh
import os
import trimesh

class Module(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        weights: dict,
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=True)
        self.net = self.hparams.net #.to(self.device)
        self.configure_optimizers()
        self.configure_eval_grid()

    
    def forward(self, x: torch.Tensor, do_grad):
        x = x.float()
        
        if do_grad:
            x.requires_grad_()

        return self.net(x, do_grad=do_grad)
  
    def training_step(self, batch, batch_idx):
        # forward ------------------------------------------------------------
        pc, sdf_gt = batch
        sdf_pred, grad_pred = self(pc, do_grad=True)

        # get loss -----------------------------------------------------------
        losses, tot_loss = self.cal_loss(sdf_pred, grad_pred, sdf_gt, grad_gt=None)
        error = self.cal_sdf_error(sdf_gt, sdf_pred)
        binned_error = self.cal_sdf_binned_error(error, sdf_gt)

        for each in losses.items():
            self.log(f"train_losses_{each[0]}", each[1])

        self.log("train_error_avg", error.mean())
        for each in binned_error.items():
            self.log(f"train_error_binned_{each[0]}", each[1])
        
        return tot_loss
    
    def validation_step(self, batch, batch_idx):
        # forward ------------------------------------------------------------
        pc, sdf_gt = batch
        sdf_pred, grad_pred = self(pc, do_grad=False)

        # get error -----------------------------------------------------------
        error = self.cal_sdf_error(sdf_gt, sdf_pred)
        binned_error = self.cal_sdf_binned_error(error, sdf_gt)

        self.log("val_error_avg", error.mean())
        for each in binned_error.items():
            self.log(f"val_error_binned_{each[0]}", each[1])
    
    def on_validation_epoch_end(self):
        # with torch.no_grad():
        #     plot_3D(self.net.decoder.get_grid().cpu().detach().numpy(), f"/root/recode/results/val_{self.current_epoch}.png")
            
        with torch.set_grad_enabled(False):
            sdf = get_sdf_pred_chunks(
                pc = self.eval_mesh_grid.to(self.device),
                fc_sdf_map = self.net
            )

            sdf = sdf.view(self.grid_dim)
        
        # plot_3D(sdf.cpu().detach().numpy(), os.path.join("/media/wanjingyi/Diskroom/ReconstructCode/", f"val_{self.global_step}.png"))    
        plot_3D(sdf.cpu().detach().numpy(), os.path.join(self.trainer.log_dir, f"val_{self.global_step}.png"))

    def on_test_end(self):
        
        with torch.set_grad_enabled(False):
            sdf = get_sdf_pred_chunks(
                pc = self.eval_mesh_grid.to(self.device),
                fc_sdf_map = self.net
            )

            sdf = sdf.view(self.grid_dim)
            
        plot_3D(sdf.cpu().detach().numpy(), os.path.join(self.trainer.log_dir, f"final.png"))
        
        mesh_pred = sdf_render_mesh(sdf, self.scene_scale, self.scene_mesh_bounds_transform,)
        mesh_pred = trimesh.exchange.ply.export_ply(mesh_pred)
        mesh_file = open(os.path.join(self.trainer.log_dir,f"mesh.ply"), "wb+")
        mesh_file.write(mesh_pred)
        mesh_file.close()
        
    
    def test_step(self, batch, batch_idx):
        # forward ------------------------------------------------------------
        pc, sdf_gt = batch
        sdf_pred, grad_pred = self(pc, do_grad=False)

        # get error -----------------------------------------------------------
        error = self.cal_sdf_error(sdf_gt, sdf_pred)
        binned_error = self.cal_sdf_binned_error(error, sdf_gt)

        self.log("test_error_avg", error.mean())
        for each in binned_error.items():
            self.log(f"test_error_binned_{each[0]}", each[1])
        

    def configure_optimizers(self):

        if isinstance(self.net, torch.nn.Module):
            parameters = self.net.parameters()
        else:
            parameters = self.net

        optimizer = self.hparams.optimizer(params=parameters)

        return {"optimizer": optimizer}
    
    def configure_eval_grid(self):
        self.scene_mesh_bounds_transform = np.array([
            [-6.15441689e-02,  9.98104361e-01, -6.38918612e-11, 9.52595752e-01], 
            [-9.98104361e-01, -6.15441689e-02, -3.38192839e-07, 1.50110338e+00],
            [-3.37555680e-07, -2.07500264e-08,  1.00000000e+00, 1.66513429e+00],
            [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
        ])
        self.scene_mesh_bounds_extents = np.array([3.5277844 ,  7.43068211, 12.99889375]) 
        self.eval_mesh_grid, self.scene_scale, self.grid_dim = get_eval_mesh_grid(self.device,self.scene_mesh_bounds_transform,self.scene_mesh_bounds_extents, grid_dim=[256,256,256])

    def cal_loss(self, sdf_pred, grad_pred, sdf_gt, grad_gt = None): 
        '''
        option 1: sdf loss
        option 2: eikonal loss
        '''
        losses = {}
        tot_loss = 0
        tot_loss_mat = 0

        # sdf loss
        sdf_loss_mat = None
        if self.hparams.weights.sdf_weight !=0:
            sdf_loss_mat = self.sdf_loss(sdf_pred, sdf_gt, loss_type="L1") # torch.Size([181, 27])
            sdf_loss_mat = sdf_loss_mat * self.hparams.weights.sdf_weight
            losses = {"sdf_loss": sdf_loss_mat.mean().item()}
            tot_loss_mat = tot_loss_mat + sdf_loss_mat

         
        # eik loss
        eik_loss_mat = None
        if self.hparams.weights.eik_weight != 0:
            eik_loss_mat = self.eik_loss(grad_pred)
            eik_loss_mat = eik_loss_mat * self.hparams.weights.eik_weight
            losses["eikonal_loss"] = eik_loss_mat.mean().item()
            tot_loss_mat = tot_loss_mat + eik_loss_mat

        tot_loss = tot_loss_mat.mean()
        losses["total_loss"] = tot_loss


        return losses, tot_loss
    
    def cal_sdf_error(self, sdf_gt, sdf_pred):
        return torch.abs(sdf_pred - sdf_gt)

    def cal_sdf_binned_error(self, sdf_loss, sdf_gt, bin_limits=np.array([-0.03, -0.02, -0.01, 0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])):
        sdf_binned_errors = binned_errors(sdf_loss, sdf_gt, bin_limits=bin_limits)
        return dict(zip(bin_limits[1:],sdf_binned_errors))


    def eik_loss(self, sdf_grad):
        eik_loss_mat = torch.abs(sdf_grad.norm(2, dim=-1) - 1)
        
        return eik_loss_mat     
    
    def sdf_loss(self, sdf_pred, sdf_gt, loss_type="L1"):
        """
            params:
            sdf: predicted sdf values.
            bounds: upper bound on abs(sdf)
            t: truncation distance up to which the sdf value is directly supevised. # to decide whether it's free space or near surface
            loss_type: L1 or L2 loss.
        """

        sdf_loss_mat = sdf_pred - sdf_gt

        if loss_type == "L1":
            sdf_loss_mat = torch.abs(sdf_loss_mat)
        elif loss_type == "L2":
            sdf_loss_mat = torch.square(sdf_loss_mat)
        else:
            raise ValueError("Must be L1 or L2")

        return sdf_loss_mat

    