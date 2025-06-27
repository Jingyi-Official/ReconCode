'''
Description: This is the easycode for training without continual learning
Author: 
Date: 2022-09-19 21:49:18
LastEditTime: 2023-05-19 16:25:57
LastEditors: Jingyi Wan
Reference: 
'''
import os
import warnings
import trimesh

import hydra
from omegaconf import DictConfig,OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Callback, Trainer
from pytorch_lightning.loggers import Logger # LightningLoggerBase
import torch
from dataset.sphere import SphereDataset
from utilities.transforms.grid_transforms import make_3D_grid
from utilities.transforms.sdf_transforms import sdf_render_mesh
from utilities.tools.vis import plot_3D

warnings.filterwarnings("ignore")

from typing import Any, Callable, Dict, List
import numpy as np

# A logger for this file
import logging
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(cfg:DictConfig):

    log.info("Set device.")
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    log.info("Set seed for random number generators in pytorch, numpy and python.random")
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating dataset <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
    log.info(f"Instantiating model <{cfg.modelmodule._target_}>")
    modelmodule: LightningModule = hydra.utils.instantiate(cfg.modelmodule)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = []  
    for _, cb in cfg.get("callback").items():
        if isinstance(cb, DictConfig) and "_target_" in cb:
            log.info(f"Instantiating callback <{cb._target_}>")
            callbacks.append(hydra.utils.instantiate(cb))
    
    log.info(f"Instantiating logger <{cfg.logger._target_}>")
    logger: Logger = hydra.utils.instantiate(cfg.logger)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("** Start training **")  
    trainer.fit(model=modelmodule, datamodule=datamodule)
    log.info("** Finish training **") 

    log.info("** Start testing **") 
    trainer.test(model=modelmodule, datamodule=datamodule, ckpt_path="last")
    log.info("** Finish testing **") 

    
if __name__ == "__main__":
    main()