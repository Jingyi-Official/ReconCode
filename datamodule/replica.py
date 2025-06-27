'''
Description: 
Author: 
Date: 2023-01-16 16:54:25
LastEditTime: 2023-04-24 04:12:53
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Reference: 
'''
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from dataset.replica import ReplicaDataset


class ReplicaDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
    ):
        super().__init__()

        self.save_hyperparameters(logger=True)

    
    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_val_dataset = ReplicaDataset(self.hparams.data_dir)
            n_train = int(len(train_val_dataset)*0.9)
            n_val = len(train_val_dataset) - n_train
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_val_dataset, [n_train, n_val])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = ReplicaDataset(self.hparams.data_dir)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4
        )
    
    def test_dataloader(self):
        
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4
        )

