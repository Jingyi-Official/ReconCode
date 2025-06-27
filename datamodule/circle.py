
from torch.utils.data import Dataset
import torch
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from dataset.circle import CircleDataset


class CircleDataModule(LightningDataModule):

    def __init__(
        self,
        xy_dim,
        R,
        center,
        xy_min,
        xy_max,
        batch_size: int = 64,
    ):
        super().__init__()

        self.save_hyperparameters()
        
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    
    def prepare_data(self):
        # Loading the train/val set
        train_dataset = CircleDataset(self.hparams.xy_dim, self.hparams.R, self.hparams.center, self.hparams.xy_min, self.hparams.xy_max)
        n_train = int(len(train_dataset)*0.9)
        n_val = len(train_dataset) - n_train
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_val])

        # Loading the test set
        self.test_dataset = CircleDataset(self.hparams.xy_dim, self.hparams.R, self.hparams.center, self.hparams.xy_min, self.hparams.xy_max)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=200000,
            shuffle=True
        )


