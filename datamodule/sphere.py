
from pytorch_lightning import LightningDataModule
from dataset.sphere import SphereDataset
import torch
from torch.utils.data import DataLoader




class SphereDataModule(LightningDataModule):

    def __init__(
        self,
        xyz_dim = [100,100,100],
        R = 0.8,
        center = [0, 0, 0],
        xyz_min = [-1, -1, -1],
        xyz_max = [1, 1, 1],
        batch_size = 1000,
    ):
        super().__init__()

        self.save_hyperparameters(logger=True)
    
    def prepare_data(self):
        # Loading the train/val set
        train_dataset = SphereDataset(self.hparams.xyz_dim, self.hparams.R, self.hparams.center, self.hparams.xyz_min, self.hparams.xyz_max)
        n_train = int(len(train_dataset)*0.9)
        n_val = len(train_dataset) - n_train
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_val])

        # Loading the test set
        self.test_dataset = SphereDataset(self.hparams.xyz_dim, self.hparams.R, self.hparams.center, self.hparams.xyz_min, self.hparams.xyz_max)


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


