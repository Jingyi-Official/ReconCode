import torch

from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

class CircleDataset(Dataset):
    
    def __init__(self, xy_dim=[100, 100], R=0.5, center=[0,0], xy_min=[-1,-1], xy_max=[1,1]):

        x = torch.linspace(xy_min[0], xy_max[0], xy_dim[0])
        y = torch.linspace(xy_min[1], xy_max[1], xy_dim[1])
        x,y = torch.meshgrid(x, y, indexing='xy')
        self.queries = torch.stack([x, y],axis=-1).reshape(-1,2)
        self.queried_sdf = torch.norm(self.queries - torch.Tensor(center), dim=-1) - R

        print("Finish loading circle dataset.")

        
    def __len__(self):
        return self.queried_sdf.shape[0]

    def __getitem__(self, idx):

        return self.queries[idx], self.queried_sdf[idx] 
    
    
    
    
