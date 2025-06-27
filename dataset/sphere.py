import torch

from torch.utils.data import Dataset
class SphereDataset(Dataset):
    
    def __init__(
            self,
            xyz_dim = [100,100,100],
            R = 0.8,
            center = [0, 0, 0],
            xyz_min = [-1, -1, -1],
            xyz_max = [1, 1, 1]
        ):

        x = torch.linspace(xyz_min[0], xyz_max[0], xyz_dim[0])
        y = torch.linspace(xyz_min[1], xyz_max[1], xyz_dim[1])
        z = torch.linspace(xyz_min[2], xyz_max[2], xyz_dim[2])
        x,y,z = torch.meshgrid(x, y, z, indexing='ij')
        self.queries = torch.stack([x, y, z],axis=-1).reshape(-1,3)
        self.queried_sdf = torch.norm(self.queries - torch.Tensor(center), dim=-1) - R

        
    def __len__(self):
        return self.queried_sdf.shape[0]

    def __getitem__(self, idx):

        return self.queries[idx], self.queried_sdf[idx] 
    
    
    
    
