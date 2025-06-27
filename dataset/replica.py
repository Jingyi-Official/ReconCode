from torch.utils.data import Dataset
import torch
import trimesh
import pysdf
import numpy as np
import os
import scipy
from utilities.geometry import get_ray_direction_camcoord
from utilities.transforms.grid_transforms import get_grid_pts


class ReplicaDataset(Dataset):
    
    def __init__(self,
                 scene_folder=None,
                 ):
        self.scene_folder = scene_folder
        self.queried_sdf = self.get_habitat_sdf()
        self.queries = get_grid_pts(self.queried_sdf.shape, self.habitat_transform)
        x, y, z = np.meshgrid(self.queries[0], self.queries[1], self.queries[2], indexing='ij')
        self.queries = np.stack([x, y, z],axis=-1).reshape(-1,3)#[:,None,:]
        self.queried_sdf = self.queried_sdf.flatten()#[:,None]
        self.queried_grad = None
        
    def __len__(self):
        return self.queried_sdf.shape[0]

    def __getitem__(self, idx):

        return self.queries[idx], self.queried_sdf[idx] #, self.queried_grad[idx]
    
    
    
    @property
    def fps(self):
        return 30
    
    @property
    def depth_H(self):
        return 680
    
    @property
    def depth_W(self):
        return 1200
    
    @property
    def rgb_H(self):
        return 680
    
    @property
    def rgb_W(self):
        return 1200

    @property
    def oriented_bounds(self):
        T_extent_to_scene, bounds_extents =  trimesh.bounds.oriented_bounds(self.scene_mesh)
        return T_extent_to_scene, bounds_extents

    @property # transform from the bbox of the scene to origin
    def inv_bounds_transform(self): 
        T_extent_to_scene, bounds_extents = self.oriented_bounds
        return T_extent_to_scene

    @property # transform from origin to the bbox of the scene
    def bounds_transform(self):
        T_extent_to_scene, bounds_extents = self.oriented_bounds
        return np.linalg.inv(T_extent_to_scene)

    @property
    def bounds_extents(self):
        T_extent_to_scene, bounds_extents = self.oriented_bounds
        return bounds_extents

    @property
    def dir_camcoord(self, type='z'):
        return get_ray_direction_camcoord(1, self.depth_H, self.depth_W, self.Ks[0,0], self.Ks[1,1], self.Ks[0,2], self.Ks[1,2], depth_type='z')

    @property
    def scene_mesh(self):
        return self.get_scene_mesh()

    def get_scene_mesh(self):
        return trimesh.load(self.get_scene_mesh_file())

    def get_scene_mesh_file(self):
        return os.path.join(self.scene_folder, 'mesh.obj')
    
    @property
    def scene_sdf_habitat(self):
        # default scene sdf provided
        queried_sdf = self.get_habitat_sdf()
        queries = get_grid_pts(queried_sdf.shape, self.habitat_transform)
        scene_sdf_habitat = scipy.interpolate.RegularGridInterpolator(queries, queried_sdf)

        return scene_sdf_habitat

    def get_habitat_sdf(self, queried_sdf_file = '1cm/sdf.npy'):
        queried_sdf_file = os.path.join(self.scene_folder, queried_sdf_file)
        return np.load(queried_sdf_file)

    def get_habitat_queries(self):
        queried_sdf = self.get_habitat_sdf()
        queries = get_grid_pts(queried_sdf.shape, self.habitat_transform)

        return queries

    @property
    def scene_sdf_stage_habitat(self):
        # default scene stage sdf provided
        queried_stage_sdf = self.get_habitat_stage_sdf()
        queries = get_grid_pts(queried_stage_sdf.shape, self.habitat_transform)
        scene_stage_sdf_habitat = scipy.interpolate.RegularGridInterpolator(queries, queried_stage_sdf)

        return scene_stage_sdf_habitat
    
    def get_habitat_stage_sdf(self, queried_stage_sdf_file = '1cm/stage_sdf.npy'):
        queried_stage_sdf_file = os.path.join(self.scene_folder, queried_stage_sdf_file)
        return np.load(queried_stage_sdf_file)

    @property
    def scene_sdf_pysdf(self):
        # scene sdf from pymesh by provided mesh
        mesh = self.scene_mesh
        scene_sdf_pysdf = pysdf.SDF(mesh.vertices, mesh.faces)

        return scene_sdf_pysdf

    @property
    def habitat_transform(self):
        return self.get_habitat_transform()

    def get_habitat_transform(self, queries_transf_file = '1cm/transform.txt'):
        queries_transf_file = os.path.join(self.scene_folder, queries_transf_file)
        return np.loadtxt(queries_transf_file)

    @property
    def scene_min_xy(self, bounds_file = 'bounds.txt'):
        bounds_file = os.path.join(self.root_dir, bounds_file)
        return np.loadtxt(bounds_file)

    @property
    def scene_islands(self, islands_file = 'unnavigable.txt'):
        islands_file = os.path.join(self.root_dir, islands_file)
        return np.loadtxt(islands_file)

    @property
    def scene_root_dir(self):
        return self.root_dir

    @property
    def scene_rgb_dir(self):
        return self.rgb_dir

    @property
    def scene_depth_dir(self):
        return self.depth_dir

    @property
    def up_camera(self):
        return np.array([0., 1., 0.])

    @property
    def up_world(self):
        return np.argmax(np.abs(np.matmul(self.up_camera, self.bounds_transform[:3, :3])))

    @property
    def up_grid(self):
        return self.bounds_transform[:3, self.up_world]

    @property
    def aligned_up(self):
        return np.dot(self.up_grid, self.up_camera) > 0
    
    @property
    def scene_bounds(self):
        return self.scene_mesh.bounds

    @property
    def bounds_corners(self):
        return trimesh.bounds.corners(self.scene_mesh.bounding_box_oriented.bounds)

    
