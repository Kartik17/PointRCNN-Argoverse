import os
import numpy as np
import torch.utils.data as torch_data
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import lib.datasets.ground_segmentation as gs
from pyntcloud import PyntCloud
import random


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train'):
        self.split = split
        is_test = (self.split == 'test')
        self.lidar_pathlist = []
        self.label_pathlist = []
        
        self.lidar_dir = os.path.join(root_dir)
        data_loader = ArgoverseTrackingLoader(os.path.join(root_dir))
        self.lidar_pathlist.extend(data_loader.lidar_list)
        self.label_pathlist.extend(data_loader.label_list)
        #self.calib_file = data_loader.calib_filename
        self.lidar_filename = [x.split('.')[0].rsplit('/',1)[1] for x in self.lidar_pathlist]
        
        assert len(self.lidar_pathlist) == len(self.label_pathlist)

        self.num_sample = len(self.lidar_pathlist)
        self.lidar_idx_list = ['%06d'%l for l in range(self.num_sample)]
        
        self.lidar_idx_table = dict(zip(self.lidar_idx_list, self.lidar_filename))
        self.argo_to_kitti = np.array([[0, -1, 0],
                                       [0, 0, -1],
                                       [1, 0, 0 ]])

        self.ground_removal = False
        
        self.lidar_dir = os.path.join('/data/')        
        self.label_dir = os.path.join('/data/')
        
    def get_lidar(self,idx):
        lidar_file = self.lidar_pathlist[idx]
        assert os.path.exists(lidar_file)
        
        
        data = PyntCloud.from_file(lidar_file)
        x = np.array(data.points.x)[:, np.newaxis]
        y = np.array(data.points.y)[:, np.newaxis]
        z = np.array(data.points.z)[:, np.newaxis]
        pts_lidar = np.concatenate([x,y,z], axis = 1)
        
        if self.ground_removal: 
            pts_lidar = gs.ground_segmentation(pts_lidar)
        
        pts_lidar = np.dot(self.argo_to_kitti,pts_lidar.T).T
        
        return pts_lidar
        
        
    def get_label(self,idx):
        
        label_file = self.label_pathlist[idx]
        assert os.path.exists(label_file)
        
        return kitti_utils.get_objects_from_label(label_file)
    

