# -*- coding: utf-8 -*-
import sys
import numpy as np
import torch

from opencood.data_utils.pre_processor.base_preprocessor import BasePreprocessor


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params, train)

        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']
        self.num_point_features = 4

        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']

        grid_size = (
            np.array(self.lidar_range[3:6]) - np.array(self.lidar_range[0:3])
        ) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        self.spconv_mode = None
        self.voxel_generator = None
        self.tv = None

        # ---------- spconv 1.x ----------
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_mode = "spconv1"
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.lidar_range,
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels
            )
            print("[SpVoxelPreprocessor] Using spconv 1.x: VoxelGeneratorV2")
            return
        except Exception:
            pass

        try:
            from spconv.utils import VoxelGenerator
            self.spconv_mode = "spconv1"
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.lidar_range,
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels
            )
            print("[SpVoxelPreprocessor] Using spconv 1.x: VoxelGenerator")
            return
        except Exception:
            pass

        # ---------- spconv 2.x : pytorch utils ----------
        try:
            from spconv.pytorch.utils import PointToVoxel
            self.spconv_mode = "spconv2_pt"
            self.voxel_generator = PointToVoxel(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                num_point_features=self.num_point_features,
                max_num_voxels=self.max_voxels,
                max_num_points_per_voxel=self.max_points_per_voxel,
                device=torch.device("cpu")
            )
            print("[SpVoxelPreprocessor] Using spconv 2.x: PointToVoxel")
            return
        except Exception:
            pass

        # ---------- spconv 2.x : low-level cpu voxelizer ----------
        try:
            from spconv.utils import Point2VoxelCPU3d
            import cumm.tensorview as tv
            self.spconv_mode = "spconv2_cpu"
            self.tv = tv
            self.voxel_generator = Point2VoxelCPU3d(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=self.max_points_per_voxel,
                max_num_voxels=self.max_voxels
            )
            print("[SpVoxelPreprocessor] Using spconv 2.x: Point2VoxelCPU3d")
            return
        except Exception as e:
            raise ImportError(
                "No supported voxel generator found. "
                "Need spconv 1.x VoxelGenerator/VoxelGeneratorV2 "
                "or spconv 2.x PointToVoxel / Point2VoxelCPU3d."
            ) from e

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.cpu().numpy()
        if hasattr(x, "numpy"):
            return x.numpy()
        raise TypeError(f"Cannot convert type {type(x)} to numpy")

    def preprocess(self, pcd_np):
        pcd_np = np.asarray(pcd_np, dtype=np.float32)
        data_dict = {}

        if self.spconv_mode == "spconv1":
            voxel_output = self.voxel_generator.generate(pcd_np)
            if isinstance(voxel_output, dict):
                voxels = voxel_output['voxels']
                coordinates = voxel_output['coordinates']
                num_points = voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output

        elif self.spconv_mode == "spconv2_pt":
            points_th = torch.from_numpy(pcd_np)
            voxels, coordinates, num_points = self.voxel_generator(points_th)
            voxels = self._to_numpy(voxels)
            coordinates = self._to_numpy(coordinates)
            num_points = self._to_numpy(num_points)

        elif self.spconv_mode == "spconv2_cpu":
            points_tv = self.tv.from_numpy(pcd_np)
            voxels_tv, coordinates_tv, num_points_tv = \
                self.voxel_generator.point_to_voxel(points_tv)
            voxels = self._to_numpy(voxels_tv)
            coordinates = self._to_numpy(coordinates_tv)
            num_points = self._to_numpy(num_points_tv)

        else:
            raise RuntimeError(f"Unknown spconv_mode: {self.spconv_mode}")

        data_dict['voxel_features'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def collate_batch(self, batch):
        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        voxel_features = []
        voxel_num_points = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            voxel_num_points.append(batch[i]['voxel_num_points'])
            coords = batch[i]['voxel_coords']
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i)
            )

        voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points))
        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points
        }

    @staticmethod
    def collate_batch_dict(batch: dict):
        voxel_features = torch.from_numpy(np.concatenate(batch['voxel_features']))
        voxel_num_points = torch.from_numpy(np.concatenate(batch['voxel_num_points']))
        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i)
            )
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points
        }