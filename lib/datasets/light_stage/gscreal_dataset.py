import torch.utils.data as data
import numpy as np
import h5py
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils

import ctypes
import multiprocessing as mp


class Dataset(data.Dataset):

    def __init__(self, data_root, human, split, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.h5_path = os.path.join(data_root, f'{human}_{split}.h5')
        self.vertices_path = os.path.join(data_root, 'vertices', f'{human}_{split}.npy')
        self.nrays = cfg.N_rand
        self.split = split

        shared_array_base = mp.Array(ctypes.c_int, 1)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        self.iter = shared_array.reshape(-1)
        self.dataset = None
        self.init_meta()
    
    def init_dataset(self):
        self.dataset = h5py.File(self.h5_path, 'r')
    
    def init_meta(self):
        dataset = h5py.File(self.h5_path, 'r')
        self._len = len(dataset['imgs'])
        self.kp_inds = dataset['kp_idxs'][:]
        self.cam_inds = dataset['img_pose_indices'][:]
        self.Rh = dataset['Rhs'][:].astype(np.float32)
        self.Th = dataset['Ths'][:].astype(np.float32)
        self.vertices = np.load(self.vertices_path, allow_pickle=True)
        self.HW = dataset['img_shape'][1:3]
        self.focals = dataset['focals'][:]
        self.centers = dataset['centers'][:]
        self.c2ws = dataset['c2ws'][:]
        dataset.close()

        self.K = np.eye(3)[None].repeat(len(self.focals), 0)
        self.K[:, 0, 0] = self.focals[:, 0]
        self.K[:, 1, 1] = self.focals[:, 1]

        self.K[:, 0, 2] = self.centers[:, 0]
        self.K[:, 1, 2] = self.centers[:, 1]
        self.K = self.K.astype(np.float32)

        swap_rot = np.eye(4)[None].astype(np.float32)
        swap_rot[:, 1, 1] = -1.
        swap_rot[:, 2, 2] = -1.
        self.RT = np.linalg.inv(self.c2ws @ swap_rot).astype(np.float32)
        self.ims = np.zeros((self._len)).astype(np.uint8)
        self.num_cams = len(self.K)
    
    def prepare_input(self, i):

        # read xyz, normal, color from the ply file
        xyz = self.vertices[i].astype(np.float32).copy()
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        Rh = self.Rh[i].copy()
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = self.Th[i].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return coord, out_sh, can_bounds, bounds, Rh, Th

    def pyramid_img_ratio(self):
        ratio_list = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
        self.iter += 1

        step = 5000
        if self.iter < 10*step:
            return ratio_list[(self.iter.item() // step)]
        else:
            return 1.
    
    def __getitem__(self, index):

        if self.dataset is None:
            self.init_dataset()

        if cfg.pyramid:
            cfg.ratio = self.pyramid_img_ratio()

        img = self.dataset['imgs'][index].reshape(*self.HW, 3) / 255.
        img = cv2.resize(img, (cfg.W, cfg.H))

        # TODO: deal with this
        msk = self.dataset['sampling_masks'][index].reshape(*self.HW, 1).astype(np.uint8)

        cam_ind = self.cam_inds[index].copy()
        K = self.K[cam_ind].copy()

        RT = self.RT[cam_ind].copy()
        R = RT[:3, :3]
        T = RT[:3, 3:] #/ 1000.
        #T = np.array(self.cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        K[:2] = K[:2] * cfg.ratio

        frame_index = i = self.kp_inds[index]
        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(i)

        if cfg.cache_shape and self.split == 'train':
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box, mmsk = if_nerf_dutils.sample_ray_h36m(
                img, msk, K, R, T, can_bounds, self.nrays, 'test')
        else:
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box, mmsk = if_nerf_dutils.sample_ray_h36m(
                img, msk, K, R, T, can_bounds, self.nrays, self.split)

        ret = {
            'coord': coord,
            'out_sh': out_sh,
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
            'mmsk': mmsk,
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = frame_index
        if cfg.test_novel_pose:
            latent_index = cfg.num_train_frame - 1

        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return self._len