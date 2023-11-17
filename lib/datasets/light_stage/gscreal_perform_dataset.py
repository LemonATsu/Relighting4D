import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from lib.utils import snapshot_data_utils as snapshot_dutils
from lib.utils import render_utils
import h5py


class Dataset(data.Dataset):

    def __init__(self, data_root, human, split, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.h5_path = os.path.join(data_root, f'{human}_{split}.h5')
        self.vertices_path = os.path.join(data_root, 'vertices', f'{human}_{split}.npy')
        self.nrays = cfg.N_rand
        self.split = split
        self.eval_cams = np.array([0, 1])

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
        self.bgs = dataset['bkgds'][:]
        self.bg_inds = dataset['bkgd_idxs'][:]
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

        self.index = np.array([i for i in range(self._len) if self.cam_inds[i] in self.eval_cams])
        self._len = len(self.index)

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
    
    def get_masks(self, i):
        kp_idx = self.kp_inds[i]
        num_novel_pose = cfg.num_novel_pose_frame
        msks = [
            self.dataset[f'masks'][kp_idx + c * num_novel_pose].reshape(*self.HW, 1).astype(np.uint8)
            for c in self.eval_cams
        ]
        msk = self.dataset[f'masks'][i].reshape(*self.HW).astype(np.uint8)
        return msks, msk

    def __getitem__(self, index):

        if self.dataset is None:
            self.init_dataset()
        index = self.index[index]

        img = self.dataset['imgs'][index].reshape(*self.HW, 3) / 255.
        img = cv2.resize(img, (cfg.W, cfg.H))
        bg = self.dataset['bkgds'][self.bg_inds[index]].reshape(*self.HW, 3) / 255.

        # TODO: deal with this
        # msk = self.dataset['masks'][index].reshape(*self.HW, 1).astype(np.uint8)
        msks, msk = self.get_masks(index)

        cam_ind = self.cam_inds[index].copy()
        K = self.K[cam_ind].copy()

        RT = self.RT[cam_ind].copy()
        R = RT[:3, :3]
        T = RT[:3, 3:] #/ 1000.
        #T = np.array(self.cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msks = [cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) for m in msks]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        """ 
        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        """
        K[:2] = K[:2] * cfg.ratio

        frame_index = i = self.kp_inds[index]
        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(i)

        ray_o, ray_d, near, far, _, _, mask_at_box = render_utils.image_rays(
            RT, K, can_bounds)
        
        Ks = self.K.copy()[self.eval_cams]
        Ks[:, :2] = Ks[:, :2] * cfg.ratio
        RTs = self.RT.copy()[self.eval_cams]

        ret = {
            'coord': coord,
            'out_sh': out_sh,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'rgb': img[mask_at_box.reshape(H, W)],
            'mask_at_box': mask_at_box,
            'msk': msk,
            'RT': RTs,
            'Ks': Ks,
            'msks': np.array(msks),
            'msk': msk,
            'img': img,
            'bg': bg,
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
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
