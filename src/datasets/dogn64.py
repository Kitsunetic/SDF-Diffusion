import math
import random
from collections import defaultdict

import h5py
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy import ndimage
from skimage import measure
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src import logging
from src.datasets.const import cls_to_synset, synset_to_cls, synset_to_taxonomy, taxonomy_to_synset
from src.utils import instantiate_from_config


class DOGN64SDF(Dataset):
    def __init__(self, datafile, cates, split) -> None:
        super().__init__()

        self.datafile = datafile
        self.cates = cates if cates == "all" else [taxonomy_to_synset[cate] for cate in cates.split("|")]

        self.files = []
        self.counter = defaultdict(int)
        self.cate_indices = defaultdict(list)
        i = 0
        with open("src/datasets/DOGN.txt") as f:
            for line in f.readlines():
                file, cls, sp = line.strip().split()
                synset, model_id = file.split("/")
                if sp == split and (self.cates == "all" or synset in self.cates):
                    self.files.append((synset, model_id))
                    self.counter[synset] += 1
                    self.cate_indices[synset].append(i)
                    i += 1

        self.synset_to_cls = synset_to_cls
        self.cls_to_synset = cls_to_synset
        if self.cates != "all":
            temp = [(synset_to_cls[cate], cate) for cate in self.cates]
            self.synset_to_cls = {synset: i for i, (cls, synset) in enumerate(temp)}
            self.cls_to_synset = {i: synset for i, (cls, synset) in enumerate(temp)}

        self.n_classes = len(self.synset_to_cls)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        synset, model_id = self.files[idx]

        # load sdf
        with h5py.File(self.datafile) as f:
            g = f[synset][model_id]
            sdf_y = g["sdf"][:]
            sdf_y = th.from_numpy(sdf_y)[None]

            # sdf_x = None
            # if "psdf" in g:
            #     sdf_x = g["psdf"][:]

        cls = self.synset_to_cls[synset]
        cls = th.tensor(cls, dtype=th.long)

        # return synset, model_id, sdf_y, sdf_x, cls
        return synset, model_id, sdf_y, cls

    def get_sample_idx(self, n_samples_per_cates):
        n = self.n_classes * n_samples_per_cates

        sample_idx = []
        for synset in self.synset_to_cls:
            sample_idx += self.cate_indices[synset][:n]

        return sample_idx


class DOGN64SDF_Test(DOGN64SDF):
    def __getitem__(self, idx, pidx=0):
        synset, model_id = self.files[idx]

        # load sdf
        with h5py.File(self.datafile) as f:
            g = f[synset][model_id]  # 이건 그냥 앞에 1dim 붙어있음, unsqueeze 안해도 됨
            sdf_y = g["sdf"][:]
            sdf_y = th.from_numpy(sdf_y)
            sdf_x = g["psdf"][pidx]
            sdf_x = th.from_numpy(sdf_x)

        cls = self.synset_to_cls[synset]
        cls = th.tensor(cls, dtype=th.long)

        return synset, model_id, sdf_y, sdf_x, cls


class DOGN64PTS(DOGN64SDF):
    def __init__(self, datafile, cates, split, n_pts=None, pts_dataset=False) -> None:
        super().__init__(datafile, cates, split)
        self.n_pts = n_pts
        self.pts_dataset = pts_dataset

    def __getitem__(self, idx):
        synset, model_id = self.files[idx]

        # load pts
        with h5py.File(self.datafile) as f:
            if not self.pts_dataset:
                pts = f[synset][model_id]["pts"][:]
            else:
                pts = f[synset][model_id][:]
        pts = th.from_numpy(pts)

        if self.n_pts is not None:
            pts = pts[th.randperm(pts.size(0))[: self.n_pts]]

        cls = self.synset_to_cls[synset]
        cls = th.tensor(cls, dtype=th.long)

        return synset, model_id, pts, cls


class DOGN64SDFPTS(DOGN64SDF):
    def __init__(self, datafile, cates, split, n_pts: int) -> None:
        super().__init__(datafile, cates, split)
        self.n_pts = n_pts

    def __getitem__(self, idx):
        synset, model_id = self.files[idx]

        # load sdf
        with h5py.File(self.datafile) as f:
            g = f[synset][model_id]
            sdf_y = g["sdf"][:]
            sdf_y = th.from_numpy(sdf_y)[None]

            pts = f[synset][model_id]["pts"][:]
            pts = th.from_numpy(pts)
            pts = pts[th.randperm(pts.size(0))[: self.n_pts]]

        cls = self.synset_to_cls[synset]
        cls = th.tensor(cls, dtype=th.long)

        # sdf_y: (R, R, R)
        # pts: (B, N, 3)
        return synset, model_id, sdf_y, pts, cls


def random_range(v_min, v_max):
    return random.random() * (v_max - v_min) + v_min


def rotate_voxel(voxel, angles):
    if angles[0] != 0:
        voxel = ndimage.rotate(voxel, angles[0], axes=(2, 1), reshape=False, mode="nearest")
    if angles[1] != 0:
        voxel = ndimage.rotate(voxel, -angles[1], axes=(2, 0), reshape=False, mode="nearest")
    if angles[2] != 0:
        voxel = ndimage.rotate(voxel, angles[2], axes=(1, 0), reshape=False, mode="nearest")
    return voxel


def zoom_voxel(voxel, scale):
    if isinstance(scale, (int, float)):
        scale = (scale,) * 3

    res = voxel.shape[0]
    out = ndimage.zoom(voxel, scale, order=1, mode="nearest")

    pads, crops = [], []
    for i in range(3):
        if out.shape[i] < voxel.shape[i]:
            diff = (voxel.shape[i] - out.shape[i]) / 2
            pad_l, pad_r = math.ceil(diff), math.floor(diff)
            pads.append((pad_l, pad_r))
            crops.append((0, res))

        elif out.shape[i] > voxel.shape[i]:
            diff = (out.shape[i] - voxel.shape[i]) / 2
            pads.append((0, 0))
            crop_l, crop_r = math.ceil(diff), math.floor(diff)
            crops.append((crop_l, res - crop_r))

        else:
            pads.append((0, 0))
            crops.append((0, res))

    out = out[crops[0][0] : crops[0][1], crops[1][0] : crops[1][1], crops[2][0] : crops[2][1]]
    out = np.pad(out, pads, mode="edge")

    return out


def shift_voxel(voxel, shift):
    voxel = ndimage.shift(voxel, shift, order=0, mode="nearest")
    return voxel


def voxel_augmentation(voxel, s_range, r_range, t_range):
    s = [random_range(s_range[0], s_range[1]) for _ in range(3)]
    r = [random_range(r_range[0], r_range[1]) for _ in range(3)]
    t = [random_range(t_range[0], t_range[1]) for _ in range(3)]
    voxel = zoom_voxel(voxel, s)
    voxel = rotate_voxel(voxel, r)
    voxel = shift_voxel(voxel, t)
    return voxel


class DOGN64SDFPTS_Augmentation(DOGN64SDFPTS):
    def __init__(self, datafile, cates, split, n_pts: int, p, rotation, scale, translation) -> None:
        super().__init__(datafile, cates, split, n_pts)
        self.p = p
        self.rotation = rotation
        self.scale = scale
        self.translation = translation

    def __getitem__(self, idx):
        synset, model_id = self.files[idx]
        cls = self.synset_to_cls[synset]
        cls = th.tensor(cls, dtype=th.long)

        # load sdf
        with h5py.File(self.datafile) as f:
            g = f[synset][model_id]
            sdf_y = g["sdf"][:]  # (R, R, R)

            if random.random() >= self.p:
                # no augmentation
                pts = f[synset][model_id]["pts"][:]
                pts = th.from_numpy(pts)
                pts = pts[th.randperm(pts.size(0))[: self.n_pts]].contiguous()

            else:
                sdf_y = voxel_augmentation(sdf_y, self.scale, self.rotation, self.translation)

                verts, faces, normals, values = measure.marching_cubes(sdf_y, level=0)
                verts = verts / sdf_y.shape[-1] * 2 - 1
                verts = th.from_numpy(verts.astype(np.float32))
                faces = th.from_numpy(faces.astype(np.int64))
                normals = th.from_numpy(normals.astype(np.float32))
                mesh = Meshes([verts], [faces], verts_normals=[normals])
                pts = sample_points_from_meshes(mesh, self.n_pts)[0]

            sdf_y = th.from_numpy(sdf_y)[None]

        return synset, model_id, sdf_y, pts, cls


def build_dataloaders(ddp, ds_opt, dl_kwargs, ds_opt_test=None):
    if dist.is_initialized():
        world_size = dist.get_world_size()
        dl_kwargs.batch_size = max(1, dl_kwargs.batch_size // world_size)
        dl_kwargs.num_workers = min(dl_kwargs.num_workers, max(1, dl_kwargs.num_workers // world_size))

    dss = [None, None, None]
    dss[0] = instantiate_from_config(ds_opt, split="train")
    dss[1] = instantiate_from_config(ds_opt, split="val")
    if ds_opt_test is None:
        ds_opt_test = ds_opt
    dss[2] = instantiate_from_config(ds_opt_test, split="test")

    log = logging.getLogger()
    log.info("Dataset Loaded:")
    for synset in dss[0].counter.keys():
        msg = f"    {synset} {synset_to_taxonomy[synset]:20}"
        msg += f" {dss[0].counter[synset]:5} {dss[1].counter[synset]:5} {dss[2].counter[synset]:5}"
        log.info(msg)

    tff = [True, False, False]
    if ddp:
        samplers = [DistributedSampler(ds, shuffle=t) for ds, t in zip(dss, tff)]
        dls = [DataLoader(ds, **dl_kwargs, sampler=sampler) for ds, sampler in zip(dss, samplers)]
    else:
        dls = [DataLoader(ds, **dl_kwargs, shuffle=t) for ds, t in zip(dss, tff)]

    return dls


def __test__():
    opt = """
    target: src.datasets.dogn64.build_dataloaders
    params:
        ds_opt:
            # target: src.datasets.dogn64.DOGN64SDF
            target: src.datasets.dogn64.DOGN64SDFPTS_Augmentation
            params:
                datafile: /dev/shm/jh/data/sdf.res32.level0.0500.PC15000.pad0.20.hdf5
                cates: all
                n_pts: 2048
                p: 0.5
                rotation: [0, 360]
                scale: [0.8, 1.0]
                translation: [-1, 1]
        dl_kwargs:
            batch_size: 4
            num_workers: 0
            pin_memory: no
            persistent_workers: no
    """
    import yaml

    from src.utils import instantiate_from_config

    opt = yaml.safe_load(opt)
    dls = instantiate_from_config(opt, False)
    for synset, model_id, sdf_y, pts, cls in dls[0]:
        break

    print(synset, model_id, sdf_y.shape, pts.shape, cls)
    print(pts.min(), pts.max())
    """
    [22:09:13 14:03:49  INFO] Dataset Loaded:
    [22:09:13 14:03:49  INFO]     02691156 airplane              2832   404   809
    [22:09:13 14:03:49  INFO]     02828884 bench                 1272   181   363
    [22:09:13 14:03:49  INFO]     02933112 cabinet               1101   157   281
    [22:09:13 14:03:49  INFO]     02958343 car                   4911   749  1499
    [22:09:13 14:03:49  INFO]     03001627 chair                 4746   677  1355
    [22:09:13 14:03:49  INFO]     03211117 display                767   109   219
    [22:09:13 14:03:49  INFO]     03636649 lamp                  1624   231   463
    [22:09:13 14:03:49  INFO]     03691459 loudspeaker           1134   161   323
    [22:09:13 14:03:49  INFO]     04090263 rifle                 1661   237   474
    [22:09:13 14:03:49  INFO]     04256520 sofa                  2222   317   634
    [22:09:13 14:03:49  INFO]     04379243 table                 5958   850  1701
    [22:09:13 14:03:49  INFO]     04401088 telephone              737   105   210
    [22:09:13 14:03:49  INFO]     04530566 vessel                1359   193   387
    ('04379243', '04379243', '04256520', '03636649') 
    ('1834fac2f46a26f91933ffef19678834', '57e3a5f82b410e24febad4f49b26ec52', 
    '199085218ed6b8f5f33e46f65e635a84', '55b002ebe262df5cba0a7d54f5c0d947') 
    torch.Size([4, 1, 64, 64, 64]) or torch.Size([4, 15000, 3])
    tensor([ 8,  7,  2, 10])
    tensor(-0.8489) tensor(0.8500)
    """


if __name__ == "__main__":
    __test__()
