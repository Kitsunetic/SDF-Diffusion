from collections import defaultdict

import h5py
import torch as th
from src import logging
from src.datasets.const import cls_to_synset, synset_to_cls, synset_to_taxonomy, taxonomy_to_synset
from src.utils import instantiate_from_config
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class DatasetSR(Dataset):
    def __init__(self, datafile_lr, datafile_hr, cates, split) -> None:
        super().__init__()

        self.datafile_lr = datafile_lr
        self.datafile_hr = datafile_hr
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

        with h5py.File(self.datafile_hr) as f:
            sdf_y = f[synset][model_id]["sdf"][:]
            sdf_y = th.from_numpy(sdf_y)[None]  # 1 r r r
        with h5py.File(self.datafile_lr) as f:
            sdf_x = f[synset][model_id]["sdf"][:]
            sdf_x = th.from_numpy(sdf_x)[None]  # 1 r r r

        # find class
        cls = synset_to_cls[synset]
        cls = th.tensor(cls, dtype=th.long)

        return synset, model_id, sdf_y, sdf_x, cls

    def get_sample_idx(self, n_samples_per_cates):
        n = self.n_classes * n_samples_per_cates

        sample_idx = []
        for synset in self.synset_to_cls:
            sample_idx += self.cate_indices[synset][:n]

        return sample_idx


def build_dataloaders(ddp, ds_kwargs, dl_kwargs):
    splits = ("train", "val", "test")
    dss = [DatasetSR(split=split, **ds_kwargs) for split in splits]

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
target: src.datasets.dogn_sr.build_dataloaders
params:
    ds_kwargs:
        datafile_lr: /dev/shm/jh/data/sdf.res32.level0.0500.PC15000.pad0.20.hdf5
        datafile_hr: /dev/shm/jh/data/sdf.res64.level0.0313.PC15000.pad0.20.hdf5
        cates: all
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
    for synset, model_id, sdf_y, sdf_x, cls in dls[0]:
        break

    print(synset, model_id, sdf_y.shape, sdf_x.shape, cls)
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
('03001627', '04379243', '03211117', '04379243')
('4a0b61d33846824ab1f04c301b6ccc90', '441e0682fa5eea135c49e0733c4459d0',
 '2c4bcdc965d6de30cfe893744630a6b9', '1ab95754a8af2257ad75d368738e0b47')
torch.Size([4, 1, 64, 64, 64]) torch.Size([4, 1, 32, 32, 32]) tensor([0, 0, 4, 1])
    """


if __name__ == "__main__":
    __test__()
