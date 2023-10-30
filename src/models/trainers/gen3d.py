import math
import random
import sys

import point_cloud_utils as pcu
import torch as th
import torch.nn as nn
from easydict import EasyDict
from einops import rearrange
from tqdm.auto import tqdm

from src import trainer
from src.datasets.const import synset_to_taxonomy
from src.models.utils import ema
from src.utils import instantiate_from_config
from src.utils.vis import make_meshes_grid, sdfs_to_meshes_np


class GEN3dPreprocessor(trainer.BasePreprocessor):
    def __init__(self, device, do_augmentation, sdf_clip, mean, std, downsample=1):
        super().__init__(device)
        self.do_augmentation = do_augmentation
        self.sdf_clip = sdf_clip
        self.mean = mean
        self.std = std
        self.downsample = downsample

    @th.no_grad()
    def __call__(self, batch, augmentation=False) -> dict:
        s = EasyDict(log={})

        s.synset, s.model_id, s.im_y, s.c = batch
        s.im_y = s.im_y.to(self.device, non_blocking=True)
        s.c = s.c.to(self.device, non_blocking=True)
        s.n = len(s.im_y)

        # flip augmentation
        if self.do_augmentation and augmentation:
            outs_y = []
            for i in range(s.n):
                if random.random() < 0.5:
                    flip = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
                    outs_y.append(s.im_y[i].flip(dims=random.choice(flip)).contiguous())
                else:
                    outs_y.append(s.im_y[i])
            s.im_y = th.stack(outs_y)

        s.im_y = self.standardize(s.im_y)

        return s

    def standardize(self, x: th.Tensor):
        if self.sdf_clip == 0:
            x = x.sign()
        else:
            x = x.clamp(-self.sdf_clip, self.sdf_clip)
            x = (x - self.mean) / self.std

        if self.downsample > 1:
            d = self.downsample
            x = rearrange(x, "b d (r1 s1) (r2 s2) (r3 s3) -> b (d s1 s2 s3) r1 r2 r3", s1=d, s2=d, s3=d).contiguous()

        return x

    def destandardize(self, x: th.Tensor):
        if self.downsample > 1:
            d = self.downsample
            x = rearrange(x, "b (d s1 s2 s3) r1 r2 r3 -> b d (r1 s1) (r2 s2) (r3 s3)", s1=d, s2=d, s3=d).contiguous()

        if self.sdf_clip == 0:
            # x = x.sign()
            pass
        else:
            x = x * self.std + self.mean
            x = x.clamp(-self.sdf_clip, self.sdf_clip)
        return x


class GEN3dTrainer(trainer.BaseTrainer):
    def __init__(
        self,
        args,
        find_unused_parameters,
        mixed_precision,
        n_samples_per_class,
        sample_at_least_per_epochs,
        n_rows,
        use_ddim,
        ema_decay,
    ):
        super().__init__(
            args,
            n_samples_per_class=n_samples_per_class,
            find_unused_parameters=find_unused_parameters,
            mixed_precision=mixed_precision,
            sample_at_least_per_epochs=sample_at_least_per_epochs,
        )
        self.n_rows = n_rows
        self.use_ddim = use_ddim
        self.ema_decay = ema_decay

    def build_network(self):
        super().build_network()

        self.model_ema: nn.Module = instantiate_from_config(self.args.model).cuda().eval().requires_grad_(False)
        self.model_ema.load_state_dict(self.model.state_dict())

        self.ddpm_train: nn.Module = instantiate_from_config(self.args.ddpm.train).cuda()
        self.ddpm_valid: nn.Module = instantiate_from_config(self.args.ddpm.valid).cuda()

    def build_sample_idx(self):
        self.class_idx = list(range(self.dl_test.dataset.n_classes))
        m = math.ceil(len(self.class_idx) / self.world_size)
        self.class_idx_rank = self.class_idx[m * self.rank : m * (self.rank + 1)]
        self.n_samples = self.n_samples_per_class * len(self.class_idx)
        self.n_samples_rank = math.ceil(self.n_samples / self.world_size)

    def save(self, out_path):
        data = {
            "epoch": self.epoch,
            "best_loss": self.best,
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict(),
        }
        th.save(data, str(out_path))

    def on_train_batch_end(self, s):
        ema(self.model, self.model_ema, self.ema_decay)

    def step(self, s):
        self.input_shape = s.im_y.shape[1:]
        diffusion_fn = self.ddpm_train
        denoise_fn = lambda x_t, t: self.model_optim(x_t, t, c=s.c)
        s.log.loss = diffusion_fn(denoise_fn, s.im_y)

    def step_test(self, b, c, ema=False):
        c = th.full((b,), c, dtype=th.long, device=self.device)
        shape = (b, *self.input_shape)
        diffusion_fn = self.ddpm_valid.sample_ddim if self.use_ddim else self.ddpm_valid.sample
        model = self.model_ema if ema else self.model
        denoise_fn = lambda x_t, t: model(x_t, t, c=c)
        im_p = diffusion_fn(denoise_fn, shape)
        return im_p

    @th.no_grad()
    def sample(self):
        self.model_optim.eval()

        outdir = self.args.exp_path / "samples" / f"e{self.epoch:04d}"
        if self.rankzero:
            outdir.mkdir(parents=True, exist_ok=True)
        self.safe_barrier()

        n = self.n_samples
        m = self.n_samples_rank
        b = self.dl_train.batch_size * 2

        with tqdm(total=n, ncols=100, file=sys.stdout, desc="Sample", disable=not self.rankzero) as t:
            for c in self.class_idx_rank:
                synset = self.dl_test.dataset.cls_to_synset[c]
                taxonomy = synset_to_taxonomy[synset]

                ims, ims_ema = [], []
                for i in range(0, self.n_samples_per_class, b):
                    b_ = min(self.n_samples_per_class - i, b)
                    im_p = self.step_test(b_, c)
                    im_p = self.preprocessor.destandardize(im_p)
                    ims.append(im_p)
                    if self.epoch > 50:
                        im_p = self.step_test(b_, c, ema=True)
                        im_p = self.preprocessor.destandardize(im_p)
                        ims_ema.append(im_p)

                    t.update(min(t.total - t.n, b_ * self.world_size))
                ims = th.cat(ims)  # b 1 r r r
                v, f = sdfs_to_meshes_np(ims, safe=True)
                v, f = make_meshes_grid(v, f, 0, 1, 0.1, nrows=self.n_rows)
                path = outdir / f"{taxonomy}.obj"
                pcu.save_mesh_vf(str(path), v, f)

                if ims_ema:
                    ims = th.cat(ims_ema)
                    v, f = sdfs_to_meshes_np(ims, safe=True)
                    v, f = make_meshes_grid(v, f, 0, 1, 0.1, nrows=self.n_rows)
                    path = outdir / f"{taxonomy}-ema.obj"
                    pcu.save_mesh_vf(str(path), v, f)
