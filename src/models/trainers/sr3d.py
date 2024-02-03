import math
import random
import sys

import point_cloud_utils as pcu
import torch as th
import torch.nn.functional as F
from easydict import EasyDict
from einops import rearrange
from torchvision.utils import save_image
from tqdm.auto import tqdm

from src import trainer
from src.datasets.const import synset_to_taxonomy
from src.models.utils import ema
from src.utils import instantiate_from_config
from src.utils.algebra import gaussian_blur
from src.utils.folding2d import get_fold_unfold
from src.utils.vis import make_meshes_grid, sdfs_to_meshes_np


class SR3dPreprocessor(trainer.BasePreprocessor):
    def __init__(
        self,
        device,
        do_augmentation,
        sdf_clip,
        mean,
        std,
        patch_size=None,
        downsample=1,
        blur_augmentation=False,
        blur_sig=[0.1, 2.0],
        blur_kernel_size=9,
    ):
        super().__init__(device)
        self.do_augmentation = do_augmentation
        self.sdf_clip = sdf_clip
        self.mean = mean
        self.std = std
        self.patch_size = patch_size
        self.downsample = downsample
        self.blur_augmentation = blur_augmentation
        self.blur_sig = blur_sig
        self.blur_kernel_size = blur_kernel_size

    @th.no_grad()
    def __call__(self, batch, augmentation=False) -> dict:
        s = EasyDict(log={})

        s.synset, s.model_id, s.im_y, s.im_x, s.c = batch
        s.im_y = s.im_y.to(self.device, non_blocking=True)
        s.im_x = s.im_x.to(self.device, non_blocking=True)
        s.c = s.c.to(self.device, non_blocking=True)
        s.n = len(s.im_x)

        # blur_augmentation
        if self.blur_augmentation and augmentation:
            outs = []
            for i in range(s.n):
                if random.random() < 0.5:
                    sig = random.random() * (self.blur_sig[1] - self.blur_sig[0]) + self.blur_sig[0]
                    outs.append(gaussian_blur(s.im_x[i, None], sig, self.blur_kernel_size))
                else:
                    outs.append(s.im_x[i, None])
            s.im_x = th.cat(outs)

        # rescale for conditional input
        s.im_x = F.interpolate(s.im_x, s.im_y.shape[2:])

        # flip augmentation
        if self.do_augmentation and augmentation:
            outs_y, outs_x = [], []
            for i in range(s.n):
                if random.random() < 0.5:
                    flip = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
                    flip = random.choice(flip)
                    outs_y.append(s.im_y[i].flip(dims=flip).contiguous())
                    outs_x.append(s.im_x[i].flip(dims=flip).contiguous())
                else:
                    outs_y.append(s.im_y[i])
                    outs_x.append(s.im_x[i])
            s.im_y = th.stack(outs_y)
            s.im_x = th.stack(outs_x)

        # standardize or normalize
        s.im_y = self.standardize(s.im_y, 1)
        s.im_x = self.standardize(s.im_x, 0)

        # make it patched
        if self.patch_size is not None:
            p = self.patch_size
            t = [random.randint(0, s.im_y.size(-1) - p) for _ in range(3)]
            s.pim_y = s.im_y[..., t[0] : t[0] + p, t[1] : t[1] + p, t[2] : t[2] + p]
            s.pim_x = s.im_x[..., t[0] : t[0] + p, t[1] : t[1] + p, t[2] : t[2] + p]
        else:
            s.pim_y = s.im_y
            s.pim_x = s.im_x
        return s

    def standardize(self, x: th.Tensor, i: int):
        if self.sdf_clip[i] == 0:
            x = x.sign()
        else:
            x = x.clamp(-self.sdf_clip[i], self.sdf_clip[i])
            x = (x - self.mean[i]) / self.std[i]

        if self.downsample > 1:
            d = self.downsample
            x = rearrange(x, "b d (r1 s1) (r2 s2) (r3 s3) -> b (d s1 s2 s3) r1 r2 r3", s1=d, s2=d, s3=d).contiguous()

        return x

    def destandardize(self, x: th.Tensor, i: int):
        if self.downsample > 1:
            d = self.downsample
            x = rearrange(x, "b (d s1 s2 s3) r1 r2 r3 -> b d (r1 s1) (r2 s2) (r3 s3)", s1=d, s2=d, s3=d).contiguous()

        if self.sdf_clip[i] == 0:
            x = x.sign()
        else:
            x = x * self.std[i] + self.mean[i]
            x = x.clamp(-self.sdf_clip[i], self.sdf_clip[i])
        return x


class SR3dTrainer(trainer.BaseTrainer):
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
        test_batch_size=None,
        predict_residual=False,
        gaussian_conditional_augmentation=False,
    ):
        self.n_rows = n_rows
        self.use_ddim = use_ddim
        self.ema_decay = ema_decay
        self.test_batch_size = test_batch_size
        self.predict_residual = predict_residual
        self.gaussian_conditional_augmentation = gaussian_conditional_augmentation

        super().__init__(
            args,
            n_samples_per_class=n_samples_per_class,
            find_unused_parameters=find_unused_parameters,
            mixed_precision=mixed_precision,
            sample_at_least_per_epochs=sample_at_least_per_epochs,
        )

    def build_network(self):
        super().build_network()

        if self.ema_decay is not None:
            self.model_ema = instantiate_from_config(self.args.model).cuda().eval().requires_grad_(False)
            self.model_ema.load_state_dict(self.model.state_dict())

        self.ddpm_train = instantiate_from_config(self.args.ddpm.train).cuda()
        self.ddpm_valid = instantiate_from_config(self.args.ddpm.valid).cuda()

    def build_dataset(self):
        super().build_dataset()

        if self.test_batch_size is None:
            self.test_batch_size = max(self.dl_test.batch_size // 4, 1)

    def build_sample_idx(self):
        self.class_idx = list(range(self.dl_test.dataset.n_classes))
        m = math.ceil(len(self.class_idx) / self.world_size)
        self.class_idx_rank = self.class_idx[m * self.rank : m * (self.rank + 1)]
        self.n_samples = self.n_samples_per_class * len(self.class_idx)
        self.n_samples_rank = math.ceil(self.n_samples / self.world_size)

    def on_train_batch_end(self, s):
        if self.ema_decay is not None:
            ema(self.model, self.model_ema, self.ema_decay)

    def step(self, s):
        def denoise_fn(x_t, t):
            if self.gaussian_conditional_augmentation:
                z_t, t_z = self.ddpm_train.q_sample_z_s(s.pim_x)
            else:
                z_t = s.pim_x
            input = th.cat([z_t, x_t], dim=1)
            if self.gaussian_conditional_augmentation:
                out = self.model_optim(input, t, c=s.c, s=t_z)
            else:
                out = self.model_optim(input, t, c=s.c)
            if self.predict_residual:
                out = out + s.pim_x
            return out

        diffusion_fn = self.ddpm_train.forward
        s.log.loss = diffusion_fn(denoise_fn, s.pim_y)

    def step_test(self, s):
        def denoise_fn_wrapper(model):
            def denoise_fn(x_t, t):
                if self.gaussian_conditional_augmentation:
                    t_z = th.full((s.n,), self.ddpm_valid.max_s, dtype=th.long, device=s.im_x.device)
                    z_t, _ = self.ddpm_valid.q_sample_z_s(s.im_x, t_z)
                else:
                    z_t = s.im_x
                input = th.cat([z_t, x_t], dim=1)
                if self.gaussian_conditional_augmentation:
                    out = model(input, t, c=s.c, s=t_z)
                else:
                    out = model(input, t, c=s.c)
                if self.predict_residual:
                    out = out + s.pim_x
                return out

            return denoise_fn

        diffusion_fn = self.ddpm_valid.sample_ddim if self.use_ddim else self.ddpm_valid.sample
        s.im_p = diffusion_fn(denoise_fn_wrapper(self.model), s.im_y.shape)

        s.im_x = self.preprocessor.destandardize(s.im_x, 0)
        s.im_y = self.preprocessor.destandardize(s.im_y, 1)
        s.im_p = self.preprocessor.destandardize(s.im_p, 1)

        if self.ema_decay is not None:
            s.im_p_ema = None
            if (self.epoch >= 50) or self.args.debug:
                s.im_p_ema = diffusion_fn(denoise_fn_wrapper(self.model_ema), s.im_y.shape)
                s.im_p_ema = self.preprocessor.destandardize(s.im_p_ema, 1)

    @th.no_grad()
    def sample(self):
        self.model_optim.eval()

        outdir = self.args.exp_path / "samples" / f"e{self.epoch:04d}"
        if self.rankzero:
            outdir.mkdir(parents=True, exist_ok=True)
        self.safe_barrier()

        n = self.n_samples
        m = self.n_samples_rank
        b = self.test_batch_size

        ims = []
        with tqdm(total=n, ncols=100, file=sys.stdout, desc="Sample", disable=not self.rankzero) as t:
            for c in self.class_idx_rank:
                synset = self.dl_test.dataset.cls_to_synset[c]
                taxonomy = synset_to_taxonomy[synset]
                sample_idx = self.dl_test.dataset.cate_indices[synset][: self.n_samples_per_class]
                jar = [self.dl_test.dataset[j] for j in sample_idx]

                ims, ims_ema = [], []
                for i in range(0, self.n_samples_per_class, b):
                    b_ = min(self.n_samples_per_class - i, b)
                    batch = self.dl_test.collate_fn(jar[i : i + b_])
                    s = self.preprocessor(batch)
                    self.step_test(s)
                    im = th.stack([s.im_x, s.im_y, s.im_p], dim=1).flatten(0, 1)
                    ims.append(im)

                    if self.ema_decay is not None and s.im_p_ema is not None:
                        im = th.stack([s.im_x, s.im_y, s.im_p_ema], dim=1).flatten(0, 1)
                        ims_ema.append(im)

                    t.update(min(t.total - t.n, b_ * self.world_size))

                ims = th.cat(ims)  # (b 3) 1 r r r
                v, f = sdfs_to_meshes_np(ims, safe=True)
                v, f = make_meshes_grid(v, f, 0, 1, 0.1, nrows=self.n_rows)
                path = outdir / f"e{self.epoch:04d}-{synset}-{taxonomy}.obj"
                pcu.save_mesh_vf(str(path), v, f)
                if self.ema_decay is not None and ims_ema:
                    ims = th.cat(ims_ema)  # (b 3) 1 r r r
                    v, f = sdfs_to_meshes_np(ims, safe=True)
                    v, f = make_meshes_grid(v, f, 0, 1, 0.1, nrows=self.n_rows)
                    path = outdir / f"e{self.epoch:04d}-{synset}-{taxonomy}-ema.obj"
                    pcu.save_mesh_vf(str(path), v, f)
