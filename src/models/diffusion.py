import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.utils.indexing import unsqueeze_as


def rand_uniform(a, b, shape, device="cpu"):
    return (b - a) * th.rand(shape, dtype=th.float, device=device) + a


def identity(*args):
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        return args


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "quad":
        betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = th.arange(n_timestep + 1, dtype=th.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = th.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f"Selected timesteps for ddim sampler: {steps_out}")
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f"Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}")
        print(
            f"For the chosen value of eta, which is {eta}, "
            f"this results in the following sigma_t schedule for ddim sampler {sigmas}"
        )
    return sigmas, alphas, alphas_prev


# gaussian diffusion trainer class


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(self, loss_type="l1", model_mean_type="eps", schedule_kwargs=None):
        super().__init__()
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.set_new_noise_schedule(schedule_kwargs, device="cpu")
        self.set_loss("cpu")

    def set_loss(self, device):
        if self.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="mean").to(device)
        elif self.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="mean").to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(th.tensor, dtype=th.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt["schedule"],
            n_timestep=schedule_opt["n_timestep"],
            linear_start=schedule_opt.get("linear_start", 1e-4),
            linear_end=schedule_opt.get("linear_end", 2e-2),
            cosine_s=schedule_opt.get("cosine_s", 8e-3),
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, th.Tensor) else betas
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        self.register_buffer("sqrt_alphas_cumprod_prev", to_torch(sqrt_alphas_cumprod_prev))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer(
            "posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))
        )

        if "ddim_S" in schedule_opt and "ddim_eta" in schedule_opt:
            self.set_ddim_schedule(schedule_opt["ddim_S"], schedule_opt["ddim_eta"])

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            unsqueeze_as(self.sqrt_recip_alphas_cumprod[t], x_t) * x_t
            - unsqueeze_as(self.sqrt_recipm1_alphas_cumprod[t], noise) * noise
        )

    def predict_noise_from_start(self, x_t, t, x_0):
        # x_0 = A x_t - B e
        # e = A/B x_t - 1/B x_0
        recip = 1 / unsqueeze_as(self.sqrt_recipm1_alphas_cumprod[t], x_t)
        return (unsqueeze_as(self.sqrt_recip_alphas_cumprod[t], x_t) * x_t - x_0) * recip

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            unsqueeze_as(self.posterior_mean_coef1[t], x_start) * x_start
            + unsqueeze_as(self.posterior_mean_coef2[t], x_t) * x_t
        )
        posterior_log_variance_clipped = unsqueeze_as(self.posterior_log_variance_clipped[t], x_t)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, x, t, clip_denoised: bool, denoise_kwargs={}, post_fn=identity):
        # noise_level = self.sqrt_alphas_cumprod_prev[t + 1].repeat(b, 1)
        # noise_level = th.tensor([self.sqrt_alphas_cumprod_prev[t + 1]], dtype=th.float, device=x.device).repeat(b, 1)
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1]
        noise_pred = post_fn(denoise_fn(x, noise_level, **denoise_kwargs))
        if self.model_mean_type == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        else:
            x_recon = noise_pred

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @th.no_grad()
    def p_sample(self, denoise_fn, x, t, clip_denoised=True, denoise_kwargs={}, post_fn=identity):
        model_mean, model_log_variance = self.p_mean_variance(
            denoise_fn, x, t, clip_denoised=clip_denoised, denoise_kwargs=denoise_kwargs, post_fn=post_fn
        )
        # noise = th.randn_like(x) if t > 0 else th.zeros_like(x)
        noise = th.randn_like(x)
        noise[t == 0] = 0
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @th.no_grad()
    def sample(
        self,
        denoise_fn,
        shape,
        clip_denoised=True,
        denoise_kwargs={},
        post_fn=identity,
        return_intermediates=False,
        show_pbar=False,
        pbar_kwargs={},
    ):
        b = shape[0]
        rankzero = not dist.is_initialized() or dist.get_rank() == 0
        tqdm_kwargs = dict(
            desc="Sample DDPM",
            total=self.num_timesteps,
            ncols=128,
            disable=not (show_pbar and rankzero),
        )
        tqdm_kwargs.update(pbar_kwargs)
        pbar = tqdm(reversed(range(0, self.num_timesteps)), **tqdm_kwargs)

        device = self.betas.device
        sample_inter = 1 | (self.num_timesteps // 10)

        img = th.randn(shape, device=device)
        ret_img = [img]
        for i in pbar:
            t = img.new_full((b,), i, dtype=th.long)
            img = self.p_sample(denoise_fn, img, t, clip_denoised=clip_denoised, denoise_kwargs=denoise_kwargs, post_fn=post_fn)
            if i % sample_inter == 0:
                ret_img += [img]

        if return_intermediates:
            return ret_img
        else:
            return ret_img[-1]

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: th.randn_like(x_start))

        # random gamma
        return (
            unsqueeze_as(continuous_sqrt_alpha_cumprod, x_start) * x_start
            + unsqueeze_as(1 - continuous_sqrt_alpha_cumprod**2, noise).sqrt() * noise
        )

    def p_losses(self, denoise_fn, x_0, noise=None, denoise_kwargs={}, post_fn=identity):
        b = x_0.size(0)
        dev = x_0.device

        t = th.randint(1, self.num_timesteps + 1, (b,), device=dev)
        v1 = self.sqrt_alphas_cumprod_prev[t - 1]
        v2 = self.sqrt_alphas_cumprod_prev[t]
        continuous_sqrt_alpha_cumprod = (v2 - v1) * th.rand(b, device=dev) + v1  # b

        noise = default(noise, lambda: th.randn_like(x_0))
        x_noisy = self.q_sample(x_0, continuous_sqrt_alpha_cumprod, noise)
        x_recon = post_fn(denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod, **denoise_kwargs))

        if self.model_mean_type == "eps":
            loss = self.loss_func(noise, x_recon)
        else:
            loss = self.loss_func(x_0, x_recon)
        return loss

    def forward(self, denoise_fn, x, denoise_kwargs={}, post_fn=identity, *args, **kwargs):
        return self.p_losses(denoise_fn, x, denoise_kwargs=denoise_kwargs, post_fn=post_fn, *args, **kwargs)

    def set_ddim_schedule(self, S, eta):
        to_torch = partial(th.tensor, dtype=th.float32, device="cpu")

        # make ddim schedule
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method="uniform",
            num_ddim_timesteps=S,
            num_ddpm_timesteps=self.num_timesteps,
            verbose=False,
        )
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=self.alphas_cumprod.cpu().numpy(), ddim_timesteps=self.ddim_timesteps, eta=eta, verbose=False
        )
        ddim_sqrt_one_minus_alphas = np.sqrt(1.0 - ddim_alphas)

        ddim_sigmas = to_torch(ddim_sigmas)
        ddim_alphas = to_torch(ddim_alphas)
        ddim_alphas_prev = to_torch(ddim_alphas_prev)
        ddim_sqrt_one_minus_alphas = to_torch(ddim_sqrt_one_minus_alphas)

        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", ddim_sqrt_one_minus_alphas)

    @th.no_grad()
    def sample_ddim(
        self,
        denoise_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoise_kwargs={},
        post_fn=identity,
        return_intermediates=False,
        log_every_t=5,
        show_pbar=False,
        pbar_kwargs={},
    ):
        assert hasattr(self, "ddim_timesteps"), "ddim parameters are not initialized"
        rankzero = not dist.is_initialized() or dist.get_rank() == 0
        dev = self.betas.device
        b = shape[0]
        timesteps = self.ddim_timesteps

        assert noise is None or noise.shape == shape
        x = th.randn(shape, device=dev) if noise is None else noise
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        tqdm_kwargs = dict(
            total=total_steps,
            desc="Sample DDIM",
            ncols=128,
            disable=not (show_pbar and rankzero),
        )
        tqdm_kwargs.update(pbar_kwargs)
        pbar = tqdm(time_range, **tqdm_kwargs)

        intermediates = [x]
        for i, step in enumerate(pbar):
            index = total_steps - i - 1
            ts = th.full((b,), step, device=dev, dtype=th.long)
            noise_level = self.sqrt_alphas_cumprod_prev[ts]

            e_t = post_fn(denoise_fn(x, noise_level, **denoise_kwargs))
            if self.model_mean_type == "x_0":
                e_t = self.predict_noise_from_start(x, ts, e_t)

            a_t = unsqueeze_as(th.full((b,), self.ddim_alphas[index], device=dev), x)
            a_prev = unsqueeze_as(th.full((b,), self.ddim_alphas_prev[index], device=dev), x)
            sigma_t = unsqueeze_as(th.full((b,), self.ddim_sigmas[index], device=dev), x)
            sqrt_one_minus_at = unsqueeze_as(th.full((b,), self.ddim_sqrt_one_minus_alphas[index], device=dev), x)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if clip_denoised:
                pred_x0.clamp_(-1.0, 1.0)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates.append(pred_x0)

            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * th.randn_like(x)
            x = a_prev.sqrt() * pred_x0 + dir_xt + noise

        if return_intermediates:
            return intermediates
        else:
            return intermediates[-1]
