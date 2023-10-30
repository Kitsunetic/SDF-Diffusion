"""
Roughly adapted from 
https://github.com/CompVis/latent-diffusion/blob/2b46bcb98c8e8fdb250cb8ff2e20874f3ccdd768/ldm/models/diffusion/ddpm.py

Edited for 3-dimensional folding/unfolding by Kitsunetic
    - Currently 3d folding/unfolding is not supported by PyTorch
    - unfoldNd (https://github.com/f-dangel/unfoldNd) is not memory-efficient because of mask based folding 
      so that it cannot be applied high-resolution (larger than 32x32x32) 3d data.
      
    - Currently this code is only targetting for only square-shaped input, (i.e. 32x64x32 is not available).
"""
from functools import reduce

import torch as th
import torch.nn as nn

__all__ = ["get_fold_unfold"]


def srange(n, k, s):
    i = 0
    while i + k <= n:
        yield i, i + k

        i += s


class Fold(nn.Module):
    def __init__(self, kernel_size, stride=1) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """
        - input:
            - x: b c k1 k2 k3 l
        """
        b, c, l = *x.shape[:2], x.size(-1)
        h = self.stride[0] * round(l ** (1 / 3) - 1) + self.kernel_size[0]
        w = self.stride[1] * round(l ** (1 / 3) - 1) + self.kernel_size[1]
        d = self.stride[2] * round(l ** (1 / 3) - 1) + self.kernel_size[2]

        out = x.new_zeros(b, c, h, w, d)
        z = 0
        for i1, i2 in srange(h, self.kernel_size[0], self.stride[0]):
            for j1, j2 in srange(w, self.kernel_size[1], self.stride[1]):
                for k1, k2 in srange(d, self.kernel_size[2], self.stride[2]):
                    out[:, :, i1:i2, j1:j2, k1:k2] = out[:, :, i1:i2, j1:j2, k1:k2] + x[..., z]
                    z += 1
        return out


class Unfold(nn.Module):
    def __init__(self, kernel_size, stride=1) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        h, w, d = x.shape[2:]

        outs = []
        for i1, i2 in srange(h, self.kernel_size[0], self.stride[0]):
            for j1, j2 in srange(w, self.kernel_size[1], self.stride[1]):
                for k1, k2 in srange(d, self.kernel_size[2], self.stride[2]):
                    outs.append(x[:, :, i1:i2, j1:j2, k1:k2].contiguous())
        out = th.stack(outs, -1)  # b d k1 k2 k3 l
        return out


def mul(seq):
    return reduce(lambda a, b: a * b, seq, 1)


def meshgrid(shape, device):
    l = len(shape)
    o = []
    for i in range(l):
        x = th.arange(0, shape[i], device=device)
        x = x.view(*(1 for _ in range(i)), shape[i], *(1 for _ in range(l - i)))
        v = [*shape, 1]
        v[i] = 1
        x = x.repeat(*v)
        o.append(x)

    arr = th.cat(o, dim=-1)
    return arr


def delta_border(shape, device):
    """
    :param h: height
    :param w: width
    :return: normalized distance to image border,
     wtith min distance = 0 at border and max dist = 0.5 at image center
    """
    lower_right_corner = th.tensor([sh - 1 for sh in shape], device=device).view(1, 1, len(shape))
    arr = meshgrid(shape, device=device) / lower_right_corner
    dist_left_up = th.min(arr, dim=-1, keepdims=True)[0]
    dist_right_down = th.min(1 - arr, dim=-1, keepdims=True)[0]
    edge_dist = th.min(th.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
    return edge_dist


def get_weighting(shape, L, device, clip_min_weight, clip_max_weight):
    weighting = delta_border(shape, device=device)
    weighting = th.clip(weighting, clip_min_weight, clip_max_weight)
    weighting = weighting.view(1, mul(shape), 1).repeat(1, 1, mul(L))
    return weighting


def get_fold_unfold(
    x, kernel_size, stride, uf=1, df=1, clip_min_weight=0.01, clip_max_weight=0.5
):  # todo load once not every time, shorten code
    """
    - input:
        - x: voxel
        - kernel_size: e.g. (32, 32, 32)
        - stride: e.g. (16, 16, 16)
        - uf: upsampling input
        - df: downsampling input
    - return:
        - fold
        - unfold
        - norm
        - weight
    """
    shape = x.shape[2:]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * len(shape)
    if isinstance(stride, int):
        stride = (stride,) * len(shape)

    # number of crops in image
    L = [(sh - ks) // st + 1 for sh, ks, st in zip(shape, kernel_size, stride)]

    if uf == 1 and df == 1:
        # fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
        fold_params = dict(kernel_size=kernel_size, stride=stride)
        unfold = Unfold(**fold_params)
        # fold = Fold(output_size=shape, **fold_params)
        fold = Fold(**fold_params)

        weighting = get_weighting(kernel_size, L, x.device, clip_min_weight, clip_max_weight).to(x.dtype)
        weighting = weighting.view((1, 1, *kernel_size, mul(L)))
        normalization = fold(weighting).view(1, 1, *shape)  # normalizes the overlap

    elif uf > 1 and df == 1:
        # fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
        fold_params = dict(kernel_size=kernel_size, stride=stride)
        unfold = Unfold(**fold_params)

        fold_params2 = dict(kernel_size=[ks * uf for ks in kernel_size], stride=[s * uf for s in stride])
        fold = Fold(**fold_params2)

        weighting = get_weighting([ks * uf for ks in kernel_size], L, x.device, clip_min_weight, clip_max_weight).to(x.dtype)
        weighting = weighting.view((1, 1, *[ks * uf for ks in kernel_size], mul(L)))
        normalization = fold(weighting).view(1, 1, *(u * uf for u in shape))  # normalizes the overlap

    elif df > 1 and uf == 1:
        # fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
        fold_params = dict(kernel_size=kernel_size, stride=stride)
        unfold = Unfold(**fold_params)

        fold_params2 = dict(kernel_size=[ks // df for ks in kernel_size], stride=[st // df for st in stride])
        fold = Fold(**fold_params2)

        weighting = get_weighting([ks // df for ks in kernel_size], L, x.device, clip_min_weight, clip_max_weight).to(x.dtype)
        weighting = weighting.view((1, 1, *[ks // df for ks in kernel_size], mul(L)))
        normalization = fold(weighting).view(1, 1, *(sh // df for sh in shape))  # normalizes the overlap

    else:
        raise NotImplementedError

    return fold, unfold, normalization, weighting


if __name__ == "__main__":
    x = th.rand(2, 3, 128, 128, 128)
    fold, unfold, norm, weight = get_fold_unfold(x, 32, 16)
    u = unfold(x)
    print(u.shape)  # 2 3 64 64 64 343

    x_recon = fold(u * weight) / norm
    print(x_recon.shape)  # 2 3 128 128 128

    diff = (x_recon - x).abs()
    print(diff.mean(), diff.min(), diff.max())  # tensor(2.2574e-08) tensor(0.) tensor(5.3644e-07)
