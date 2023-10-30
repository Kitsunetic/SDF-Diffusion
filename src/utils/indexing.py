import torch as th
import torch.nn.functional as F


def discrete_grid_sample(p, x):
    """
    - p: b n 3
    - x: b d r r r
    """
    r = x.size(-1)
    p = p[..., 0] * r**2 + p[..., 0] * r + p[..., 2]
    p = p[:, None, :].repeat(1, x.size(1), 1)  # b d n
    y = x.flatten(2).gather(-1, p)  # b d n
    y = y.transpose_(1, 2).contiguous()  # b n d
    return y


def random_sample(x, n, dim=-1):
    idx = th.randperm(x.size(dim))[:n]
    if dim < 0:
        dim = x.dim() + dim
    u = [slice(None) for _ in range(dim)] + [idx]
    return x[u]


def batched_randperm(shape, dim=-1, device="cpu"):
    """adapted from https://discuss.pyth.org/t/batch-version-of-torch-randperm/111121/2"""
    idx = th.argsort(th.rand(shape, device=device), dim=dim)
    return idx


def batched_random_sample(x, n, dim=-1):
    # 오류있음
    idx = th.argsort(th.rand(th.Size([x.size(0), x.size(dim)]), dim=1, device=x.device))
    if dim < 0:
        dim = x.dim() + dim
    u = [slice(None) for _ in range(dim)] + [slice(None, n)]
    idx = idx[u]
    return x.gather(dim, idx)


def random_point_sampling(x, n):
    # b n d
    idx = th.argsort(th.rand(x.shape[:2], device=x.device))  # b n
    idx = idx[:, :n]
    idx = idx[..., None].repeat(1, 1, x.size(-1))
    out = x.gather(1, idx)
    return out


def patchify2d(h, p, patch_scale):
    """
    - h: b d r r
    - p: b 2
    """
    r = h.size(-1)
    grid = th.linspace(0, 2 / patch_scale, r // patch_scale, device=h.device)
    grid = th.stack(th.meshgrid(grid, grid, indexing="xy"), dim=-1)[None]  # 1 r' r' 2
    grid = p[:, None, None] + grid  # b r' r' 2

    h = F.grid_sample(h, grid, padding_mode="border", align_corners=True)
    return h


def unsqueeze_as(x, y):
    if isinstance(y, th.Tensor):
        d = y.dim()
    else:
        d = len(y)
    return x.view(list(x.shape) + [1] * (d - x.dim()))
