import torch as th
import torch.nn.functional as F


def gaussian_filter_nd(shape, sig, normalized=True, device="cpu"):
    dims = len(shape)
    grid = th.stack(th.meshgrid([th.linspace(-1, 1, s, device=device) for s in shape], indexing="ij"), dim=-1)
    grid = th.exp(-grid.square().sum(dim=-1) / 2 / sig**2) / ((2 * th.pi) ** 0.5 * sig) ** dims
    if normalized:
        grid /= grid.sum()
    return grid


def gaussian_blur(x, sig, kernel_size):
    dims = x.dim() - 2
    kernel = gaussian_filter_nd([kernel_size for _ in range(dims)], sig, normalized=True, device=x.device)
    kernel = kernel[None, None].repeat(x.size(1), 1, *[1 for _ in range(dims)])
    x = getattr(F, f"conv{dims}d")(x, kernel, padding="same", groups=x.size(1))
    return x
