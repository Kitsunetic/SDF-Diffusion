import importlib
import os
import random
import time
from collections import OrderedDict, defaultdict
from math import inf

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from tqdm import tqdm


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, val, n=1):
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt

    def get(self):
        return self.avg

    def __call__(self):
        return self.avg


class AverageMeters:
    def __init__(self, *keys) -> None:
        # self.data = OrderedDict({key: AverageMeter() for key in keys})
        self.data = defaultdict(AverageMeter)
        for k in keys:
            self.data[k]

    def __getitem__(self, key):
        return self.data[key]()

    def __getattr__(self, key):
        return self.data[key]()

    def update_dict(self, n, g):
        for k, v in g.items():
            self.data[k].update(v, n)

    def to_msg(self, format="%s:%.4f"):
        msgs = []
        for k, v in self.data.items():
            if k == "loss":
                msgs = [format % (k, v())] + msgs
            else:
                msgs.append(format % (k, v()))
        return " ".join(msgs)


def tqdm_(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            return tqdm(*args, **kwargs)
        else:
            return BlackHole()
    else:
        return tqdm(*args, **kwargs)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def get_model_params(model):
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    return model_size


class ChainDataset(Dataset):
    def __init__(self, *datasets) -> None:
        super().__init__()
        self.datasets = datasets
        self.lens = []
        self.cum_lens = []
        self.indices = []
        cum_n = 0
        for i, dataset in enumerate(self.datasets):
            n = len(dataset)
            self.lens.append(n)
            self.cum_lens.append(cum_n)
            self.indices += [i for _ in range(n)]
            cum_n += n
        self.total_len = sum(self.lens)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        ds_idx = self.indices[idx]
        out = self.datasets[ds_idx][idx - self.cum_lens[ds_idx]]
        return out


class SubDataset(Dataset):
    def __init__(self, dataset, indices) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        subidx = self.indices[idx]
        return self.dataset[subidx]


class Tiktok:
    def __init__(self) -> None:
        self.tok()

    def tik(self):
        return time.time() - self.now

    def tok(self):
        self.now = time.time()

    def tiktok(self):
        sec = self.tik()
        self.tok()
        return sec


class ChachedDataset(Dataset):
    def __init__(self, use_cache: bool) -> None:
        super().__init__()
        self.use_cache = use_cache
        self.cache = {}

    def __contains__(self, idx):
        return idx in self.cache

    def get(self, idx):
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

    def put(self, idx, data):
        if self.use_cache:
            self.cache[idx] = data


def instantiate_from_config(config, *args, **kwargs):
    # https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/util.py#L78
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(*args, **config.get("params", dict()), **kwargs)


def get_obj_from_str(string, reload=False):
    # https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/util.py#L88
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def tensor_to_image(images, nrow):
    # images: b 3 h w, [-1, 1]
    grid = make_grid(images, nrow=nrow).permute(1, 2, 0)  # H W 3 [-1, 1]
    # (x+1)/2 * 255 + 0.5 = 127.5x + 128, (반올림이 되게 하기 위해 0.5를 더함, 안 더하면 내림이 됨)
    grid = grid.mul_(127.5).add_(128).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    return grid


def try_remove_file(file):
    for _ in range(10):
        try:
            os.remove(file)
            break
        except:
            print("Warn: Failed to remove", file)
            time.sleep(0.1)


def safe_all_reduce(x, reduce_op=dist.ReduceOp.SUM):
    if dist.is_initialized():
        dist.all_reduce(x, reduce_op)
    return x


def safe_all_mean(x):
    x = safe_all_reduce(x)
    if dist.is_initialized():
        x /= dist.get_world_size()
    return x


def safe_all_gather(x, dim=0):
    if dist.is_initialized():
        xs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(xs, x)
        x = torch.cat(xs, dim=dim)
    return x


def safe_barrier():
    if dist.is_initialized():
        dist.barrier()


def safe_broadcast(x, src):
    if dist.is_initialized():
        dist.broadcast(x, src)


def refine_state_dict(ckpt):
    module_in_module = False
    for k in ckpt["model"]:
        if k.startswith("model."):
            module_in_module = True
            break

    if module_in_module:
        state_dict = OrderedDict()
        for k, v in ckpt["model"].items():
            if k.startswith("model."):
                state_dict[k[6:]] = v
    else:
        state_dict = ckpt["model"]
    return state_dict


class BlackHole(int):
    def __setattr__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        return self

    def __getitem__(self, *args, **kwargs):
        return self


def sdf_standardize(sdf, c, mu, sig, gamma):
    sdf = sdf.clamp(c[0], c[1])
    if gamma != 1.0:
        sdf = sdf.sign() * sdf.abs().pow(gamma)
    sdf = (sdf - mu) / sig
    return sdf


def sdf_destandardize(sdf, c, mu, sig, gamma):
    sdf = sdf * sig + mu
    if gamma != 1.0:
        sdf = sdf.sign() * sdf.abs().pow(1 / gamma)
    sdf = sdf.clamp(c[0], c[1])
    return sdf


def infinite_loop(self, iter):
    while True:
        for x in iter:
            yield x


def infinite_dataloader(dl, n_iters=inf):
    step = 0
    keep = True
    while keep:
        for batch in dl:
            yield batch
            step += 1
            if step > n_iters:
                keep = False
                break
