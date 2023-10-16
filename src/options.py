import multiprocessing as mp
import os

os.environ["OMP_NUM_THREADS"] = str(min(mp.cpu_count(), 16))
import argparse
from datetime import datetime
from pathlib import Path

from easydict import EasyDict
from omegaconf import DictConfig, ListConfig, OmegaConf


def _load_yaml_recursive(cfg):
    keys_to_del = []
    for k in cfg.keys():
        if k == "__parent__":
            if isinstance(cfg[k], ListConfig):
                cfg2 = load_yaml(cfg[k][0])
                path = cfg[k][1].split(".")
                for p in path:
                    cfg2 = cfg2[p]
            else:
                cfg2 = load_yaml(cfg[k])

            keys_to_del.append(k)
            cfg = OmegaConf.merge(cfg2, cfg)
        elif isinstance(cfg[k], DictConfig):
            cfg[k] = _load_yaml_recursive(cfg[k])

    for k in keys_to_del:
        del cfg[k]

    return cfg


def load_yaml(path):
    cfg = OmegaConf.load(path)
    cfg = _load_yaml_recursive(cfg)
    return cfg


def get_config(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--gpus", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", default="train", help="train|make")
    parser.add_argument("--outdir")
    opt, unknown = parser.parse_known_args(argv)

    cfg = load_yaml(opt.config_file)
    cli = OmegaConf.from_dotlist(unknown)
    args = OmegaConf.merge(cfg, cli)

    args.gpus = list(map(int, opt.gpus.split(",")))
    args.debug = opt.debug
    args.mode = opt.mode
    args.outdir = opt.outdir

    if args.mode == "train":
        n = datetime.now()
        timestr = f"{n.year%100}{n.month:02d}{n.day:02d}_{n.hour:02d}{n.minute:02d}{n.second:02d}"
        timestr += "_" + Path(opt.config_file).stem
        if args.memo:
            timestr += "_%s" % args.memo
        if args.debug:
            timestr += "_debug"

        args.exp_path = os.path.join(args["exp_dir"], timestr)
        (Path(args.exp_path) / "samples").mkdir(parents=True, exist_ok=True)
        print("Start on exp_path:", args.exp_path)

        with open(os.path.join(args.exp_path, "args.yaml"), "w") as f:
            OmegaConf.save(args, f)

        print(OmegaConf.to_yaml(args, resolve=True))
        args = OmegaConf.to_container(args, resolve=True)
        args = EasyDict(args)
        args.exp_path = Path(args.exp_path)
    elif args.mode == "make":
        assert opt.outdir

        args = OmegaConf.to_container(args, resolve=True)
        args = EasyDict(args)
        args.exp_path = Path(args.exp_path)
        args.outdir = Path(args.outdir)
    else:
        raise NotImplementedError(args.mode)

    if args.debug:
        args.epochs = 2
        args.sample.save_sample_after_epoch = 0

    return args


def __test__():
    args = get_config(
        [
            "config/ddpm/vqaeonet/default.yaml",
            # "config/ae/aeonet/vqaeonet8192.yaml",
            "--gpus=0,1,2",
            "--debug",
            "dataset.params.batch_size=133",
            "memo=test",
        ]
    )
    from pprint import pprint

    pprint(args)


if __name__ == "__main__":
    __test__()
