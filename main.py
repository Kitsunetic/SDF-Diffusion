import os

import torch.multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = str(min(16, mp.cpu_count()))

import torch
import torch.distributed as dist

from src import logging, options, utils


def main_worker(rank, args):
    if args.ddp:
        dist.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=rank)

    args.rank = rank
    args.rankzero = rank == 0
    args.gpu = args.gpus[rank]
    torch.cuda.set_device(args.gpu)

    if args.rankzero:
        logging.basicConfig(args.exp_path / "main.log")
    else:
        logging.basicConfig(None, lock=True)
    args.log = logging.getLogger()

    args.seed += rank
    utils.seed_everything(args.seed)

    if args.ddp:
        print(f"main_worker with rank:{rank} (gpu:{args.gpu}) is loaded", torch.__version__)
    else:
        print(f"main_worker with gpu:{args.gpu} in main thread is loaded", torch.__version__)

    trainer = utils.instantiate_from_config(args.trainer, args)
    trainer.fit()
    utils.safe_barrier()


def main():
    args = options.get_config()

    args.world_size = len(args.gpus)
    args.ddp = args.world_size > 1
    port = utils.find_free_port()
    args.dist_url = f"tcp://127.0.0.1:{port}"

    if args.ddp:
        pc = mp.spawn(main_worker, nprocs=args.world_size, args=(args,), join=False)
        pids = " ".join(map(str, pc.pids()))
        print("\33[101mProcess Ids:", pids, "\33[0m")
        try:
            pc.join()
        except KeyboardInterrupt:
            print("\33[101mkill %s\33[0m" % pids)
            os.system("kill %s" % pids)
    else:
        main_worker(0, args)


if __name__ == "__main__":
    main()
