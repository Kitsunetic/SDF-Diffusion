import math
import random
import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.logging import CustomLogger

_TQDM_NCOLS = 169


class BasePreprocessor(metaclass=ABCMeta):
    def __init__(self, device) -> None:
        self.device = device

    @abstractmethod
    def __call__(self, batch, augmentation=False):
        pass


class BaseWorker(metaclass=ABCMeta):
    def __init__(self, args) -> None:
        self.args = args

    @property
    def rank(self):
        return self.args.rank

    @property
    def rankzero(self):
        return self.args.rank == 0

    @property
    def world_size(self):
        return self.args.world_size

    @property
    def ddp(self):
        return self.args.ddp

    @property
    def log(self) -> CustomLogger:
        return self.args.log

    def _tqdm(self, total, prefix):
        if self.rankzero:
            desc = f"{prefix} [{self.epoch:04d}/{self.args.epochs:04d}]"
            return tqdm(total=total, ncols=_TQDM_NCOLS, file=sys.stdout, desc=desc, leave=True)
        else:
            return utils.BlackHole()

    def safe_gather(self, x, cat=True, cat_dim=0):
        if self.ddp:
            xs = [torch.empty_like(x) for _ in range(self.args.world_size)]
            dist.all_gather(xs, x)
            if cat:
                return torch.cat(xs, dim=cat_dim)
            else:
                return xs
        else:
            return x

    def safe_reduce(self, x, op=dist.ReduceOp.SUM):
        if self.ddp:
            dist.all_reduce(x, op=op)
        return x

    def safe_barrier(self):
        if self.ddp:
            dist.barrier()

    def collect_log(self, s, prefix="", postfix=""):
        keys = list(s.log.keys())
        if self.ddp:
            g = s.log.loss.new_tensor([self._t2f(s.log[k]) for k in keys], dtype=torch.float) * s.n
            dist.all_reduce(g)
            n = s.n * self.args.world_size
            g /= n

            out = OrderedDict()
            for k, v in zip(keys, g.tolist()):
                out[prefix + k + postfix] = v
        else:
            out = OrderedDict()
            for k in keys:
                out[prefix + k + postfix] = self._t2f(s.log[k])
            n = s.n
        return n, out

    def g_to_msg(self, g):
        msg = ""
        for k, v in g.items():
            msg += " %s:%.4f" % (k, v)
        return msg[1:]

    def _t2f(self, x):
        if isinstance(x, torch.Tensor):
            return x.item()
        else:
            return x


class BaseTrainer(BaseWorker):
    def __init__(
        self,
        args,
        n_samples_per_class=10,
        find_unused_parameters=True,
        sample_at_least_per_epochs=None,
        mixed_precision=False,
    ) -> None:
        super().__init__(args)

        self.n_samples_per_class = n_samples_per_class
        self.find_unused_parameters = find_unused_parameters
        self.sample_at_least_per_epochs = sample_at_least_per_epochs
        self.mixed_precision = mixed_precision

        if self.mixed_precision:
            self.scaler = GradScaler()

        self.best = math.inf
        self.best_epoch = -1

        self.build_network()
        self.build_dataset()
        self.build_sample_idx()
        self.build_preprocessor()

    def build_network(self):
        self.model = utils.instantiate_from_config(self.args.model).cuda()
        if self.ddp:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model_optim = DDP(
                self.model,
                device_ids=[self.args.gpu],
                find_unused_parameters=self.find_unused_parameters,
            ).cuda()
        else:
            self.model_optim = self.model

        self.optim = utils.instantiate_from_config(self.args.optim, self.model_optim.parameters())
        if "sched" in self.args:
            self.sched = utils.instantiate_from_config(self.args.sched, self.optim)
        else:
            self.sched = None
        if "criterion" in self.args:
            self.criterion = utils.instantiate_from_config(self.args.criterion)
            if hasattr(self.criterion, "cuda"):
                self.criterion.cuda()

        # self.log.info(self.model)
        self.log.info("Model Params: %.2fM" % (self.model_params / 1e6))

    def build_dataset(self):
        dls = utils.instantiate_from_config(self.args.dataset, self.ddp)
        if len(dls) == 3:
            self.dl_train, self.dl_valid, self.dl_test = dls
            l1, l2, l3 = len(self.dl_train.dataset), len(self.dl_valid.dataset), len(self.dl_test.dataset)
            self.log.info("Load %d train, %d valid, %d test items" % (l1, l2, l3))
        elif len(dls) == 2:
            self.dl_train, self.dl_valid = dls
            l1, l2 = len(self.dl_train.dataset), len(self.dl_valid.dataset)
            self.log.info("Load %d train, %d valid items" % (l1, l2))
        else:
            raise NotImplementedError

    def build_preprocessor(self):
        self.preprocessor: BasePreprocessor = utils.instantiate_from_config(self.args.preprocessor, device=self.device)

    def build_sample_idx(self):
        # indices to generate at sample generation step
        self.n_generate = n = self.dl_test.dataset.n_classes * self.n_samples_per_class
        self.n_generate_rank = m = math.ceil(n / self.args.world_size)

        if hasattr(self.dl_test.dataset, "get_sample_idx"):
            self.sample_idx = self.dl_test.dataset.get_sample_idx(n)
            self.sample_idx = self.sample_idx[self.args.rank * m : (self.args.rank + 1) * m]
            if len(self.sample_idx) < m:
                self.sample_idx += [0 for _ in range(m - len(self.sample_idx))]
        else:
            self.sample_idx = random.sample(list(range(len(self.dl_test.dataset))), m)

    def save(self, out_path):
        data = {
            "epoch": self.epoch,
            "best_loss": self.best,
            "model": self.model.state_dict(),
        }
        torch.save(data, str(out_path))

    def step(self, s):
        pass

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def model_params(self):
        model_size = 0
        for param in self.model.parameters():
            if param.requires_grad:
                model_size += param.data.nelement()
        return model_size

    def on_train_batch_end(self, s):
        pass

    def on_valid_batch_end(self, s):
        pass

    def train_epoch(self, dl: "DataLoader", prefix="Train"):
        self.model_optim.train()
        o = utils.AverageMeters()

        if self.rankzero:
            desc = f"{prefix} [{self.epoch:04d}/{self.args.epochs:04d}]"
            t = tqdm(total=len(dl.dataset), ncols=150, file=sys.stdout, desc=desc, leave=True)
        for batch in dl:
            s = self.preprocessor(batch, augmentation=True)
            with autocast(self.mixed_precision):
                self.step(s)

            if self.mixed_precision:
                self.scaler.scale(s.log.loss).backward()
                if self.args.train.clip_grad > 0:  # gradient clipping
                    self.scaler.unscale_(self.optim)
                    nn.utils.clip_grad.clip_grad_norm_(self.model_optim.parameters(), self.args.train.clip_grad)
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                s.log.loss.backward()
                if self.args.train.clip_grad > 0:  # gradient clipping
                    nn.utils.clip_grad.clip_grad_norm_(self.model_optim.parameters(), self.args.train.clip_grad)
                self.optim.step()
            self.optim.zero_grad()

            self.step_sched(is_on_batch=True)

            n, g = self.collect_log(s)
            o.update_dict(n, g)
            if self.rankzero:
                t.set_postfix_str(o.to_msg(), refresh=False)
                t.update(min(n, t.total - t.n))

            self.on_train_batch_end(s)

            if self.args.debug:
                break
        if self.rankzero:
            t.close()
        return o

    @torch.no_grad()
    def valid_epoch(self, dl: "DataLoader", prefix="Valid"):
        self.model_optim.eval()
        o = utils.AverageMeters()

        if self.rankzero:
            desc = f"{prefix} [{self.epoch:04d}/{self.args.epochs:04d}]"
            t = tqdm(total=len(dl.dataset), ncols=150, file=sys.stdout, desc=desc, leave=True)
        for batch in dl:
            s = self.preprocessor(batch, augmentation=False)
            self.step(s)

            n, g = self.collect_log(s)
            o.update_dict(n, g)
            if self.rankzero:
                t.set_postfix_str(o.to_msg(), refresh=False)
                t.update(min(n, t.total - t.n))

            self.on_valid_batch_end(s)

            if self.args.debug:
                break
        if self.rankzero:
            t.close()
        return o

    @torch.no_grad()
    def evaluation(self, o1, o2):
        self.step_sched(o2.loss, is_on_epoch=True)

        improved = False
        if self.rankzero:  # scores are not calculated in other nodes
            flag = ""
            if o2.loss < self.best or (
                self.sample_at_least_per_epochs is not None
                and (self.epoch - self.best_epoch) >= self.sample_at_least_per_epochs
            ):
                self.best = min(self.best, o2.loss)
                self.best_epoch = self.epoch
                self.save(self.args.exp_path / "best_ep{:04d}.pth".format(self.epoch))
                saved_files = sorted(list(self.args.exp_path.glob("best_ep*.pth")))
                if len(saved_files) > self.args.train.num_saves:
                    to_deletes = saved_files[: len(saved_files) - self.args.train.num_saves]
                    for to_delete in to_deletes:
                        utils.try_remove_file(str(to_delete))

                flag = "*"
                improved = self.epoch > self.args.sample.epochs_to_save or self.args.debug

            msg = "Epoch[%03d/%03d]" % (self.epoch, self.args.epochs)
            msg += " loss[%.4f;%.4f]" % (o1.loss, o2.loss)
            msg += " (best:%.4f%s)" % (self.best, flag)
            for k in sorted(list(set(o1.data.keys()) | set(o2.data.keys()))):
                if k == "loss":
                    continue

                if k in o1.data and k in o2.data:
                    msg += " %s[%.4f;%.4f]" % (k, o1[k], o2[k])
                elif k in o2.data:
                    msg += " %s[-;%.4f]" % (k, o2[k])
                else:
                    msg += " %s[%.4f;-]" % (k, o1[k])
            self.log.info(msg)
            self.log.flush()

        # share improved condition with other nodes
        if self.ddp:
            improved = torch.tensor([improved], device="cuda")
            dist.broadcast(improved, 0)

        return improved

    def fit_loop(self):
        o1 = self.train_epoch(self.dl_train)
        o2 = self.valid_epoch(self.dl_valid)
        improved = self.evaluation(o1, o2)
        if improved:
            self.sample()

    def fit(self):
        for self.epoch in range(1, self.args.epochs + 1):
            self.fit_loop()

    def sample(self):
        pass

    def step_sched(self, loss=None, is_on_batch=False, is_on_epoch=False):
        if self.sched is None:
            return
        if (is_on_batch and self.args.sched.step_on_batch) or (is_on_epoch and self.args.sched.step_on_epoch):
            if self.sched.__class__.__name__ in ("ReduceLROnPlateau", "ReduceLROnPlateauWithWarmup"):
                assert loss is not None
                self.sched.step(loss)
            else:
                self.sched.step()


class StepTrainer(BaseTrainer):
    def __init__(
        self,
        args,
        n_steps,
        save_per_steps,
        valid_per_steps,
        sample_per_steps,
        n_samples_per_class=10,
        find_unused_parameters=True,
        sample_at_least_per_epochs=None,
        mixed_precision=False,
    ) -> None:
        super().__init__(
            args,
            n_samples_per_class,
            find_unused_parameters,
            sample_at_least_per_epochs,
            mixed_precision,
        )
        self.n_steps = n_steps
        self.save_per_steps = save_per_steps
        self.valid_per_steps = valid_per_steps
        self.sample_per_steps = sample_per_steps

    def train_batch(self, batch, o: utils.AverageMeters):
        s = self.preprocessor(batch, augmentation=True)
        with autocast(self.mixed_precision):
            self.step(s)

        if self.mixed_precision:
            self.scaler.scale(s.log.loss).backward()
            if self.args.train.clip_grad > 0:  # gradient clipping
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad.clip_grad_norm_(self.model_optim.parameters(), self.args.train.clip_grad)
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            s.log.loss.backward()
            if self.args.train.clip_grad > 0:  # gradient clipping
                nn.utils.clip_grad.clip_grad_norm_(self.model_optim.parameters(), self.args.train.clip_grad)
            self.optim.step()
        self.optim.zero_grad()

        n, g = self.collect_log(s)
        o.update_dict(n, g)

        self.on_train_batch_end(s)

    @torch.no_grad()
    def valid_epoch(self, dl: "DataLoader", prefix="Valid"):
        self.model_optim.eval()
        o = utils.AverageMeters()

        if self.rankzero:
            desc = f"{prefix} [{self.epoch:04d}/{self.n_steps:04d}]"
            t = tqdm(total=len(dl.dataset), ncols=150, file=sys.stdout, desc=desc, leave=True)
        for batch in dl:
            s = self.preprocessor(batch, augmentation=False)
            self.step(s)

            n, g = self.collect_log(s)
            o.update_dict(n, g)
            if self.rankzero:
                t.set_postfix_str(o.to_msg(), refresh=False)
                t.update(min(n, t.total - t.n))

            self.on_valid_batch_end(s)

            if self.args.debug:
                break
        if self.rankzero:
            t.close()
        print()
        return o

    @torch.no_grad()
    def evaluation(self, o1, o2):
        self.step_sched(o2.loss, is_on_epoch=True)

        msg = "Epoch[%03d/%03d]" % (self.epoch, self.n_steps)
        msg += " loss[%.4f;%.4f]" % (o1.loss, o2.loss)
        for k in sorted(list(set(o1.data.keys()) | set(o2.data.keys()))):
            if k == "loss":
                continue

            if k in o1.data and k in o2.data:
                msg += " %s[%.4f;%.4f]" % (k, o1[k], o2[k])
            elif k in o2.data:
                msg += " %s[-;%.4f]" % (k, o2[k])
            else:
                msg += " %s[%.4f;-]" % (k, o1[k])
        self.log.info(msg)
        self.log.flush()

    def fit(self):
        o_train = utils.AverageMeters()
        with tqdm(total=self.n_steps, ncols=150, file=sys.stdout, disable=not self.rankzero, desc="Step") as t:
            self.model_optim.train()
            for self.epoch, batch in enumerate(utils.infinite_dataloader(self.dl_train), 1):
                self.train_batch(batch, o_train)
                t.set_postfix_str(o_train.to_msg())

                if self.save_per_steps is not None and (self.epoch % self.save_per_steps == 0 or self.args.debug):
                    self.save(self.args.exp_path / "best_step{:08d}.pth".format(self.epoch))
                if self.valid_per_steps is not None and (self.epoch % self.valid_per_steps == 0 or self.args.debug):
                    with torch.no_grad():
                        self.model_optim.eval()
                        o_valid = self.valid_epoch(self.dl_valid)
                        self.evaluation(o_train, o_valid)
                        o_train = utils.AverageMeters()
                if self.sample_per_steps is not None and (self.epoch % self.sample_per_steps == 0 or self.args.debug):
                    with torch.no_grad():
                        self.model_optim.eval()
                        self.sample()
                self.model_optim.train()

                t.update()
                if self.args.debug and self.epoch >= 2:
                    break
                if self.epoch >= self.n_steps:
                    break
