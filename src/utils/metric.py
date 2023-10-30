import sys

import torch as th
from lib.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from lib.PyTorchEMD.emd import earth_mover_distance as EMD
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes
from src.utils import BlackHole
from torch_scatter import scatter_sum
from tqdm import tqdm, trange

cham3D = chamfer_3DDist()


class PointDist:
    def __init__(self, a, b) -> None:
        """
        :params a: b n 3
        :params b: b n 3
        """
        self.a = a
        self.b = b
        self._knn_done = False

    def _calc_knn(self):
        self._dist1, _, self._knn1 = knn_points(self.a, self.b, return_nn=True, return_sorted=False)
        self._dist2, _, self._knn2 = knn_points(self.b, self.a, return_nn=True, return_sorted=False)
        # dists: b n 1
        # knn: b n 1 3

    def chamfer_l1(self, reduction="mean"):
        if not self._knn_done:
            self._calc_knn()
        if reduction == "mean":
            d1 = (self.a - self._knn1[:, :, 0]).abs().sum(dim=-1).mean()
            d2 = (self.b - self._knn2[:, :, 0]).abs().sum(dim=-1).mean()
        elif reduction == "batchmean":
            d1 = (self.a - self._knn1[:, :, 0]).abs().sum(dim=-1).flatten(1).mean(1)
            d2 = (self.b - self._knn2[:, :, 0]).abs().sum(dim=-1).flatten(1).mean(1)
        return d1 + d2

    def chamfer_l1_legacy(self, reduction="mean"):
        if not self._knn_done:
            self._calc_knn()
        if reduction == "mean":
            d1 = (self.a - self._knn1[:, :, 0]).square().sum(dim=-1).sqrt().mean()
            d2 = (self.b - self._knn2[:, :, 0]).square().sum(dim=-1).sqrt().mean()
        elif reduction == "batchmean":
            d1 = (self.a - self._knn1[:, :, 0]).square().sum(dim=-1).sqrt().flatten(1).mean(1)
            d2 = (self.b - self._knn2[:, :, 0]).square().sum(dim=-1).sqrt().flatten(1).mean(1)
        return d1 + d2

    def chamfer_l2(self, reduction="mean"):
        if not self._knn_done:
            self._calc_knn()
        if reduction == "mean":
            d1 = self._dist1.mean()
            d2 = self._dist2.mean()
            return d1 + d2
        elif reduction == "batchmean":
            d1 = self._dist1.flatten(1).mean(1)
            d2 = self._dist2.flatten(1).mean(1)
            return d1 + d2
        else:
            raise NotImplementedError(reduction)

    def recall(self, threshold=1e-3):
        if not self._knn_done:
            self._calc_knn()
        return (self._dist1.flatten(1) < threshold).float().mean(1)  # b

    def precision(self, threshold=1e-3):
        if not self._knn_done:
            self._calc_knn()
        return (self._dist2.flatten(1) < threshold).float().mean(1)  # b

    def f1(self, threshold=1e-3, reduction="mean"):
        r = self.recall(threshold)  # b
        p = self.precision(threshold)  # b
        f1 = 2 * r * p / (r + p)  # b
        f1.nan_to_num_(0, 0, 0)

        if reduction == "mean":
            return f1.mean()
        elif reduction == "batchmean":
            return f1
        else:
            raise NotImplementedError(reduction)

    def emd(self, reduction="mean"):
        dist = EMD(self.a, self.b, transpose=False)  # b
        if reduction == "mean":
            return dist.mean()
        elif reduction == "batchmean":
            return dist
        else:
            raise NotImplementedError(reduction)


############################################################################
# evaluation code from PVD
# https://github.com/alexzhou907/PVD
############################################################################
def iou(a: th.Tensor, b: th.Tensor, res: int, reduction="mean"):
    """
    - input:
        - a: b n 3, [0, 1]
        - b: b m 3, [0, 1]
    """
    a = (a * res + (0.5 / res)).clamp_(0, res - 1).long()
    b = (b * res + (0.5 / res)).clamp_(0, res - 1).long()
    a = a[..., 0] + a[..., 1] * res + a[..., 2] * res**2  # b n
    b = b[..., 0] + b[..., 1] * res + b[..., 2] * res**2  # b m
    a = scatter_sum(th.ones_like(a), a, 1, dim_size=res**3) > 0  # b (r r r), bool
    b = scatter_sum(th.ones_like(b), b, 1, dim_size=res**3) > 0  # b (r r r), bool
    union = (a | b).sum(1)  # b
    inter = (a & b).sum(1)  # b
    score = (inter / union).nan_to_num_(0, 0, 0)

    if reduction == "mean":
        return score.mean()
    elif reduction == "batchmean":
        return score


def sdf_iou(a: th.Tensor, b: th.Tensor, reduction="mean"):
    """
    - a, b: b 1 r r r
    """
    a = a < 0
    b = b < 0
    union = (a | b).flatten(1).sum(1)  # b
    inter = (a & b).flatten(1).sum(1)  # b
    score = (inter / union).nan_to_num_(0, 0, 0)

    if reduction == "mean":
        return score.mean()
    elif reduction == "batchmean":
        return score


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = th.min(all_dist, dim=1)
    min_val, _ = th.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = th.tensor(cov).to(all_dist)
    return {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
        "lgan_mmd_smp": mmd_smp,
    }


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    dev = Mxx.device

    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = th.cat((th.ones(n0, device=dev), th.zeros(n1, device=dev)))
    M = th.cat((th.cat((Mxx, Mxy), 1), th.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float("inf")
    val, idx = (M + th.diag(INFINITY * th.ones(n0 + n1, device=dev))).topk(k, 0, False)

    count = th.zeros(n0 + n1, device=dev)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = th.ge(count, (float(k) / 2) * th.ones(n0 + n1, device=dev)).float()

    s = {
        "tp": (pred * label).sum(),
        "fp": (pred * (1 - label)).sum(),
        "fn": ((1 - pred) * label).sum(),
        "tn": ((1 - pred) * (1 - label)).sum(),
    }

    s.update(
        {
            "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
            "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),
            "acc": th.eq(label, pred).float().mean(),
        }
    )
    return s


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, show_pbar=False):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []

    pbar = tqdm(total=N_sample) if show_pbar else BlackHole()
    for sample_b_start in range(N_sample):
        sample_batch = sample_pcs[sample_b_start].cuda(non_blocking=True)

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end].cuda(non_blocking=True)

            batch_size_ref = ref_batch.size(0)
            # sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch[None].repeat(batch_size_ref, 1, 1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr, _, _ = cham3D(sample_batch_exp, ref_batch)
            cd = (dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1)
            # cd = th.rand(1, batch_size_ref, device="cuda")
            cd_lst.append(cd)

            emd_batch = EMD(sample_batch_exp, ref_batch, transpose=False).view(1, -1)
            # emd_batch = th.rand(1, batch_size_ref, device="cuda")
            emd_lst.append(emd_batch)

        cd_lst = th.cat(cd_lst, dim=1)
        emd_lst = th.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

        pbar.update()
    pbar.close()

    all_cd = th.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = th.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


def compute_all_dists(sample_pcs, ref_pcs, batch_size, show_pbar=False):
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size, show_pbar=show_pbar)  # n_ref, n_sample
    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size, show_pbar=show_pbar)  # n_ref, n_ref
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size, show_pbar=show_pbar)  # n_sample, n_sample
    return M_rs_cd, M_rs_emd, M_rr_cd, M_rr_emd, M_ss_cd, M_ss_emd


def compute_all_metrics(M_rs_cd, M_rs_emd, M_rr_cd, M_rr_emd, M_ss_cd, M_ss_emd):
    results = {}

    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({"%s-CD" % k: v for k, v in res_cd.items()})
    res_emd = lgan_mmd_cov(M_rs_emd.t())
    results.update({"%s-EMD" % k: v for k, v in res_emd.items()})

    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({"1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if "acc" in k})
    one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    results.update({"1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if "acc" in k})

    return results


def pairwise_cd(xs, ys, b):
    n, m = len(xs), len(ys)
    cd_all = []
    for i in trange(n, ncols=100, file=sys.stdout, desc="pairwise_cd"):
        x = xs[i, None].cuda(non_blocking=True)  # 1 n 3
        cd_lst = []
        for j in range(0, m, b):
            b_ = min(m - j, b)
            y = ys[j : j + b_].cuda(non_blocking=True)  # b n 3
            dl, dr, _, _ = cham3D(x.repeat(b_, 1, 1).contiguous(), y.contiguous())  # b n, b n
            cd = (dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1)  # 1 b
            cd_lst.append(cd)
        cd_lst = th.cat(cd_lst, dim=1)  # 1 m
        cd_all.append(cd_lst)
    cd_all = th.cat(cd_all, dim=0)  # n m
    return cd_all


def unpairwise_cd(xs, ys, xsx, ysx, b):
    # ~x is sparser one
    n, m = len(xs), len(ys)
    cd_all = []
    for i in trange(n, ncols=100, file=sys.stdout, desc="pairwise_cd"):
        x_d = xs[i, None].cuda(non_blocking=True)  # 1 n 3
        x_s = xsx[i, None].cuda(non_blocking=True)
        cd_lst = []
        for j in range(0, m, b):
            b_ = min(m - j, b)
            y_d = ys[j : j + b_].cuda(non_blocking=True)  # b n 3
            y_s = ysx[j : j + b_].cuda(non_blocking=True)  # b n 3
            dl, _, _, _ = cham3D(x_s.repeat(b_, 1, 1).contiguous(), y_d.contiguous())  # b n, b n
            _, dr, _, _ = cham3D(x_d.repeat(b_, 1, 1).contiguous(), y_s.contiguous())  # b n, b n
            cd = (dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1)  # 1 b
            cd_lst.append(cd)
        cd_lst = th.cat(cd_lst, dim=1)  # 1 m
        cd_all.append(cd_lst)
    cd_all = th.cat(cd_all, dim=0)  # n m
    return cd_all


def pairwise_emd(xs, ys, b):
    n, m = len(xs), len(ys)
    emd_all = []
    for i in trange(n, ncols=100, file=sys.stdout, desc="pairwise_emd"):
        x = xs[i, None].cuda(non_blocking=True)  # 1 n 3
        emd_lst = []
        for j in range(0, m, b):
            b_ = min(m - j, b)
            y = ys[j : j + b_].cuda(non_blocking=True)  # b n 3
            emd = EMD(x.repeat(b_, 1, 1).contiguous(), y.contiguous(), transpose=False).view(1, -1)  # 1 b
            emd_lst.append(emd)
        emd_lst = th.cat(emd_lst, dim=1)  # 1 m
        emd_all.append(emd_lst)
    emd_all = th.cat(emd_all, dim=0)  # n m
    return emd_all


def pairwise_mesh_to_point(xs, ys, b):
    n, m = len(xs), len(ys)
    cd_all = []
    for i in trange(n, ncols=100, file=sys.stdout, desc="pairwise_cd"):
        x = xs[i, None].cuda(non_blocking=True)  # 1 n 3
        cd_lst = []
        for j in range(0, m, b):
            b_ = min(m - j, b)
            y = ys[j : j + b_].cuda(non_blocking=True)  # b n 3
            dl, dr, _, _ = cham3D(x.repeat(b_, 1, 1).contiguous(), y.contiguous())  # b n, b n
            cd = (dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1)  # 1 b
            cd_lst.append(cd)
        cd_lst = th.cat(cd_lst, dim=1)  # 1 m
        cd_all.append(cd_lst)
    cd_all = th.cat(cd_all, dim=0)  # n m
    return cd_all


def compute_metric_cd(xs, ys, b):
    rs = pairwise_cd(ys, xs, b)
    rr = pairwise_cd(ys, ys, b)
    ss = pairwise_cd(xs, xs, b)

    results = {}
    res_mmd = lgan_mmd_cov(rs.T)
    results.update({"%s-CD" % k: v for k, v in res_mmd.items()})
    res_1nn = knn(rr, rs, ss, 1, sqrt=False)
    results.update({"1-NN-CD-%s" % k: v for k, v in res_1nn.items() if "acc" in k})
    return results


def compute_metric_ucd(xs, ys, xsx, ysx, b):
    rs = unpairwise_cd(xs, ys, xsx, ysx, b)
    rr = pairwise_cd(ys, ys, b)
    ss = pairwise_cd(xsx, xsx, b)

    results = {}
    res_mmd = lgan_mmd_cov(rs.T)
    results.update({"%s-CD" % k: v for k, v in res_mmd.items()})
    res_1nn = knn(rr, rs, ss, 1, sqrt=False)
    results.update({"1-NN-CD-%s" % k: v for k, v in res_1nn.items() if "acc" in k})
    return results


def compute_metric_emd(xs, ys, b):
    rs = pairwise_emd(ys, xs, b)
    rr = pairwise_emd(ys, ys, b)
    ss = pairwise_emd(xs, xs, b)

    results = {}
    res_mmd = lgan_mmd_cov(rs.T)
    results.update({"%s-EMD" % k: v for k, v in res_mmd.items()})
    res_1nn = knn(rr, rs, ss, 1, sqrt=False)
    results.update({"1-NN-EMD-%s" % k: v for k, v in res_1nn.items() if "acc" in k})
    return results


def compute_metric_all(xs, ys, b):
    results_cd = compute_metric_cd(xs, ys, b)
    results_emd = compute_metric_emd(xs, ys, b)

    results = {}
    results.update(results_cd)
    results.update(results_emd)
    return results


############################################################################
# point to triangle distance
# https://github.com/wang-ps/mesh2sdf/blob/master/csrc/makelevelset3.cpp#L19
############################################################################
def mag2(x):
    return x.square().sum(dim=-1, keepdim=True)


def dot(a, b):
    return (a * b).sum(dim=-1, keepdim=True)


def point_segment_distance(x0, x1, x2):
    dx = x2 - x1
    m2 = mag2(dx)
    # s12 = dot(x2 - x0, dx) / m2
    s12 = dot(x2 - x0, dx) / m2.clamp(min=1e-30)
    s12.clamp_(0.0, 1.0)
    return (x0 - (s12 * x1 + (1 - s12) * x2)).square().sum(dim=-1, keepdim=True)


def point_triangle_distance(x0, x1, x2, x3):
    # warn! return is squared distance
    # x: ... 3
    x13, x23, x03 = x1 - x3, x2 - x3, x0 - x3
    m13, m23, d = mag2(x13), mag2(x23), dot(x13, x23)
    invdet = 1.0 / (m13 * m23 - d * d).clamp(min=1e-30)
    a, b = dot(x13, x03), dot(x23, x03)
    w23 = invdet * (m23 * a - d * b)  # ...
    w31 = invdet * (m13 * b - d * a)
    w12 = 1 - w23 - w31

    dist = x0.new_zeros(*x0.shape[:-1], 1)  # ... 1
    mask0 = (w23 >= 0) & (w31 >= 0) & (w12 >= 0)
    mask1 = (w23 > 0) & ~mask0
    mask2 = (w31 > 0) & ~(mask0 & mask1)
    mask3 = ~(mask0 & mask1 & mask2)
    u0 = point_segment_distance(x0, x1, x2)
    u1 = point_segment_distance(x0, x1, x3)
    u2 = point_segment_distance(x0, x2, x3)
    dist += (x0 - (w23 * x1 + w31 * x2 + w12 * x3)).square().sum(dim=-1, keepdim=True) * mask0
    dist += th.cat([u0, u1], dim=-1).amin(dim=-1, keepdim=True) * mask1
    dist += th.cat([u0, u2], dim=-1).amin(dim=-1, keepdim=True) * mask2
    dist += th.cat([u1, u1], dim=-1).amin(dim=-1, keepdim=True) * mask3
    return dist[..., 0]  # ...


def point_to_mesh_distance(x, v, f):
    # x: points n 3
    # v: vertices m 3
    # f: faces l 3, long or int
    # return: n l
    fv = v.gather(0, f.long().flatten()[:, None].repeat(1, 3)).view(-1, 3, 3)  # l 3 3
    fv = fv[None].repeat(x.size(0), 1, 1, 1)  # n l 3 3
    xx = x[:, None].repeat(1, f.size(0), 1)  # n l 3
    return point_triangle_distance(xx, *fv.unbind(dim=2))


def pairwise_p2m(xs, ys, n_pts=None):
    # xs: b n_pts 3
    # ys: meshes list of (verts, faces) x m
    n, m = len(xs), len(ys)
    n_pts = n_pts or xs[0].size(0)

    # sample points from meshes(ys)
    ypts = []
    for i in range(m):
        yv = ys[i][0].float().cuda(non_blocking=True)  # n_verts 3
        yf = ys[i][1].long().cuda(non_blocking=True)  # n_faces 3
        ymesh = Meshes([yv], [yf])
        ypt = sample_points_from_meshes(ymesh, n_pts)  # 1 n_pts 3
        ypts.append(ypt)
    ypts = th.cat(ypts)  # m n_pts 3

    dist_all = []
    for i in trange(n, ncols=100, file=sys.stdout, desc="pairwise_p2m"):
        x = xs[i]  # n_pts 3
        dist_lst = []
        for j in range(m):
            yv = ys[j][0].float().cuda(non_blocking=True)  # n_verts 3
            yf = ys[j][1].long().cuda(non_blocking=True)  # n_faces 3
            y = ypts[j]  # n_pts 3
            dl = point_to_mesh_distance(x, yv, yf)  # n_pts n_faces
            dl = dl.amin(dim=-1).mean().view(1, 1)  # 1 1
            dr, *_ = knn_points(x[None], y[None])
            dr = dr[0, :, 0].mean().view(1, 1)
            dist = dl + dr  # 1 1
            dist_lst.append(dist)
        dist_lst = th.cat(dist_lst, dim=1)  # 1 m
        dist_all.append(dist_lst)
    dist_all = th.cat(dist_all, dim=0)  # n m
    return dist_all


def pairwise_m2m(xs, ys, n_pts):
    # xs: meshes list of (verts, faces) x m
    # ys: meshes list of (verts, faces) x m
    n, m = len(xs), len(ys)

    # sample points from meshes(xs)
    xpts = []
    for i in range(n):
        xv = xs[i][0].float().cuda(non_blocking=True)  # n_verts 3
        xf = xs[i][1].long().cuda(non_blocking=True)  # n_faces 3
        xmesh = Meshes([xv], [xf])
        xpt = sample_points_from_meshes(xmesh, n_pts)  # 1 n_pts 3
        xpts.append(xpt)
    xpts = th.cat(xpts)  # n n_pts 3
    # sample points from meshes(ys)
    ypts = []
    for i in range(m):
        yv = ys[i][0].float().cuda(non_blocking=True)  # n_verts 3
        yf = ys[i][1].long().cuda(non_blocking=True)  # n_faces 3
        ymesh = Meshes([yv], [yf])
        ypt = sample_points_from_meshes(ymesh, n_pts)  # 1 n_pts 3
        ypts.append(ypt)
    ypts = th.cat(ypts)  # m n_pts 3

    dist_all = []
    for i in trange(n, ncols=100, file=sys.stdout, desc="pairwise_m2m"):
        xv = xs[i][0].float().cuda(non_blocking=True)  # n_verts 3
        xf = xs[i][1].long().cuda(non_blocking=True)  # n_faces 3
        x = xpts[i]  # n_pts 3
        dist_lst = []
        for j in range(m):
            yv = ys[j][0].float().cuda(non_blocking=True)  # n_verts 3
            yf = ys[j][1].long().cuda(non_blocking=True)  # n_faces 3
            y = ypts[j]  # n_pts 3
            dl = point_to_mesh_distance(x, yv, yf)  # n_pts n_faces
            dl = dl.amin(dim=-1).mean().view(1, 1)  # 1 1
            dr = point_to_mesh_distance(y, xv, xf)  # n_pts n_faces
            dr = dr.amin(dim=-1).mean().view(1, 1)
            dist = dl + dr  # 1 1
            dist_lst.append(dist)
        dist_lst = th.cat(dist_lst, dim=1)  # 1 m
        dist_all.append(dist_lst)
    dist_all = th.cat(dist_all, dim=0)  # n m
    return dist_all


def compute_metric_p2m(xs, ys, b, n_pts=None):
    """
    - xs: (sample) b n 3
    - ys: (reference) list of meshes (verts, faces)
    """
    n_pts = n_pts or xs[0].size(0)
    rs = pairwise_p2m(xs, ys, n_pts=n_pts)
    rr = pairwise_m2m(ys, ys, n_pts=n_pts)
    ss = pairwise_cd(xs, xs, b)

    results = {}
    res_mmd = lgan_mmd_cov(rs.T)
    results.update({"%s-CD" % k: v for k, v in res_mmd.items()})
    res_1nn = knn(rr, rs, ss, 1, sqrt=False)
    results.update({"1-NN-CD-%s" % k: v for k, v in res_1nn.items() if "acc" in k})
    return results


def compute_metric_m2p(xs, ys, b, n_pts=None):
    """
    - xs: (reference) list of meshes (verts, faces)
    - ys: (sample) b n 3
    """
    n_pts = n_pts or xs[0].size(0)
    rs = pairwise_p2m(ys, xs, n_pts=n_pts)
    rr = pairwise_cd(ys, ys, b)
    ss = pairwise_m2m(xs, xs, n_pts=n_pts)

    results = {}
    res_mmd = lgan_mmd_cov(rs.T)
    results.update({"%s-CD" % k: v for k, v in res_mmd.items()})
    res_1nn = knn(rr, rs, ss, 1, sqrt=False)
    results.update({"1-NN-CD-%s" % k: v for k, v in res_1nn.items() if "acc" in k})
    return results


def compute_metric_m2m(xs, ys, b, n_pts=None):
    """
    - xs: (sample) list of meshes (verts, faces)
    - ys: (reference) list of meshes (verts, faces)
    """
    n_pts = n_pts or xs[0].size(0)
    rs = pairwise_m2m(xs, ys, n_pts=n_pts)
    rr = pairwise_m2m(ys, ys, n_pts=n_pts)
    ss = pairwise_m2m(xs, xs, n_pts=n_pts)

    results = {}
    res_mmd = lgan_mmd_cov(rs.T)
    results.update({"%s-CD" % k: v for k, v in res_mmd.items()})
    res_1nn = knn(rr, rs, ss, 1, sqrt=False)
    results.update({"1-NN-CD-%s" % k: v for k, v in res_1nn.items() if "acc" in k})
    return results
