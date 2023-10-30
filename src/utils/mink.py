import MinkowskiEngine as ME
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


def sparse_tensor_like(x, feat, coo):
    return ME.SparseTensor(feat, coo, tensor_stride=x.tensor_stride, coordinate_manager=x.coordinate_manager)


def full_like(x, coo, value):
    if isinstance(value, torch.Tensor):
        value = value.squeeze().contiguous()
        assert value.dim() == 1
        d = value.size(0)
    else:
        d = 1

    coo = coo.to(x.F.device)
    feat = coo.new_full((coo.size(0), d), value, dtype=torch.float32)
    return sparse_tensor_like(x, feat, coo)


def zeros_like(x, coo=None):
    if coo is None:
        coo = x.C
    return full_like(x, coo, 0)


def ones_like(x, coo):
    if coo is None:
        coo = x.C
    return full_like(x, coo, 1)


def sparse_cat_union(a: ME.SparseTensor, b: ME.SparseTensor):
    cm = a.coordinate_manager
    assert cm == b.coordinate_manager, "different coords_man"
    assert a.tensor_stride == b.tensor_stride, "different tensor_stride"

    zeros_cat_with_a = torch.zeros([a.F.shape[0], b.F.shape[1]], dtype=a.dtype).to(a.device)
    zeros_cat_with_b = torch.zeros([b.F.shape[0], a.F.shape[1]], dtype=a.dtype).to(a.device)

    feats_a = torch.cat([a.F, zeros_cat_with_a], dim=1)
    feats_b = torch.cat([zeros_cat_with_b, b.F], dim=1)

    new_a = ME.SparseTensor(
        features=feats_a,
        coordinates=a.C,
        coordinate_manager=cm,
        tensor_stride=a.tensor_stride,
    )

    new_b = ME.SparseTensor(
        features=feats_b,
        coordinates=b.C,
        coordinate_manager=cm,
        tensor_stride=a.tensor_stride,
    )

    return new_a + new_b


def sparse_zero_union(a: ME.SparseTensor, b: ME.SparseTensor, fill_value_a=0, fill_value_b=0):
    cm = a.coordinate_manager
    assert cm == b.coordinate_manager, "different coords_man"
    assert a.tensor_stride == b.tensor_stride, "different tensor_stride"

    a0 = full_like(a, a.coo, fill_value_a)
    b0 = full_like(a, a.coo, fill_value_b)
    union = ME.MinkowskiUnion()
    au = union(a, b0)
    bu = union(a0, b)
    return au, bu


def sparse_bceloss(pred: ME.SparseTensor, target: ME.SparseTensor):
    assert pred.F.size(1) == 1
    target = get_target(pred, target.coordinate_map_key)
    return F.binary_cross_entropy_with_logits(pred.F.squeeze(), target.to(pred.F.dtype))


class SparseBCELoss(nn.Module):
    def forward(self, pred: ME.SparseTensor, target: ME.SparseTensor):
        return sparse_bceloss(pred, target)


@torch.no_grad()
def iou(a, b):
    a = ME.SparseTensor(a.C.new_ones(a.C.size(0), 1), a.C, tensor_stride=a.tensor_stride)
    b = ME.SparseTensor(
        b.C.new_ones(b.C.size(0), 1), b.C, tensor_stride=b.tensor_stride, coordinate_manager=a.coordinate_manager
    )
    u = ME.MinkowskiUnion()(a, b)
    return ((u.F == 2).sum() / u.F.size(0)).nan_to_num_(0, 0, 0)


@torch.no_grad()
def iou_batch(a, b):
    a = ME.SparseTensor(a.C.new_ones(a.C.size(0), 1), a.C, tensor_stride=a.tensor_stride)
    b = ME.SparseTensor(
        b.C.new_ones(b.C.size(0), 1), b.C, tensor_stride=b.tensor_stride, coordinate_manager=a.coordinate_manager
    )
    u = ME.MinkowskiUnion()(a, b)
    batch_idx = u.C[:, 0].contiguous()
    inter = scatter_sum((u.F == 2).float(), batch_idx, dim=0).squeeze_(1)  # b
    union = scatter_sum(torch.ones_like(u.F), batch_idx, dim=0).squeeze_(1)  # b
    return (inter / union).nan_to_num_(0, 0, 0)  # b


@torch.no_grad()
def get_target(out, target_key, kernel_size=1):
    target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
    cm = out.coordinate_manager
    strided_target_key = cm.stride(target_key, out.tensor_stride[0])
    kernel_map = cm.kernel_map(
        out.coordinate_map_key,
        strided_target_key,
        kernel_size=kernel_size,
        region_type=1,
    )
    for k, curr_in in kernel_map.items():
        target[curr_in[0].long()] = 1
    return target
