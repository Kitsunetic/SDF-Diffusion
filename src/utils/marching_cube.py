# https://github.com/96lives/gca/blob/main/utils/marching_cube.py
import mcubes
import torch
import trimesh


def marching_cube(query_points: torch.Tensor, df: torch.Tensor, march_th, upsample=1, voxel_size=None):
    """
    Args:
        query_points: (N, 3) torch tensor
        df: (N) torch tensor
        march_th: threshold for marching cube algorithm
        upsample: required for upsampling the resolution of the marching cube

    Returns:
        mesh (trimesh object): obtained mesh from marching cube algorithm
    """
    df_points = query_points.clone().detach().cpu()
    offset = df_points.min(dim=0).values
    df_points = df_points - offset
    df_coords = torch.round(upsample * df_points).long() + 1
    march_bbox = df_coords.max(dim=0).values + 2  # out max
    voxels = torch.ones(march_bbox.tolist()).to(df.device)
    voxels[df_coords[:, 0], df_coords[:, 1], df_coords[:, 2]] = df.clone().detach().cpu()

    v, t = mcubes.marching_cubes(voxels.cpu().detach().numpy(), march_th)
    v = (v - 1) / upsample
    v += offset.cpu().numpy()
    if voxel_size is not None:
        v *= voxel_size
    mesh = trimesh.Trimesh(v, t)
    return mesh


def marching_cubes_sparse_voxel(coord: torch.Tensor, voxel_size=None):
    return marching_cube(
        query_points=coord.float(), df=torch.zeros(coord.shape[0]), march_th=0.5, upsample=1, voxel_size=voxel_size
    )


def marching_cubes_occ_grid(occ: torch.Tensor, threshold=0.5, scale=None):
    """
    :param occ: tensor H x W x D
    :param scale: tuple of scale_min, scale_max
    :return: normalized mesh, where each vertices are in [0, 1]
    """
    grid_shape = torch.tensor(occ.shape) + 2
    padded_grid = torch.zeros(grid_shape.tolist())
    padded_grid[1:-1, 1:-1, 1:-1] = occ
    v, t = mcubes.marching_cubes(padded_grid.cpu().detach().numpy(), threshold)
    v = (v - 1) / (occ.shape[0] - 1)

    if scale is not None:
        scale_min, scale_max = scale[0], scale[1]
        v = (scale_max - scale_min) * v
        v = v + scale_min
    return trimesh.Trimesh(v, t)


def coo_to_mesh(c: torch.Tensor):
    """
    - input:
        - c: n 3
    - return:
        - v: n 3, int32 cpu
        - f: n 3, int32 cpu
    """
    v_kernel = c.new_tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )
    f_kernel = c.new_tensor(
        [
            [0, 2, 1],
            [2, 4, 1],
            [1, 4, 5],
            [4, 7, 5],
            [2, 6, 4],
            [6, 7, 4],
            [3, 6, 0],
            [6, 2, 0],
            [5, 7, 3],
            [7, 6, 3],
            [3, 0, 5],
            [0, 1, 5],
        ]
    )

    n = c.size(0)
    v = c[:, None, :] + v_kernel[None, :, :]
    f = torch.arange(n, dtype=c.dtype, device=c.device)[:, None, None] * 8 + f_kernel[None, :, :]

    v = v.flatten(0, 1).contiguous()
    f = f.flatten(0, 1).contiguous()
    return v, f
