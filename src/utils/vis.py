import mcubes
import numpy as np
import point_cloud_utils as pcu
import torch
from pytorch3d.ops import cubify, sample_points_from_meshes
from pytorch3d.structures import Meshes
from skimage import measure


def mc_from_psr(psr_grid, pytorchify=False, real_scale=False, zero_level=0):
    """
    Run marching cubes from PSR grid
    from Shape as Points
    """
    batch_size = psr_grid.shape[0]
    s = psr_grid.shape[-1]  # size of psr_grid
    psr_grid_numpy = psr_grid.squeeze().detach().cpu().numpy()

    if batch_size > 1:
        verts, faces, normals = [], [], []
        for i in range(batch_size):
            verts_cur, faces_cur, normals_cur, values = measure.marching_cubes(psr_grid_numpy[i], level=0)
            verts.append(verts_cur)
            faces.append(faces_cur)
            normals.append(normals_cur)
        verts = np.stack(verts, axis=0)
        faces = np.stack(faces, axis=0)
        normals = np.stack(normals, axis=0)
    else:
        try:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy, level=zero_level)
        except:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy)
    if real_scale:
        verts = verts / (s - 1)  # scale to range [0, 1]
    else:
        verts = verts / s  # scale to range [0, 1)

    if pytorchify:
        device = psr_grid.device
        verts = torch.Tensor(np.ascontiguousarray(verts)).to(device)
        faces = torch.Tensor(np.ascontiguousarray(faces)).to(device)
        normals = torch.Tensor(np.ascontiguousarray(-normals)).to(device)

    return verts, faces, normals


def make_pointclouds_grid(pts, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - pts: list of (n (3 or 6)), numpy or Tensor
    - return:
        - pts: N (3 or 6)
    """
    if isinstance(pts[0], torch.Tensor):
        return _make_pointclouds_grid_torch(pts, min_v, max_v, padding, nrows)
    elif isinstance(pts[0], np.ndarray):
        return _make_pointclouds_grid_numpy(pts, min_v, max_v, padding, nrows)
    else:
        raise TypeError


def _make_pointclouds_grid_numpy(pts, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - pts: list of (n (3 or 6)), numpy
    - return:
        - pts: N (3 or 6)
    """
    dist = max_v - min_v
    out_pts = []
    for i in range(len(pts)):
        pos_x, pos_y = i % nrows, i // nrows
        off_x = pos_x * (dist + padding)
        off_y = pos_y * (dist + padding)
        offset = np.array([[off_x, off_y, *((0,) * (pts[0].shape[-1] - 2))]])
        out_pts.append(pts[i] + offset)
    pts = np.concatenate(out_pts, 0)  # N (3 or 6)
    return pts


@torch.no_grad()
def _make_pointclouds_grid_torch(pts, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - pts: list of (n (3 or 6)), Tensor
    - return:
        - pts: N (3 or 6)
    """
    dist = max_v - min_v
    out_pts = []
    for i in range(len(pts)):
        pos_x, pos_y = i % nrows, i // nrows
        off_x = pos_x * (dist + padding)
        off_y = pos_y * (dist + padding)
        offset = pts[i].new_tensor([[off_x, off_y, *((0,) * (pts[0].shape[-1] - 2))]])
        out_pts.append(pts[i] + offset)
    pts = torch.cat(out_pts, 0)  # N (3 or 6)
    return pts


def make_meshes_grid(verts, faces, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - verts: list of (n (3 ~)), numpy
        - faces: list of n 3, numpy, int
    - return:
        - verts: n (3 ~)
        - faces: n 3
    """
    assert len(verts) == len(faces)

    dist = max_v - min_v
    face_offset = 0
    out_verts, out_faces = [], []
    for i in range(len(verts)):
        pos_x, pos_y = i % nrows, i // nrows
        off_x = pos_x * (dist + padding)
        off_y = pos_y * (dist + padding)
        offset = np.array([[off_x, off_y, *((0,) * (verts[0].shape[-1] - 2))]])
        out_verts.append(verts[i] + offset)
        out_faces.append(faces[i] + face_offset)
        face_offset += verts[i].shape[0]
    verts = np.concatenate(out_verts)  # N (3 or 6)
    faces = np.concatenate(out_faces)  # N 3
    return verts, faces


def random_color(verts):
    """
    - input:
        - verts: n 3
    """
    color = np.random.random(1, 3)
    color = np.repeat(color, verts.shape[0], 0)  # n 3
    return np.concatenate([verts, color], -1)  # n 6


def sdfs_to_meshes(psrs, safe=False):
    """
    - input:
        - psrs: b 1 r r r
    - return:
        - meshes
    """
    mvs, mfs, mns = [], [], []
    for psr in psrs:
        if safe:
            try:
                mv, mf, mn = mc_from_psr(psr, pytorchify=True)
            except:
                mv = psrs.new_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                mf = psrs.new_tensor([[0, 1, 2]], dtype=torch.long)
                mn = psrs.new_tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        else:
            mv, mf, mn = mc_from_psr(psr, pytorchify=True)
        mvs.append(mv)
        mfs.append(mf)
        mns.append(mn)

    mesh = Meshes(mvs, mfs, verts_normals=mns)
    return mesh


def sdfs_to_meshes_np(psrs, safe=False, rescale_verts=False):
    """
    - input:
        - psrs: b 1 r r r
    - return:
        - verts: list of (n 3)
        - faces: list of (m 3)
    """
    mesh = sdfs_to_meshes(psrs, safe=safe)
    vs1, fs1 = mesh.verts_list(), mesh.faces_list()
    vs2, fs2 = [], []
    for i in range(len(vs1)):
        v = (vs1[i] * 2 - 1) if rescale_verts else vs1[i]
        vs2.append(v.cpu().numpy())
        fs2.append(fs1[i].cpu().numpy())
    return vs2, fs2


def sdf_to_point(sdf, n_points, safe=False):
    """
    - input:
        - sdf: 1 r r r
    - return:
        - point: n_points 3
    """
    if safe:
        try:
            mv, mf, mn = mc_from_psr(sdf, pytorchify=True)
            mesh = Meshes([mv], [mf], verts_normals=[mn])
            pts = sample_points_from_meshes(mesh, n_points)
        except RuntimeError:
            pts = sdf.new_zeros(1, n_points, 3)
    else:
        mv, mf, mn = mc_from_psr(sdf, pytorchify=True)
        mesh = Meshes([mv], [mf], verts_normals=[mn])
        pts = sample_points_from_meshes(mesh, n_points)

    return pts[0]


def sdfs_to_points(sdfs, n_points, safe=False):
    """
    - input:
        - sdfs: b 1 r r r
    - return:
        - points: b n_points 3
    """
    return torch.stack([sdf_to_point(sdf, n_points, safe=safe) for sdf in sdfs])


def sdf_to_point_fast(sdf, n_points):
    """
    - input:
        - sdf: 1 r r r
    - return:
        - point: n_points 3
    """
    mesh = cubify(-sdf, 0)
    pts = sample_points_from_meshes(mesh, n_points)
    return pts[0]


def sdfs_to_points_fast(sdfs, n_points):
    """
    - input:
        - sdfs: b 1 r r r
    - return:
        - points: b n_points 3
    """
    return torch.stack([sdf_to_point_fast(sdf, n_points) for sdf in sdfs])


def save_sdf_as_mesh(path, sdf, safe=False):
    """
    - input:
        - sdf: 1 r r r
    """
    verts, faces = sdfs_to_meshes_np(sdf[None], safe=safe)
    pcu.save_mesh_vf(str(path), verts[0], faces[0])


def udf2mesh(udf, grad, b_max, b_min, resolution):
    """
    - udf: r r r, numpy
    - grad: r r r 3, numpy
    """
    v_all = []
    f_all = []
    threshold = 0.005  # accelerate extraction
    v_num = 0
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            for k in range(resolution - 1):
                ndf_loc = udf[i : i + 2]
                ndf_loc = ndf_loc[:, j : j + 2, :]
                ndf_loc = ndf_loc[:, :, k : k + 2]
                if np.min(ndf_loc) > threshold:
                    continue
                grad_loc = grad[i : i + 2]
                grad_loc = grad_loc[:, j : j + 2, :]
                grad_loc = grad_loc[:, :, k : k + 2]

                res = np.ones((2, 2, 2))
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            if np.dot(grad_loc[0][0][0], grad_loc[ii][jj][kk]) < 0:
                                res[ii][jj][kk] = -ndf_loc[ii][jj][kk]
                            else:
                                res[ii][jj][kk] = ndf_loc[ii][jj][kk]

                if res.min() < 0:
                    vertices, triangles = mcubes.marching_cubes(res, 0.0)
                    # print(vertices)
                    # vertices -= 1.5
                    # vertices /= 128
                    vertices[:, 0] += i  # / resolution
                    vertices[:, 1] += j  # / resolution
                    vertices[:, 2] += k  # / resolution
                    triangles += v_num
                    # vertices =
                    # vertices[:,1] /= 3  # TODO
                    v_all.append(vertices)
                    f_all.append(triangles)

                    v_num += vertices.shape[0]
                    # print(v_num)

    v_all = np.concatenate(v_all)
    f_all = np.concatenate(f_all)
    # Create mesh
    v_all = v_all / (resolution - 1.0) * (b_max - b_min)[None, :] + b_min[None, :]

    return v_all, f_all
