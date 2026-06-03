import inspect

import numpy as np
import utils3d


_u3d_np = utils3d.numpy if hasattr(utils3d, "numpy") else utils3d.np


def icosahedron() -> tuple[np.ndarray, np.ndarray]:
    if hasattr(_u3d_np, "icosahedron"):
        vertices, faces = _u3d_np.icosahedron()
        return vertices.astype(np.float32), faces.astype(np.int32)

    vertices, faces = utils3d.np.create_icosahedron_mesh()
    return vertices.astype(np.float32), faces.astype(np.int32)


def image_uv(width: int, height: int) -> np.ndarray:
    if hasattr(_u3d_np, "image_uv"):
        return _u3d_np.image_uv(width=width, height=height)

    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu, vv], axis=-1)


def uv_to_pixel(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    fn = _u3d_np.uv_to_pixel
    params = inspect.signature(fn).parameters
    if "width" in params and "height" in params:
        return fn(uv, width=width, height=height)
    return fn(uv, size=(height, width))


def unproject_cv(uv: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    fn = _u3d_np.unproject_cv
    params = inspect.signature(fn).parameters
    depth_param = params.get("depth")
    if depth_param is not None and depth_param.default is inspect.Signature.empty:
        depth = np.ones(uv.shape[:-1], dtype=np.float32)
        return fn(uv, depth=depth, intrinsics=intrinsics, extrinsics=extrinsics)
    return fn(uv, intrinsics=intrinsics, extrinsics=extrinsics)


def project_cv(points: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray):
    return _u3d_np.project_cv(points, extrinsics=extrinsics, intrinsics=intrinsics)


def intrinsics_from_fov(*, fov_x=None, fov_y=None, fov_max=None, fov_min=None, aspect_ratio=None):
    return _u3d_np.intrinsics_from_fov(
        fov_x=fov_x,
        fov_y=fov_y,
        fov_max=fov_max,
        fov_min=fov_min,
        aspect_ratio=aspect_ratio,
    )


def intrinsics_to_fov(intrinsics: np.ndarray):
    return _u3d_np.intrinsics_to_fov(intrinsics)


def extrinsics_look_at(eye: np.ndarray, look_at: np.ndarray, up: np.ndarray):
    eye_arr = np.asarray(eye, dtype=np.float32)
    look_at_arr = np.asarray(look_at, dtype=np.float32)
    up_arr = np.broadcast_to(np.asarray(up, dtype=np.float32), look_at_arr.shape).copy()

    direction = look_at_arr - eye_arr
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
    up_norm = up_arr / np.maximum(np.linalg.norm(up_arr, axis=-1, keepdims=True), 1e-8)
    collinear = np.abs(np.sum(direction * up_norm, axis=-1)) > 0.999
    if np.any(collinear):
        up_arr[collinear] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    return _u3d_np.extrinsics_look_at(eye, look_at, up_arr)


def depth_edge(depth: np.ndarray, rtol: float = 0.05) -> np.ndarray:
    if hasattr(_u3d_np, "depth_edge"):
        return _u3d_np.depth_edge(depth, rtol=rtol)

    depth = np.asarray(depth)
    finite = np.isfinite(depth)
    edge = ~finite

    def mark_discontinuity(a, b):
        valid = np.isfinite(a) & np.isfinite(b)
        denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-6)
        return valid & (np.abs(a - b) > rtol * denom)

    horizontal = mark_discontinuity(depth[..., :, 1:], depth[..., :, :-1])
    edge[..., :, 1:] |= horizontal
    edge[..., :, :-1] |= horizontal

    vertical = mark_discontinuity(depth[..., 1:, :], depth[..., :-1, :])
    edge[..., 1:, :] |= vertical
    edge[..., :-1, :] |= vertical

    return edge
