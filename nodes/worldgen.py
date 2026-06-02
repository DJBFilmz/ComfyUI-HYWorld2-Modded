import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except ImportError:
    folder_paths = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLDGEN_DIR = PROJECT_ROOT / "hyworld2" / "worldgen"
SH_C0 = 0.28209479177387814


def _output_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return PROJECT_ROOT / "output"


def _sanitize_name(value, fallback="scene"):
    base = os.path.basename(str(value or fallback).replace("\\", "/"))
    base = os.path.splitext(base)[0]
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" ._")
    return base or fallback


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_cpu_float_tensor(value):
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().cpu().float()


def _normalize_image_tensor(images):
    images = _as_cpu_float_tensor(images)
    if images is None:
        return None
    if images.dim() == 5 and images.shape[0] == 1 and images.shape[2] in (1, 3, 4):
        images = images[0].permute(0, 2, 3, 1)
    elif images.dim() == 4 and images.shape[1] in (1, 3, 4) and images.shape[-1] not in (1, 3, 4):
        images = images.permute(0, 2, 3, 1)
    if images.dim() != 4:
        return None
    if images.shape[-1] == 1:
        images = images.repeat(1, 1, 1, 3)
    return images[..., :3].clamp(0.0, 1.0).contiguous()


def _normalize_pose_tensor(camera_poses):
    poses = _as_cpu_float_tensor(camera_poses)
    if poses is None:
        return None
    if poses.dim() == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.dim() == 3 and poses.shape[-2:] == (4, 4):
        return poses
    return None


def _normalize_intrinsics_tensor(camera_intrinsics):
    intrs = _as_cpu_float_tensor(camera_intrinsics)
    if intrs is None:
        return None
    if intrs.dim() == 4 and intrs.shape[0] == 1:
        intrs = intrs[0]
    if intrs.dim() == 3 and intrs.shape[-2:] == (3, 3):
        return intrs
    return None


def _normalize_points_tensor(points):
    points = _as_cpu_float_tensor(points)
    if points is None:
        return None
    if points.dim() == 5 and points.shape[0] == 1:
        points = points[0]
    if points.dim() == 4 and points.shape[-1] == 3:
        return points
    if points.dim() == 2 and points.shape[-1] == 3:
        return points
    return None


def _points_and_colors_from_ply_data(ply_data, images=None, prefer_filtered=False, max_points=0):
    if not isinstance(ply_data, dict):
        raise ValueError("ply_data must be a PLY_DATA dictionary.")

    source = ply_data.get("pts3d_filtered") if prefer_filtered else None
    points = _normalize_points_tensor(source)
    if points is None:
        points = _normalize_points_tensor(ply_data.get("pts3d"))
    if points is None:
        splats = ply_data.get("splats")
        if isinstance(splats, dict):
            means = splats.get("means")
            if isinstance(means, list):
                means = means[0]
            points = _normalize_points_tensor(means)
    if points is None:
        raise ValueError("PLY_DATA has no pts3d/pts3d_filtered/splats.means points.")

    image_tensor = _normalize_image_tensor(images)
    if image_tensor is None:
        image_tensor = _normalize_image_tensor(ply_data.get("images"))

    if points.dim() == 4:
        flat_points = points.reshape(-1, 3)
        if image_tensor is not None and image_tensor.shape[0] == points.shape[0]:
            flat_colors = image_tensor.reshape(-1, 3)
        else:
            flat_colors = torch.full_like(flat_points, 0.5)
    else:
        flat_points = points.reshape(-1, 3)
        flat_colors = torch.full_like(flat_points, 0.5)

    finite = torch.isfinite(flat_points).all(dim=1)
    flat_points = flat_points[finite]
    flat_colors = flat_colors[finite]

    if max_points and max_points > 0 and flat_points.shape[0] > max_points:
        generator = torch.Generator().manual_seed(42)
        idx = torch.randperm(flat_points.shape[0], generator=generator)[:max_points]
        flat_points = flat_points[idx]
        flat_colors = flat_colors[idx]

    colors_u8 = (flat_colors.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return flat_points.numpy().astype(np.float32), colors_u8


def _write_point_ply(path, points, colors):
    path = Path(path)
    _ensure_dir(path.parent)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    n = min(points.shape[0], colors.shape[0])
    points, colors = points[:n], colors[:n]

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices = np.empty(n, dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
    os.replace(tmp_path, path)
    return str(path)


def _save_rgb_image(path, image):
    arr = (image.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_depth16(path, depth):
    depth_np = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth_u16 = depth_np.astype(np.float16).view(np.uint16)
    Image.fromarray(depth_u16).save(path)


def _depths_from_points(points, poses):
    points = _normalize_points_tensor(points)
    poses = _normalize_pose_tensor(poses)
    if points is None or poses is None or points.dim() != 4:
        return None
    if points.shape[0] != poses.shape[0]:
        return None
    depths = []
    for i in range(points.shape[0]):
        pts = points[i].reshape(-1, 3)
        c2w = poses[i]
        w2c = torch.linalg.inv(c2w)
        homog = torch.cat([pts, torch.ones((pts.shape[0], 1), dtype=pts.dtype)], dim=1)
        cam = (homog @ w2c.T)[:, :3]
        depth = cam.norm(dim=1).reshape(points.shape[1], points.shape[2])
        depths.append(depth.clamp_min(0.0))
    return torch.stack(depths, dim=0)


def _normal_maps_to_tensor(normal_maps):
    normals = _normalize_image_tensor(normal_maps)
    if normals is None:
        return None
    return normals.clamp(0.0, 1.0)


def _run_command(command, cwd, env=None):
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    process_env["PYTHONPATH"] = os.pathsep.join([
        str(WORLDGEN_DIR),
        str(PROJECT_ROOT),
        process_env.get("PYTHONPATH", ""),
    ])
    print(f"[WorldGen] Running: {' '.join(map(str, command))}")
    process = subprocess.Popen(
        [str(x) for x in command],
        cwd=str(cwd),
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        print(f"[WorldGen] {line}")
        lines.append(line)
        if len(lines) > 500:
            lines = lines[-500:]
    rc = process.wait()
    if rc != 0:
        raise RuntimeError(f"WorldGen command failed with exit code {rc}.\n" + "\n".join(lines[-80:]))
    return "\n".join(lines[-120:])


class VNCCS_WorldGenExportBankFromPLY:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
                "images": ("IMAGE",),
                "scene_name": ("STRING", {"default": "comfy_worldgen_scene"}),
            },
            "optional": {
                "root_dir": ("STRING", {"default": ""}),
                "result_name": ("STRING", {"default": "worldstereo-memory-dmd"}),
                "global_max_points": ("INT", {"default": 3_000_000, "min": 0, "max": 50_000_000, "step": 100_000}),
                "aligned_max_points": ("INT", {"default": 2_000_000, "min": 0, "max": 50_000_000, "step": 100_000}),
                "write_aligned_from_filtered": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("scene_dir", "bank_dir", "global_pcd", "aligned_pcd", "info")
    FUNCTION = "export_bank"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def export_bank(
        self,
        ply_data,
        images,
        scene_name="comfy_worldgen_scene",
        root_dir="",
        result_name="worldstereo-memory-dmd",
        global_max_points=3_000_000,
        aligned_max_points=2_000_000,
        write_aligned_from_filtered=True,
    ):
        root = Path(root_dir) if root_dir.strip() else _output_root() / "hyworld2_worldgen"
        scene_dir = _ensure_dir(root / _sanitize_name(scene_name))
        bank_dir = _ensure_dir(scene_dir / "render_results" / f"generation_bank_{_sanitize_name(result_name, 'worldstereo-memory-dmd')}")

        global_points, global_colors = _points_and_colors_from_ply_data(
            ply_data, images=images, prefer_filtered=False, max_points=int(global_max_points)
        )
        aligned_points, aligned_colors = _points_and_colors_from_ply_data(
            ply_data, images=images, prefer_filtered=bool(write_aligned_from_filtered), max_points=int(aligned_max_points)
        )

        global_path = _write_point_ply(bank_dir / "global_pcd.ply", global_points, global_colors)
        aligned_path = _write_point_ply(bank_dir / "aligned_pcd.ply", aligned_points, aligned_colors)
        info = {
            "scene_dir": str(scene_dir),
            "bank_dir": str(bank_dir),
            "global_points": int(global_points.shape[0]),
            "aligned_points": int(aligned_points.shape[0]),
            "note": "Official-like generation bank exported from current WorldMirror PLY_DATA; sky_pcd is not generated by this shortcut node.",
        }
        with open(bank_dir / "pcd_info.json", "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)
        return (str(scene_dir), str(bank_dir), global_path, aligned_path, json.dumps(info, indent=2))


class VNCCS_WorldGenBuildGSDataFromWorldMirror:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
                "images": ("IMAGE",),
                "camera_poses": ("TENSOR",),
                "camera_intrinsics": ("TENSOR",),
                "scene_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "depth_maps": ("IMAGE",),
                "normal_maps": ("IMAGE",),
                "out_name": ("STRING", {"default": "gs_data"}),
                "points_max": ("INT", {"default": 3_000_000, "min": 0, "max": 50_000_000, "step": 100_000}),
                "write_normals": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("gs_data_dir", "info")
    FUNCTION = "build_gs_data"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def build_gs_data(
        self,
        ply_data,
        images,
        camera_poses,
        camera_intrinsics,
        scene_dir="",
        depth_maps=None,
        normal_maps=None,
        out_name="gs_data",
        points_max=3_000_000,
        write_normals=True,
    ):
        scene = Path(scene_dir) if scene_dir.strip() else _output_root() / "hyworld2_worldgen" / "comfy_worldgen_scene"
        gs_dir = _ensure_dir(scene / _sanitize_name(out_name, "gs_data"))
        images_dir = _ensure_dir(gs_dir / "images")
        depths_dir = _ensure_dir(gs_dir / "depths")
        normals_dir = _ensure_dir(gs_dir / "normals")

        image_tensor = _normalize_image_tensor(images)
        poses = _normalize_pose_tensor(camera_poses)
        intrs = _normalize_intrinsics_tensor(camera_intrinsics)
        if image_tensor is None:
            raise ValueError("images must be a ComfyUI IMAGE tensor.")
        if poses is None or intrs is None:
            raise ValueError("camera_poses and camera_intrinsics must be valid tensors.")
        n = min(image_tensor.shape[0], poses.shape[0], intrs.shape[0])
        if n <= 0:
            raise ValueError("No frames available for gs_data.")

        pts_grid = _normalize_points_tensor(ply_data.get("pts3d") if isinstance(ply_data, dict) else None)
        computed_depths = _depths_from_points(pts_grid, poses)
        normal_tensor = _normal_maps_to_tensor(normal_maps) if write_normals else None

        cameras = {}
        for i in range(n):
            name = f"frame_{i:06d}"
            _save_rgb_image(images_dir / f"{name}.png", image_tensor[i])
            if computed_depths is not None and i < computed_depths.shape[0]:
                _save_depth16(depths_dir / f"{name}.png", computed_depths[i].numpy())
            elif depth_maps is not None:
                depth_img = _normalize_image_tensor(depth_maps)
                if depth_img is not None and i < depth_img.shape[0]:
                    approx = depth_img[i, ..., 0].numpy().astype(np.float32)
                    _save_depth16(depths_dir / f"{name}.png", approx)
            if normal_tensor is not None and i < normal_tensor.shape[0]:
                normal_arr = (normal_tensor[i].numpy() * 255.0 + 0.5).astype(np.uint8)
                Image.fromarray(normal_arr).save(normals_dir / f"{name}.png")

            w2c = torch.linalg.inv(poses[i]).numpy().tolist()
            cameras[name] = {
                "extrinsic": w2c,
                "intrinsic": intrs[i].numpy().tolist(),
            }

        with open(gs_dir / "cameras.json", "w", encoding="utf-8") as handle:
            json.dump(cameras, handle, indent=2)

        points, colors = _points_and_colors_from_ply_data(
            ply_data, images=images, prefer_filtered=False, max_points=int(points_max)
        )
        points_path = _write_point_ply(gs_dir / "points.ply", points, colors)
        meta = {
            "source": "ComfyUI WorldMirror shortcut",
            "frames": n,
            "points": int(points.shape[0]),
            "depth_source": "pts3d camera distance" if computed_depths is not None else "depth_maps image fallback",
            "points_path": points_path,
        }
        with open(gs_dir / "meta_info.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        return (str(gs_dir), json.dumps(meta, indent=2))


class VNCCS_WorldGenRunOfficialGSData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "result_name": ("STRING", {"default": "worldstereo-memory-dmd"}),
                "out_name": ("STRING", {"default": "gs_data"}),
                "nproc_per_node": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "save_normal": ("BOOLEAN", {"default": True}),
                "split_sky": ("BOOLEAN", {"default": True}),
                "split_align": ("BOOLEAN", {"default": False}),
                "extra_args": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("gs_data_dir", "log")
    FUNCTION = "run_gen_gs_data"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def run_gen_gs_data(
        self,
        scene_dir,
        result_name="worldstereo-memory-dmd",
        out_name="gs_data",
        nproc_per_node=1,
        save_normal=True,
        split_sky=True,
        split_align=False,
        extra_args="",
    ):
        scene = Path(scene_dir)
        if not scene.exists():
            raise FileNotFoundError(f"scene_dir not found: {scene}")
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(int(nproc_per_node)),
            "gen_gs_data.py",
            "--root_path", str(scene),
            "--out_name", out_name,
            "--result_name", result_name,
        ]
        if save_normal:
            cmd.append("--save_normal")
        if split_sky:
            cmd.append("--split_sky")
        if split_align:
            cmd.append("--split_align")
        if extra_args.strip():
            cmd.extend(extra_args.split())
        log = _run_command(cmd, WORLDGEN_DIR)
        return (str(scene / out_name), log)


class VNCCS_WorldGenTrain3DGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_data_dir": ("STRING", {"default": ""}),
                "result_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "max_steps": ("INT", {"default": 1500, "min": 1, "max": 100000, "step": 100}),
                "save_steps": ("STRING", {"default": "1500"}),
                "eval_steps": ("STRING", {"default": "1500"}),
                "ply_steps": ("STRING", {"default": "1500"}),
                "save_ply": ("BOOLEAN", {"default": True}),
                "disable_video": ("BOOLEAN", {"default": True}),
                "disable_viewer": ("BOOLEAN", {"default": True}),
                "depth_loss": ("BOOLEAN", {"default": True}),
                "normal_loss": ("BOOLEAN", {"default": True}),
                "use_scale_regularization": ("BOOLEAN", {"default": True}),
                "antialiased": ("BOOLEAN", {"default": True}),
                "extra_args": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result_dir", "log")
    FUNCTION = "train"
    CATEGORY = "VNCCS/WorldGen"
    OUTPUT_NODE = True

    def train(
        self,
        gs_data_dir,
        result_dir,
        max_steps=1500,
        save_steps="1500",
        eval_steps="1500",
        ply_steps="1500",
        save_ply=True,
        disable_video=True,
        disable_viewer=True,
        depth_loss=True,
        normal_loss=True,
        use_scale_regularization=True,
        antialiased=True,
        extra_args="",
    ):
        data_dir = Path(gs_data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"gs_data_dir not found: {data_dir}")
        out_dir = Path(result_dir) if result_dir.strip() else data_dir.parent / "gs_results"
        _ensure_dir(out_dir)

        cmd = [
            sys.executable, "-m", "world_gs_trainer", "default",
            "--data_dir", str(data_dir),
            "--result_dir", str(out_dir),
            "--max_steps", str(int(max_steps)),
            "--save_steps", *save_steps.split(),
            "--eval_steps", *eval_steps.split(),
            "--ply_steps", *ply_steps.split(),
        ]
        if save_ply:
            cmd.append("--save_ply")
        if disable_video:
            cmd.append("--disable_video")
        if disable_viewer:
            cmd.append("--disable_viewer")
        if depth_loss:
            cmd.append("--depth_loss")
        if normal_loss:
            cmd.append("--normal_loss")
        if use_scale_regularization:
            cmd.append("--use_scale_regularization")
        if antialiased:
            cmd.append("--antialiased")
        if extra_args.strip():
            cmd.extend(extra_args.split())

        log = _run_command(cmd, WORLDGEN_DIR)
        return (str(out_dir), log)


NODE_CLASS_MAPPINGS = {
    "VNCCS_WorldGenExportBankFromPLY": VNCCS_WorldGenExportBankFromPLY,
    "VNCCS_WorldGenBuildGSDataFromWorldMirror": VNCCS_WorldGenBuildGSDataFromWorldMirror,
    "VNCCS_WorldGenRunOfficialGSData": VNCCS_WorldGenRunOfficialGSData,
    "VNCCS_WorldGenTrain3DGS": VNCCS_WorldGenTrain3DGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_WorldGenExportBankFromPLY": "WorldGen Export Generation Bank",
    "VNCCS_WorldGenBuildGSDataFromWorldMirror": "WorldGen Build GS Data From WorldMirror",
    "VNCCS_WorldGenRunOfficialGSData": "WorldGen Run Official gen_gs_data",
    "VNCCS_WorldGenTrain3DGS": "WorldGen Train Native 3DGS",
}

