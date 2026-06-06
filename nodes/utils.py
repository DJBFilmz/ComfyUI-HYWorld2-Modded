from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except ImportError:
    folder_paths = None


def _output_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return Path(__file__).resolve().parents[1] / "output"


def _hyworld2_workspaces_root(root_dir: str) -> Path:
    value = str(root_dir or "").strip()
    if value:
        root = Path(value)
    else:
        root = _output_root() / "hyworld2_worldgen"
    if root.name == "output":
        root = root / "hyworld2_worldgen"
    return root


def _natural_key(path: Path):
    import re

    parts = []
    for part in path.parts:
        split = re.split(r"(\d+)", part.lower())
        parts.extend(int(piece) if piece.isdigit() else piece for piece in split)
    return parts


def _read_video_rgb(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Video has no readable frames: {path}")
    return frames


def _worldstereo_keyframe_indices(num_frames: int):
    num_frames = int(num_frames)
    if num_frames <= 0:
        return np.asarray([], dtype=np.int64)
    keyframe_count = max(1, (num_frames - 1) // 4 + 1)
    indices = np.rint(np.linspace(0, num_frames - 1, keyframe_count)).astype(np.int64)
    return np.unique(np.clip(indices, 0, num_frames - 1))


def _matching_render_indices(render_count: int, final_count: int):
    render_count = int(render_count)
    final_count = int(final_count)
    if render_count <= 0 or final_count <= 0:
        return np.asarray([], dtype=np.int64)
    if final_count == render_count:
        return np.arange(render_count, dtype=np.int64)
    keyframes = _worldstereo_keyframe_indices(render_count)
    if final_count == len(keyframes):
        return keyframes
    return np.rint(np.linspace(0, render_count - 1, final_count)).astype(np.int64).clip(0, render_count - 1)


def _find_result_video(traj_dir: Path, result_name: str):
    name = str(result_name or "").strip()
    if name and name.lower() not in {"auto", "*"}:
        path = traj_dir / f"{name}_result.mp4"
        return path if path.is_file() and path.stat().st_size > 0 else None
    candidates = [
        path
        for path in traj_dir.glob("*_result.mp4")
        if path.is_file() and path.name not in {"render_result.mp4"} and path.stat().st_size > 0
    ]
    return sorted(candidates, key=_natural_key)[0] if candidates else None


def _find_workspaces(root: Path, workspace_filter: str):
    if (root / "render_results").is_dir():
        return [root]
    workspace_glob = str(workspace_filter or "*").strip() or "*"
    return [path for path in root.glob(workspace_glob) if (path / "render_results").is_dir()]


def _to_tensor(frame):
    return torch.from_numpy(np.asarray(frame, dtype=np.float32) / 255.0)


def _to_image_item(frame):
    return _to_tensor(frame).unsqueeze(0).contiguous()


def _resize_rgb(frame, size):
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))


class HYWorld2DatasetFramePairs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_dir": ("STRING", {"default": ""}),
                "workspace_filter": ("STRING", {"default": "*"}),
                "result_name": ("STRING", {"default": "worldstereo-memory-dmd"}),
                "max_pairs": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("final_frames", "dataset_frames", "context_frames")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "collect"
    CATEGORY = "VNCCS/HYWorld2/Utils"

    @classmethod
    def IS_CHANGED(cls, root_dir="", workspace_filter="*", result_name="worldstereo-memory-dmd", max_pairs=0):
        root = _hyworld2_workspaces_root(root_dir)
        state = [str(root.resolve() if root.exists() else root), str(workspace_filter), str(result_name), str(int(max_pairs))]
        if root.exists():
            for workspace in sorted(_find_workspaces(root, workspace_filter), key=_natural_key):
                for render_path in sorted((workspace / "render_results").glob("**/render.mp4"), key=_natural_key):
                    result_path = _find_result_video(render_path.parent, result_name)
                    if result_path is None:
                        continue
                    state.append(f"{render_path}:{render_path.stat().st_mtime_ns}:{render_path.stat().st_size}")
                    state.append(f"{result_path}:{result_path.stat().st_mtime_ns}:{result_path.stat().st_size}")
        return "|".join(state)

    def collect(self, root_dir="", workspace_filter="*", result_name="worldstereo-memory-dmd", max_pairs=0):
        root = _hyworld2_workspaces_root(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"HYWorld2 workspaces root not found: {root}")

        workspaces = _find_workspaces(root, workspace_filter)
        render_paths = []
        for workspace in sorted(workspaces, key=_natural_key):
            render_paths.extend(sorted((workspace / "render_results").glob("**/render.mp4"), key=_natural_key))

        final_images = []
        dataset_images = []
        context_images = []
        max_pairs = int(max_pairs)
        pair_limit = max_pairs if max_pairs > 0 else None
        first_size = None
        trajectories_used = 0
        skipped = 0

        for render_path in render_paths:
            result_path = _find_result_video(render_path.parent, result_name)
            if result_path is None:
                skipped += 1
                continue

            render_frames = _read_video_rgb(render_path)
            final_frames = _read_video_rgb(result_path)
            indices = _matching_render_indices(len(render_frames), len(final_frames))
            if len(indices) != len(final_frames):
                raise RuntimeError(
                    f"Frame pairing failed for {render_path.parent}: "
                    f"render={len(render_frames)}, final={len(final_frames)}, indices={len(indices)}"
                )

            trajectories_used += 1
            previous_final_frame = None
            for final_frame, render_index in zip(final_frames, indices):
                if first_size is None:
                    first_size = (int(final_frame.shape[1]), int(final_frame.shape[0]))
                final_frame = _resize_rgb(final_frame, first_size)
                render_frame = _resize_rgb(render_frames[int(render_index)], first_size)
                if previous_final_frame is None:
                    context_frame = np.zeros_like(final_frame)
                else:
                    context_frame = previous_final_frame
                final_images.append(_to_image_item(final_frame))
                dataset_images.append(_to_image_item(render_frame))
                context_images.append(_to_image_item(context_frame))
                previous_final_frame = final_frame
                if pair_limit is not None and len(final_images) >= pair_limit:
                    break
            if pair_limit is not None and len(final_images) >= pair_limit:
                break

        if not final_images:
            raise RuntimeError(f"No HYWorld2 dataset frame pairs found under {root}")

        print(
            "[HYWorld2 Dataset Frame Pairs] "
            f"root={root}, workspaces={len(workspaces)}, trajectories={trajectories_used}, "
            f"skipped={skipped}, pairs={len(final_images)}, size={first_size[0]}x{first_size[1]}, output=list"
        )
        return (final_images, dataset_images, context_images)


NODE_CLASS_MAPPINGS = {
    "HYWorld2DatasetFramePairs": HYWorld2DatasetFramePairs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYWorld2DatasetFramePairs": "HYWorld2 Dataset Frame Pairs",
}
