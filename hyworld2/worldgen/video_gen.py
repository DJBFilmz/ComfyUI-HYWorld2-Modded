import argparse
import gc
import json
import os
from glob import glob
from pathlib import Path

import imagesize
import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils import export_to_video
from moge.model.v2 import MoGeModel
from torch.distributed.device_mesh import init_device_mesh
from tqdm import tqdm
from transformers import Sam3VideoModel, Sam3VideoProcessor

from models.worldstereo_wrapper import WorldStereo
from src.data_utils import sort_trajs, load_mutli_traj_dataset
from src.distributed_compat import distributed_backend
from src.general_utils import set_seed, load_video, rank0_log, Timer
from src.retrieval_wm import PanoramaMemoryBank
from src.sp_utils.parallel_states import initialize_parallel_state

os.environ["TOKENIZERS_PARALLELISM"] = "false"
timer = Timer()

SAM3_REPO_ID = "MIUProject/sam3"
MOGE_ID = "Ruicheng/moge-2-vitl-normal"


def resolve_moge_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        for name in ("model.pt", "model.pth", "pytorch_model.bin", "model.safetensors"):
            checkpoint = os.path.join(path, name)
            if os.path.isfile(checkpoint):
                return checkpoint
    return path


def get_pipeline_execution_device(pipeline, fallback_device):
    execution_device = getattr(pipeline, "_execution_device", None)
    if callable(execution_device):
        execution_device = execution_device()
    if execution_device is None:
        execution_device = fallback_device
    return torch.device(execution_device)


def free_pipeline_offload_hooks(pipeline, context):
    if not hasattr(pipeline, "maybe_free_model_hooks"):
        return
    try:
        pipeline.maybe_free_model_hooks()
    except Exception as exc:
        rank0_log(f"Offload hook cleanup skipped after {context} ({type(exc).__name__}: {exc})")


def load_traj_prompt(render_root, view_id, traj_id):
    caption_path = Path(render_root) / view_id / traj_id / "traj_caption.json"
    with open(caption_path, "r", encoding="utf-8") as handle:
        return json.load(handle)["prompt"]


def encode_prompt_cache(pipeline, prompt, negative_prompt, do_classifier_free_guidance, device):
    execution_device = get_pipeline_execution_device(pipeline, device)
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=prompt if prompt else "",
            negative_prompt=negative_prompt if negative_prompt else None,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=execution_device,
        )
    free_pipeline_offload_hooks(pipeline, "prompt encode")
    prompt_embeds = prompt_embeds.detach().to("cpu")
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.detach().to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompt_embeds, negative_prompt_embeds


def build_prompt_cache(worldstereo, scene, render_list, model_type, device):
    prompt_cache = {}
    negative_prompt = worldstereo.cfg.get("negative_prompt", "")
    do_cfg = model_type != "worldstereo-memory-dmd"
    render_root = Path(scene) / "render_results"
    for render_path in render_list:
        render_parts = Path(render_path).parts
        view_id, traj_id = render_parts[-3], render_parts[-2]
        prompt = load_traj_prompt(render_root, view_id, traj_id)
        prompt_cache[(view_id, traj_id)] = encode_prompt_cache(
            worldstereo.pipeline,
            prompt,
            negative_prompt,
            do_classifier_free_guidance=do_cfg,
            device=device,
        )
    return prompt_cache


def move_memory_aux_models(memory_bank, target_device):
    moved = []
    for attr in ("moge_model", "sam3_model"):
        model = getattr(memory_bank, attr, None)
        if model is not None:
            model.to(target_device)
            moved.append(attr)
    if moved:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rank0_log(f"Moved memory auxiliary model(s) to {target_device}: {', '.join(moved)}")


def worldstereo_keyframe_indices(num_frames, device=None):
    keyframe_count = max(1, (int(num_frames) - 1) // 4 + 1)
    indices = torch.linspace(0, int(num_frames) - 1, keyframe_count, device=device).round().long()
    return torch.unique_consecutive(indices.clamp(0, int(num_frames) - 1))


def slice_render_conditioning_to_keyframes(pipeline_kwargs):
    render_video = pipeline_kwargs.get("render_video")
    num_frames = int(pipeline_kwargs.get("num_frames") or 0)
    if not isinstance(render_video, torch.Tensor) or num_frames <= 0:
        return
    keyframe_indices = worldstereo_keyframe_indices(num_frames, device=render_video.device)
    if render_video.shape[2] == keyframe_indices.numel():
        return

    old_frames = int(render_video.shape[2])
    pipeline_kwargs["render_video"] = render_video.index_select(2, keyframe_indices).contiguous()
    render_mask = pipeline_kwargs.get("render_mask")
    if isinstance(render_mask, torch.Tensor) and render_mask.shape[2] == old_frames:
        pipeline_kwargs["render_mask"] = render_mask.index_select(2, keyframe_indices.to(render_mask.device)).contiguous()
    camera_embedding = pipeline_kwargs.get("camera_embedding")
    if isinstance(camera_embedding, torch.Tensor) and camera_embedding.shape[2] == old_frames:
        pipeline_kwargs["camera_embedding"] = camera_embedding.index_select(2, keyframe_indices.to(camera_embedding.device)).contiguous()
    camera_qt = pipeline_kwargs.get("camera_qt")
    if isinstance(camera_qt, torch.Tensor) and camera_qt.shape[1] == old_frames:
        pipeline_kwargs["camera_qt"] = camera_qt.index_select(1, keyframe_indices.to(camera_qt.device)).contiguous()
    ref_index = pipeline_kwargs.get("ref_index")
    max_ref_index = max(0, keyframe_indices.numel() - 2)
    if isinstance(ref_index, torch.Tensor) and ref_index.numel() > 0 and max_ref_index < 19:
        remapped_ref_index = torch.round(ref_index.float() * (float(max_ref_index) / 19.0)).long()
        pipeline_kwargs["ref_index"] = remapped_ref_index.clamp_(0, max_ref_index)
        rank0_log(
            "Remapped ref_index for shortened keyframe timeline: "
            f"max {int(ref_index.max().item())} -> {int(pipeline_kwargs['ref_index'].max().item())}"
        )
    rank0_log(f"Render VAE conditioning sliced to keyframes: {old_frames} -> {pipeline_kwargs['render_video'].shape[2]}")


def sample_camera_tensors_to_frame_count(w2cs, Ks, frame_count):
    frame_count = int(frame_count)
    if frame_count <= 0 or w2cs.shape[0] == frame_count:
        return w2cs, Ks
    indices = np.linspace(0, w2cs.shape[0] - 1, frame_count, dtype=int)
    indices = torch.as_tensor(indices, dtype=torch.long, device=w2cs.device)
    return w2cs.index_select(0, indices), Ks.index_select(0, indices)


if __name__ == '__main__':
    # == parse configs ==
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="worldstereo-memory-dmd", choices=["worldstereo-memory", "worldstereo-memory-dmd"],
                        help="Model type (e.g., 'worldstereo-memory', 'worldstereo-memory-dmd')")
    parser.add_argument("--target_path", default=None, type=str, help="target path")
    parser.add_argument("--align_nframe", default=8, type=int, help="align downsample nframe")
    parser.add_argument("--nframe", default=0, type=int, help="Override WorldStereo video frame count; 0 keeps model config")
    parser.add_argument("--max_reference", default=8, type=int, help="max reference number")
    parser.add_argument("--downsampled_pts", default=2_000_000, type=int, help="Downsampled points number")
    parser.add_argument("--kb_anomaly_percentile", default=90, type=float, help="alignment anoamly percentile")
    parser.add_argument("--pcd_nb_neighbors", default=10, type=int, help="pointcloud filtering number of neighbors")
    parser.add_argument("--pcd_std_ratio", default=2.0, type=float, help="pointcloud filtering std ratio")
    parser.add_argument("--pretrained_path", type=str, default="hanshanxue/WorldStereo",
                        help="WorldStereo repo id or local parent directory containing the model_type subfolder")
    parser.add_argument("--single_model_path", type=str, default="",
                        help="Optional HYWorld2 single-transformer checkpoint from the ComfyUI WorldStereoLight loader")
    parser.add_argument("--moge_path", type=str, default=MOGE_ID,
                        help="MoGe repo id or local checkpoint directory")
    parser.add_argument("--sam3_path", type=str, default=SAM3_REPO_ID,
                        help="SAM3 repo id or local checkpoint directory")
    parser.add_argument("--local_files_only", action="store_true", help="If True, avoid downloading the file and return the path to the local cached file if it exists.")
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP model sharding")
    parser.add_argument("--skip_exist", action="store_true", help="skip existing videos")
    parser.add_argument("--seed", default=1024, type=int, help="Random seed")

    args = parser.parse_args()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=distributed_backend(),
        rank=rank,
        world_size=world_size,
    )
    device_num = torch.cuda.device_count()
    mesh_size = (world_size // device_num, device_num)
    mesh_dims = ("rep", "shard")
    device_mesh = init_device_mesh("cuda", mesh_size, mesh_dim_names=mesh_dims)

    # == init logger ==
    rank0_log(f"World size: {world_size}")

    # == init SP ==
    parallel_dims = initialize_parallel_state(sp=world_size)
    sp_enabled = parallel_dims.sp_enabled
    sp_size = parallel_dims.sp if sp_enabled else 1
    sp_rank = parallel_dims.sp_rank if sp_enabled else 0
    data_rank = dist.get_rank() // sp_size
    data_world_size = dist.get_world_size() // sp_size
    global_seed = args.seed + data_rank
    set_seed(global_seed)
    print(f"Global rank:{dist.get_rank()}, Local rank:{local_rank}, SP_rank:{sp_rank}, SP_group:{data_rank}, seed:{global_seed}.")

    # == setup WorldStereo first so prompt embeddings can be cached before MoGe/SAM/memory bank occupy VRAM ==
    if args.single_model_path:
        worldstereo = WorldStereo.from_single_transformer(
            args.pretrained_path,
            args.single_model_path,
            subfolder=args.model_type,
            local_files_only=args.local_files_only,
            sp_world_size=sp_size,
            fsdp=args.fsdp,
            device_mesh=device_mesh,
            device=device,
            model_device="cpu",
        )
    else:
        worldstereo = WorldStereo.from_pretrained(
            args.pretrained_path,
            subfolder=args.model_type,
            local_files_only=args.local_files_only,
            sp_world_size=sp_size,
            fsdp=args.fsdp,
            device_mesh=device_mesh,
            device=device,
        )
    if torch.cuda.is_available():
        worldstereo.pipeline.enable_sequential_cpu_offload()
    if int(args.nframe) > 0:
        worldstereo.cfg.nframe = int(args.nframe)
        rank0_log(f"Override WorldStereo nframe: {worldstereo.cfg.nframe}")
    dist.barrier()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(torch.float)

    moge_model = None
    sam3_model = None
    sam3_processor = None
    rank0_log("WorldStereo init over...")

    # Auto-select autocast precision: prefer bf16, then fp16, fall back to fp32 (disable autocast)
    if torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif torch.cuda.get_device_capability(device)[0] >= 7:  # fp16 requires SM >= 70
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None  # no half-precision support, fall back to fp32
    rank0_log(f"Autocast dtype: {autocast_dtype if autocast_dtype else 'disabled (fp32)'}")

    # load data
    if os.path.exists(f"{args.target_path}/panorama.png"):
        scene_list = [args.target_path]  # single path VLM inference
    else:
        scene_list = glob(f"{args.target_path}/*")
    scene_list.sort()
    rank0_log(f"Building dataset. {len(scene_list)} scenes found.")

    # == evaluation ==
    with torch.no_grad():
        for scene in tqdm(scene_list):
            scene_name = os.path.basename(scene)
            rank0_log(f"Processing scene {scene_name}.")
            scene_type = json.load(open(f"{scene}/meta_info.json"))["scene_type"]

            # Generation order: (view*_up-->left-->right)-->wonder0,1,2...-->iter*
            with timer.track("Sorting trajectories"):
                render_list = sort_trajs(f"{scene}/render_results")

            rank0_log(f"Scene {Path(scene).name}: {len(render_list)} renderings found.")

            if os.path.exists(f"{scene}/render_results/generation_bank_{args.model_type}/aligned_pcd.ply") and args.skip_exist:
                rank0_log(f"Scene {Path(scene).name}: aligned_pcd.ply exists, skip.")
                continue

            with timer.track("Prompt embedding cache"):
                prompt_cache = build_prompt_cache(worldstereo, scene, render_list, args.model_type, device)

            if moge_model is None:
                with timer.track("MoGe/SAM3 model initialization"):
                    moge_model = MoGeModel.from_pretrained(resolve_moge_checkpoint(args.moge_path)).to(device)
                    sam3_model = Sam3VideoModel.from_pretrained(
                        args.sam3_path,
                        local_files_only=args.local_files_only,
                    ).to(device, dtype=torch.bfloat16)
                    sam3_processor = Sam3VideoProcessor.from_pretrained(
                        args.sam3_path,
                        local_files_only=args.local_files_only,
                    )

            width, height = imagesize.get(str(Path(render_list[0]).parent.parent / "start_frame.png"))
            rank0_log("Enable memory control, initializing memory bank.")
            with timer.track("[IO] Memory Bank Initialization"):
                memory_bank = PanoramaMemoryBank(root_path=scene, image_width=width, image_height=height, device=device, nframe=worldstereo.cfg.nframe,
                                                 max_reference=args.max_reference, align_nframe=args.align_nframe, rank=sp_rank, world_size=sp_size, moge_model=moge_model,
                                                 sam3_model=sam3_model, sam3_processor=sam3_processor, results_name=args.model_type, valid_threshold=0.15, pts_num=args.downsampled_pts,
                                                 kb_anomaly_percentile=args.kb_anomaly_percentile, pcd_nb_neighbors=args.pcd_nb_neighbors, pcd_std_ratio=args.pcd_std_ratio)
            move_memory_aux_models(memory_bank, "cpu")

            for render_path in render_list:
                with timer.track("[IO] Loading cameras"):
                    render_parts = Path(render_path).parts
                    view_id, traj_id = render_parts[-3], render_parts[-2]
                    rank0_log(f"Scene {scene_name}: view: {view_id}, traj: {traj_id}.")

                    target_cameras = json.load(open(f"{scene}/render_results/{view_id}/{traj_id}/camera.json"))
                    tar_w2cs = torch.from_numpy(np.array(target_cameras["extrinsic"])).to(dtype=torch.float32, device=device)
                    tar_Ks = torch.from_numpy(np.array(target_cameras["intrinsic"])).to(dtype=torch.float32, device=device)

                    if args.skip_exist and os.path.exists(f"{scene}/render_results/{view_id}/{traj_id}/{args.model_type}_result.mp4"):
                        if memory_bank is not None:  # Only update the memory bank
                            gen_frames = load_video(f"{scene}/render_results/{view_id}/{traj_id}/{args.model_type}_result.mp4")
                            update_w2cs, update_Ks = sample_camera_tensors_to_frame_count(tar_w2cs, tar_Ks, len(gen_frames))
                            memory_bank.update_memory(gen_frames=gen_frames, tar_w2cs_full=update_w2cs, tar_Ks_full=update_Ks, view_id=view_id, traj_id=traj_id)
                        continue

                # All ranks run retrieval; sequence-parallel rendering happens inside.
                with timer.track("Memory Retrieval"):
                    retrieved_frames, ref_index, ref_index_dict, ref_w2cs, _ = memory_bank.retrieval(tar_w2cs, tar_Ks, view_id=view_id, traj_id=traj_id)
                    combined_frames = retrieved_frames / 255
                if rank == 0:  # Rank 0 saves retrieval results
                    with timer.track("[IO] Save Memory retrieval results"):
                        os.makedirs(f"{scene}/render_results/{view_id}/{traj_id}/memory_inputs", exist_ok=True)
                        export_to_video(combined_frames, f"{scene}/render_results/{view_id}/{traj_id}/memory_inputs/{args.model_type}.mp4", fps=16)
                        if ref_index_dict is not None:
                            with open(f"{scene}/render_results/{view_id}/{traj_id}/memory_inputs/{args.model_type}_ref_index.json", "w") as w:
                                json.dump(ref_index_dict, w, indent=2)
                        if ref_w2cs is not None:
                            ref_w2cs = ref_w2cs.cpu().numpy().tolist()
                            with open(f"{scene}/render_results/{view_id}/{traj_id}/memory_inputs/{args.model_type}_ref_w2cs.json", "w") as w:
                                json.dump(ref_w2cs, w, indent=2)

                dist.barrier()
                with timer.track("[IO] Loading meta inputs"):
                    meta_data = load_mutli_traj_dataset(cfg=worldstereo.cfg, input_path=f"{scene}/render_results", output_path=f"{scene}/render_results",
                                                        view_id=view_id, traj_id=traj_id, device=device, ref_index=ref_index, model_type=args.model_type, task_type="panorama")

                # ==== Pipline Inputs ====
                pipeline_kwargs = {k: v for k, v in meta_data.items() if v is not None}
                pipeline_kwargs.update(
                    generator=generator,
                    output_type="pt",
                    latent_cond_mode=worldstereo.cfg.latent_cond_mode,
                )
                cached_prompt_embeds, cached_negative_prompt_embeds = prompt_cache[(view_id, traj_id)]
                pipeline_kwargs.pop("prompt", None)
                pipeline_kwargs.update(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=cached_prompt_embeds.to(device),
                    negative_prompt_embeds=(
                        cached_negative_prompt_embeds.to(device)
                        if cached_negative_prompt_embeds is not None
                        else None
                    ),
                )

                if args.model_type == "worldstereo-memory-dmd":
                    pipeline_kwargs["mode"] = "test"
                    slice_render_conditioning_to_keyframes(pipeline_kwargs)
                else:
                    pipeline_kwargs["guidance_scale"] = 5.0

                # pipeline inference
                with timer.track("Video Model Inference"), torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    output = worldstereo.pipeline(**pipeline_kwargs).frames[0].float()

                gc.collect()
                torch.cuda.empty_cache()

                if dist.get_rank() % sp_size == 0:
                    with timer.track("[IO] Save Results"):
                        # [f,c,h,w]->[f,h,w,c]
                        output = output.permute(0, 2, 3, 1).cpu().numpy()
                        export_to_video(output, f"{scene}/render_results/{view_id}/{traj_id}/{args.model_type}_result.mp4", fps=16)
                dist.barrier()

                # update memory bank
                if memory_bank is not None:
                    with timer.track("[IO] Reload results for memory update* (need to be optimized)"):
                        gen_frames = load_video(f"{scene}/render_results/{view_id}/{traj_id}/{args.model_type}_result.mp4")
                    update_w2cs, update_Ks = sample_camera_tensors_to_frame_count(tar_w2cs, tar_Ks, len(gen_frames))
                    memory_bank.update_memory(gen_frames=gen_frames, tar_w2cs_full=update_w2cs, tar_Ks_full=update_Ks, view_id=view_id, traj_id=traj_id)
                dist.barrier()

            if memory_bank is not None:
                with timer.track("Run World Mirror"):
                    memory_bank.apply_worldmirror(skip_exist=True)
                dist.barrier()

                move_memory_aux_models(memory_bank, device)
                with timer.track("Memory bank Alignment"):
                    memory_bank.alignment(debug_mode=False)
                dist.barrier()

                # memory bank over, export pcd
                with timer.track("[IO] Save final aligned pointcloud (update memory)"):
                    memory_bank.export_pcd(f"{memory_bank.root_path}/render_results/generation_bank_{args.model_type}", N_points=args.downsampled_pts)
                dist.barrier()

            if rank == 0:
                timer.summary()

    if dist.is_initialized():
        dist.destroy_process_group()
