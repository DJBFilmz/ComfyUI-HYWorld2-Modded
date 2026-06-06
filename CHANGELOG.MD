# Changelog

## 1.5.1

### Added

- Added `HYWorld2 Klein World Expansion`, a memory-bank expansion node that runs a ComfyUI Flux/Klein-style reference-latent pipeline over selected HYWorld2 trajectory keyframes and writes reusable per-trajectory result videos.
- Added Klein expansion caching/state tracking via `hyworld2_klein_expansion_state.json`, including seed/settings reuse for previously generated result videos.
- Added memory-bank normalization for Klein outputs so generated frames, camera intrinsics, and memory-bank point grids stay aligned when the output resolution differs from the original trajectory render size.
- Added `HYWorld2 Dataset Frame Pairs`, a utility node for collecting final/generated frames, matching source render frames, and previous-frame context images from HYWorld2 workspaces for dataset and LoRA preparation workflows.
- Registered the new HYWorld2 utility nodes through the main node mappings.

### Changed

- Updated the package version to `1.5.1`.
- Updated `HYWorld2 Trajectories` so trajectory rendering only prepares and sorts render outputs; QwenVL caption generation is deferred to the downstream world-expansion stage.
- Updated HYWorld2 trajectory cache validation so missing trajectory caption files no longer invalidate otherwise complete rendered trajectory workspaces.
- Added evenly spaced keyframe selection for workflows that need a smaller frame subset from WorldStereo-style trajectory renders.

### Fixed

- Reduced `Load WorldStereoLight Model` on-demand Wan2.1 downloads by avoiding unused full Wan text-encoder weights and full VAE weights when single-file WorldStereoLight encoder/VAE assets are used.

## 1.5.0

### Added

- Added the native HYWorld2 world-generation workflow:
  - `HYWorld2 Workspace`
  - `HYWorld2 QwenVL`
  - `HYWorld2 Trajectories`
  - `HYWorld2 Memory Bank`
  - `HYWorld2 World Expansion`
  - `HYWorld2 Prepare WorldMirror Batch`
  - `HYWorld2 Memory Alignment`
  - `HYWorld2 GS Data`
  - `HYWorld2 Train 3DGS`
- Added native HYWorld2 worldgen modules for trajectory planning, trajectory rendering, memory-bank retrieval, WorldStereo scene expansion, GS dataset export, 3DGS training, and mesh export.
- Added Qwen Pano helper nodes for reference/canvas preparation, prompt generation, panorama generation, and seam blending.
- Added local QwenVL support for HYWorld2 planning and trajectory captioning, with 4B instruct and 4-bit quantization defaults.
- Added robust LLM JSON parsing with `json-repair`.
- Added native callable APIs for `traj_generate.py` and `traj_render.py`, allowing HYWorld2 trajectories to run in-process without `runpy`, `sys.argv`, cwd, env, or subprocess-style script execution.
- Added native Windows wheel support for HYWorld2 worldgen dependencies, including the HYWorld gsplat fork, recast/navmesh, PyTorch3D, and fused SSIM.
- Added `Load PLY File` for preview/debug workflows that need to inspect existing PLY files without rerunning the full pipeline.
- Added a complete native world-generation workflow example.

### Changed

- Reworked HYWorld2 workspace, trajectory, expansion, and memory-bank execution around automatic cache detection instead of manual mode switching.
- Updated trajectory generation to reuse existing workspace outputs when the source panorama, settings, and seed match.
- Updated World Expansion to reuse existing generated result videos when the seed and output files match.
- Added seed-aware cache state to HYWorld2 trajectory and expansion stages.
- Added manual `scene_type` selection for HYWorld2 trajectories (`auto`, `indoor`, `outdoor`).
- Reduced automatic scene-type classification input size to a 480px preview image.
- Removed OpenAI/vLLM fallback behavior from trajectory generation; model work is routed through the local HYWorld2 QwenVL path.
- Renamed and clarified the additional navigation trajectory option.
- Added optional extreme-detail trajectory planning for notable environment details, with duplicate removal against the primary object pass.
- Improved SAM3 loading/progress logging during trajectory generation.
- Added `memory_images` output to `HYWorld2 Memory Bank` for inspecting frames passed downstream.
- Added prerequisite checks for incomplete HYWorld2 workspaces before constructing the memory bank.
- Updated `HYWorld2 Memory Alignment` to require connected `raw_splats` depth for alignment instead of internally launching a second WorldMirror reconstruction.
- Added `enable_splat_upsample` to the simple WorldMirror V2 reconstruction node; it remains enabled by default and can be disabled for depth-only workflows.
- Added 3DGS trainer controls for `batch_size`, patch-size presets, and panorama/generated-frame sampling presets.
- Improved 3DGS trainer metadata, camera export, mesh export, and preview compatibility.
- Updated Gaussian preview to handle `worldmirror` and `hyworld2_worldgen` coordinate bases more explicitly.
- Added preview-basis conversion for HYWorld2 trainer PLYs without mutating the original PLY files.
- Updated installer messaging and dependency handling for native HYWorld wheels.

### Fixed

- Fixed HYWorld2 trajectory rendering in single-process ComfyUI runs by avoiding distributed process-group requirements.
- Fixed tensor indexing issues in WorldStereo generation caused by tensor `ref_index` values.
- Fixed trainer batch handling for image IDs, depth tensors, and normal tensors.
- Fixed trainer post-run distributed barrier errors in single-process mode.
- Fixed Gaussian preview loading for HYWorld2 trainer PLYs by avoiding invalid `.splat` cache substitution.
- Fixed HYWorld2 trainer preview basis issues that caused black, upside-down, or incorrectly oriented previews.
- Fixed persistent coordinate-basis widget behavior in the background preview node.
- Fixed ComfyUI cache invalidation warnings caused by `IS_CHANGED` receiving unresolved upstream values.
- Fixed Memory Bank cache diagnostics and made cache-key logging explicit for HYWorld2 workspace, trajectory, and memory-bank nodes.

### Dependencies

- Updated `numpy` requirement to `>=2.0,<2.3`.
- Updated `transformers` requirement to `>=5.2.0`.
- Added HYWorld2 worldgen/runtime dependencies including `json-repair`, `utils3d`, `cupy-cuda13x`, `zim-anything`, trainer utilities, and native build helpers.
- Removed direct PyPI `gsplat` dependency in favor of the vendored HYWorld gsplat fork installed by the project installer.

## 1.4.0

### Added

- Added WorldStereoLight model loading with two selectable light INT4 checkpoints from `MIUProject/VNCCS_WorldStereoLight`.
- Added automatic on-demand downloading for the selected WorldStereoLight checkpoint and metadata.
- Added streaming single-transformer WorldStereo checkpoint loading, including experimental packed INT4 linear support.
- Added bundled WorldStereo workflows for the base-model path and the new WorldStereoLight path.
- Added SeedVR2 upscaling support for WorldStereo output frames.
- Added additional WorldStereo camera trajectory presets and frame handling options.
- Added VAE memory mode controls and conditioning cache support for lower-memory WorldStereo generation.
- Added PyTorch3D installation/build improvements, including a bundled Windows CUDA 13.0 wheel.

### Changed

- Updated the package version to `1.4.0`.
- Reworked the former `Load WorldStereo Single Model` node into `Load WorldStereoLight Model`.
- Simplified the WorldStereoLight loader UI by removing manual model type, precision, and turbo LoRA controls; these are now inferred from checkpoint metadata.
- Updated WorldStereo model loading to better support large local checkpoints, CPU/offload loading paths, and metadata-driven runtime setup.
- Improved WorldStereo pipeline input preparation, image handling, conditioning-frame alignment, and reusable embedding caches.
- Improved WorldStereo ControlNet conditioning token handling and latent-frame alignment.
- Improved CUDA toolkit discovery and native extension build diagnostics in the gsplat/PyTorch3D helper script.
- Updated workflows to use the new WorldStereo model-loading paths.

### Fixed

- Fixed WorldStereo mask conditioning alignment when mask downsampling is disabled.
- Fixed ControlNet input alignment issues in WorldStereo models and pipelines.
- Fixed first-frame/source-frame handling for WorldMirror sequence preparation from WorldStereo outputs.
- Fixed several WorldStereo import, local-model, and optional dependency edge cases around heavy runtime components.

## 1.3.0

### Added

- Added WorldMirror V2 low-VRAM controls for head frame chunking, Gaussian parameter chunking, transformer MLP token chunking, and selectable head compute modes.
- Added a single `low_vram_mode` profile for the clean WorldMirror V2 node, applying the proven low-memory defaults while keeping individual tuning controls in the advanced node.
- Added WorldMirror V2 `depth_only` and `depth+gs` execution modes to skip unnecessary heads when pushing high reconstruction resolutions.
- Added `torchao` as a main dependency for WorldMirror V2 FP8 weight-only quantization.

### Changed

- Updated the package version to `1.3.0`.
- Reworked WorldMirror V2 loading so models are prepared on CPU before CUDA residency, avoiding the old temporary fp32 VRAM spike during load.
- Updated FP8 loading to cast the model to bf16 first, move it to CUDA, and then apply torchao FP8 quantization on the target GPU for compatibility with torchao tensor subclasses.
- Raised WorldMirror V2 `target_size` limits to `4096` for experimental high-resolution runs.
- Updated the clean WorldMirror V2 node to infer dense splat upsample size from the incoming view resolution and hide fixed camera/source/upsample-mode controls.
- Kept dense splat scale, adaptive scale, voxel pruning, and point-cap controls available in the clean node because they directly affect density and file size.
- Kept the advanced WorldMirror V2 node fully manual by exposing all individual low-VRAM and high-resolution tuning controls instead of the grouped `low_vram_mode` profile.
- Reduced WorldMirror V2 head and Gaussian renderer peak memory by chunking DPT head frames and Gaussian parameter-head frames.
- Reduced transformer peak memory by chunking MLP evaluation and using functional linear operations with in-place bias application.
- Updated DPT dense head output handling so Gaussian features are not forced back to float32 before splat generation.
- Updated the bundled WorldMirror panorama workflow for the new high-resolution and low-VRAM defaults.

### Fixed

- Fixed FP8 WorldMirror V2 failures caused by applying torchao quantization before moving the model to CUDA.
- Fixed `model_cpu_offload` incompatibility with torchao FP8 weights by detecting FP8/tensor-subclass models and keeping them GPU-resident.
- Fixed high-resolution WorldMirror V2 OOM pressure from CUDA double-precision positional embedding generation.
- Fixed edge-filter handling when `head_compute_mode` skips normal prediction by reporting the skipped filter instead of assuming normals exist.
- Fixed an inconsistent clean-node UI where grouped low-VRAM mode and its individual internal controls were shown at the same time.

## 1.2.0

### Added

- Added the project `LICENSE` file and updated package metadata to use it.
- Added a bundled demo video asset at `img/video.mov`.
- Added WorldMirror V2 advanced/clean reconstruction flows with expanded controls for camera conditioning, splat camera source, splat color source, native model masks, voxel pruning, splat scale tuning, opacity floor, depth priors, adaptive target sizing, and debug logging.
- Added experimental high-resolution Gaussian splat upsampling from depth backprojection, with depth-adaptive, footprint-adaptive, and hybrid scale modes.
- Added depth-aware point capping for high-resolution splat upsampling to preserve more far-depth splats when reducing point count.
- Added model CPU offload support for WorldMirror V2 inference via `accelerate`.
- Added optional FP8 weight-only quantization support for WorldMirror V2 through `torchao`.
- Added diagnostic output images for WorldMirror V2 point and Gaussian filter masks.
- Added Gaussian `.splat` preview cache generation when saving Gaussian PLY files.
- Added fullscreen flight mode to the Gaussian Splat Viewer, including FPS controls and a dedicated fullscreen toggle.
- Added viewer support for camera pose and intrinsic tensors so previews can initialize from reconstruction camera data.
- Added fallback camera utility implementations for WorldStereo when the upstream camera utility imports fail.

### Changed

- Updated package version to `1.2.0` and refreshed the package description, display name, publisher ID, project URLs, and optional dependencies.
- Reworked WorldMirror V2 model loading to download/load `tencent/HY-World-2.0` from the ComfyUI models directory and apply selected precision after loading.
- Improved WorldMirror V2 inference setup so input camera poses, intrinsics, and optional depth priors are passed through explicit conditioning flags.
- Changed WorldMirror V2 Gaussian splat positioning to support either input camera priors or predicted cameras.
- Changed WorldMirror V2 Gaussian color handling so splats can use source image RGB by default, reducing black spherical-harmonic artifacts.
- Updated WorldMirror V2 geometric filtering so confidence, edge, sky, and native model masks can be combined while splat filtering remains optional.
- Updated Gaussian splat extraction so spherical harmonic tensors are not accidentally collapsed when they already use `[N, SH, 3]` shape.
- Replaced unsafe/native Gaussian PLY saving with a safer writer that validates tensor lengths, filters oversized splats by quantile, writes via temporary files, and avoids fallback writes on disk-full errors.
- Simplified `VNCCS_SavePLY` inputs to always save Gaussian splats without pointcloud/rotation toggles.
- Simplified `VNCCS_BackgroundPreview` inputs by removing the preview width control and replacing legacy extrinsics/intrinsics inputs with `camera_poses` and `camera_intrinsics`.
- Updated `VNCCS_Equirect360ToViews` defaults and controls: pitches now default to `0,-60,60`, output size can go up to `4096`, and sampling mode can be selected between bilinear and bicubic.
- Reworked equirectangular view extraction to use configurable sampling and removed the old distortion-correction/mask output path.
- Added configurable camera pose conventions and pose/intrinsics offsets for panorama-derived camera priors.
- Improved camera pose matrix construction for exported camera priors.
- Improved Gaussian Splat Viewer rendering performance with invalidation-based rendering, lower idle work, viewport scale compensation, and better resize/FOV preservation.
- Improved Gaussian viewer camera pivot behavior for small scenes and fullscreen transitions.
- Scoped WorldStereo runtime monkey patches to the loading/inference block instead of leaving global distributed/diffusers patches active.
- Updated WorldStereo import handling and tests to work in environments where optional heavy dependencies are unavailable.
- Updated FlashAttention import handling to support multiple `flash_attn` package layouts and fall back cleanly when FlashAttention is not installed.
- Updated Gaussian renderer inference so splat positions can use input cameras when available.
- Hardened Panorama Mapper image path resolution so inputs cannot escape the ComfyUI input directory.
- Replaced the large Visual Studio-focused `.gitignore` with a smaller project-oriented ignore file for OS, editor, Python, build, CUDA/native, model, output, and local workflow artifacts.

### Fixed

- Fixed WorldMirror V2 failures caused by offloaded/meta tensors by detecting incompatible model state and reporting a targeted reload/offload error.
- Fixed stale splat weights after voxel merging by removing weights that no longer match the pruned Gaussian count.
- Fixed viewer scaling artifacts where fullscreen or resized previews made Gaussians appear too large.
- Fixed Gaussian Splat Viewer idle performance by rendering only while active or on heartbeat.
- Fixed background preview loading to prefer generated `.splat` cache files when present.
- Fixed unsafe panorama path handling for relative paths containing parent-directory traversal.
- Fixed WorldStereo behavior when `folder_paths` is unavailable.

### Tests

- Updated WorldStereo node tests to import modules more defensively, skip optional Torch-dependent cases when Torch is unavailable, and cover the new fallback camera utility behavior.
