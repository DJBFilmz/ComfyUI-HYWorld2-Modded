# Changelog

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
