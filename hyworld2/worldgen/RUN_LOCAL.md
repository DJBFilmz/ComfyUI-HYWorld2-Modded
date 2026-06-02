# HY-World Native GS Trainer In This Repo

This directory was copied from the official Tencent-Hunyuan/HY-World-2.0 repository.

Important files:

- `world_gs_trainer.py` - native Gaussian Splatting optimizer.
- `gen_gs_data.py` - official GS data preparation script.
- `gs/` - dataset parser, camera normalization, rendering helpers.
- `src/` - panorama/video/data utility helpers used by data preparation.

The trainer expects a HY-World `gs_data` dataset, not generic COLMAP:

```text
gs_data/
  images/
    frame_0000.png
  depths/
    frame_0000.png
  normals/
    frame_0000.png
  cameras.json
  points.ply
  meta_info.json
```

Depth priors must be encoded as float16 bit-pattern PNGs:

```python
depth_uint16 = depth_float32.astype(np.float16).view(np.uint16)
Image.fromarray(depth_uint16).save(path)
```

`cameras.json` extrinsics are expected as world-to-camera (`w2c`) matrices.

Recommended dependency install, avoiding source builds:

```powershell
& 'F:\ComfyUI-Easy-Install\ComfyUI-Easy-Install\python_embeded\python.exe' -m pip install --only-binary=:all: tyro viser nerfview torchmetrics pytorch-msssim splines imagesize
```

Run from this directory:

```powershell
cd e:\Development\ComfyUI_HYWorld2\hyworld2\worldgen
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
```

Smoke test:

```powershell
& 'F:\ComfyUI-Easy-Install\ComfyUI-Easy-Install\python_embeded\python.exe' -m world_gs_trainer default `
  --data_dir e:\path\to\gs_data `
  --result_dir e:\path\to\gs_results_smoke `
  --max_steps 1 `
  --save_steps 1 `
  --eval_steps 1 `
  --ply_steps 1 `
  --save_ply `
  --disable_video `
  --disable_viewer `
  --use_scale_regularization `
  --antialiased `
  --depth_loss `
  --normal_loss
```

Full optimization starter:

```powershell
& 'F:\ComfyUI-Easy-Install\ComfyUI-Easy-Install\python_embeded\python.exe' -m world_gs_trainer default `
  --data_dir e:\path\to\gs_data `
  --result_dir e:\path\to\gs_results `
  --max_steps 8000 `
  --save_steps 2000 4000 6000 8000 `
  --eval_steps 2000 4000 6000 8000 `
  --ply_steps 1000 2000 3000 4000 5000 6000 7000 8000 `
  --save_ply `
  --disable_video `
  --disable_viewer `
  --use_scale_regularization `
  --antialiased `
  --depth_loss `
  --normal_loss `
  --strategy.refine-start-iter 150 `
  --strategy.refine-stop-iter 750 `
  --strategy.refine-every 100 `
  --strategy.refine-scale2d-stop-iter 750 `
  --strategy.reset-every 99990 `
  --strategy.grow-grad2d 0.0001 `
  --strategy.prune-scale3d 0.1
```
