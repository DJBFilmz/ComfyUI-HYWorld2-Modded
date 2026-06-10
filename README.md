# ComfyUI HY-World 2.0 - WorldMirror 3D

This is a modded ver of the original fork. Basically a testing grounds for myself. Feel free to download if you wish, maybe it'll have some things that fit your needs, but... it probably won't. - David J. Buchanan

Original Message Below:



ComfyUI custom nodes for 3D scene reconstruction from a single image, panorama, image set, or video using HY-World 2.0 / WorldMirror.

## Nodes

Category: `VNCCS/3D`

| Node | Description |
|------|-------------|
| `VNCCS_LoadWorldMirrorModel` | Download and load WorldMirror V1 model |
| `VNCCS_WorldMirror3D` | V1 inference: PLY point cloud, depth, normals, Gaussian splat |
| `VNCCS_LoadWorldMirrorV2Model` | Download and load WorldMirror V2 model |
| `VNCCS_WorldMirrorV2_3D` | V2 inference: PLY point cloud, depth, normals, Gaussian splat |
| `VNCCS_WorldMirrorV2_3D_Clean` | V2 reconstruction defaults for cleaner dense splats |
| `VNCCS_PLYSceneRenderer` | Render PLY scenes from arbitrary camera angles |
| `VNCCS_SplatRefiner` | Refine Gaussian splat data |
| `VNCCS_DecomposePLYData` | Extract XYZ / RGB / normals / opacity tensors from PLY |
| `VNCCS_SavePLY` | Save PLY files to disk |
| `VNCCS_Equirect360ToViews` | Extract perspective views from equirectangular panoramas |
| `VNCCS_PanoramaMapper` | Map panoramas to wall / floor / ceiling projections |

## Installation

### Via ComfyUI Manager

Search for **HY-World 2.0** and click Install. `requirements.txt` and `install.py` run automatically.

`install.py` installs the HY-World vendored `gsplat_maskgaussian` fork from:

```text
hyworld2/worldgen/third_party/gsplat_maskgaussian
```

Do not install upstream/PyPI `gsplat` separately. HY-World worldgen and the native scene trainer require fork-only rasterization arguments such as `distloss` and `gauss_masks`; the project must have only one installed Python package named `gsplat`.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AHEKOT/ComfyUI_HYWorld2
cd ComfyUI_HYWorld2
pip install -r requirements.txt
python install.py
```

## HY-World gsplat_maskgaussian

`gsplat_maskgaussian` is a CUDA extension and is built from the vendored HY-World source. It replaces upstream `gsplat` under the same package name.

Required build tools:

- CUDA Toolkit matching the PyTorch CUDA major/minor version as closely as possible.
- MSVC C++ Build Tools on Windows with the Desktop development with C++ workload.
- `ninja`, listed in `requirements.txt`.
- Git, if the vendored GLM headers need to be restored.

Manual rebuild:

```bash
# Windows embedded ComfyUI Python
scripts\pipinstall.bat

# Any Python environment
python scripts/build_gsplat.py
```

Successful installation is verified by a CUDA smoke test that calls `gsplat.rendering.rasterization` with `distloss=True` and `gauss_masks`.

## Workflows

Example workflows are in the `workflows/` directory:

- `World-single-image.json` - single image to 3D scene
- `World-Mirror-panorama.json` - equirectangular panorama to 3D scene

## Requirements

- Python 3.10+
- PyTorch with CUDA
- NVIDIA GPU for gsplat rendering/training
- Flash Attention for HY-World model inference where supported:

```bash
pip install flash-attn --no-build-isolation
```
