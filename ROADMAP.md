# HY-World 2.0 ComfyUI Roadmap

This roadmap tracks the local ComfyUI integration status for the HY-World 2.0 world generation pipeline.

## Scope

The official early worldgen stages are intentionally skipped for now because the current ComfyUI workflow already covers part of that path:

- [ ] `traj_generate.py` - panorama depth, global PCD/mesh, VLM/navmesh trajectory planning.
- [ ] `traj_render.py` - render `global_pcd.ply` into trajectory videos and masks.
- [ ] `video_gen.py` - WorldStereo 2.0 video generation and official memory-bank WorldMirror invocation.

## Current Workflow

- [x] Panorama input workflow exists: `workflows/World-Mirror-panorama.json`.
- [x] Panorama splitting exists via `VNCCS_Equirect360ToViews`.
- [x] WorldMirror V2 reconstruction exists via `VNCCS_WorldMirrorV2_3D`.
- [x] Gaussian/PLY preview path exists via `VNCCS_SavePLY` and `VNCCS_BackgroundPreview`.

## Worldgen Tail Integration

- [x] Add `nodes/worldgen.py` for worldgen tail nodes.
- [x] Register worldgen nodes in `nodes/__init__.py`.
- [x] Add a node to export an official-like `generation_bank_*` from current WorldMirror `PLY_DATA`.
- [x] Add a node to build `gs_data` directly from current WorldMirror tensors.
- [x] Add a wrapper node to run official `gen_gs_data.py` when official `render_results` are available.
- [x] Add a wrapper node to run native `world_gs_trainer.py`.

## Remaining Work

- [ ] Validate direct `gs_data` output against `hyworld2/worldgen/gs/opencv.py`.
- [ ] Add or update a workflow that chains:
  `WorldMirrorV2 -> WorldGen Export Bank -> WorldGen Build GS Data -> WorldGen Train 3DGS`.
- [ ] Add smoke tests for the new worldgen nodes with a tiny synthetic scene.
- [ ] Expose native 3DGS mesh export (`--export_mesh`) in the WorldGen training node.
- [ ] Expose native 3DGS SPZ export (`--convert_to_spz`) in the WorldGen training node.
- [ ] Expose native 3DGS SPX export (`--convert_to_spx`) in the WorldGen training node.
- [ ] Decide whether to replace the direct `gs_data` shortcut with official `memory_bank.alignment()` once the skipped stages are integrated.
- [ ] Add UI docs explaining the difference between feed-forward WorldMirror splats and optimized scene-level 3DGS.
