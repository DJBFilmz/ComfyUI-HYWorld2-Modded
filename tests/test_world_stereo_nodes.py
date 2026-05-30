import math
import sys
import os
import importlib.util
import importlib.abc

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "worldstereo"))


@pytest.fixture
def torch_module():
    return pytest.importorskip("torch")


def load_world_stereo_module():
    pytest.importorskip("torch")
    module_path = os.path.join(PROJECT_ROOT, "nodes", "world_stereo.py")
    spec = importlib.util.spec_from_file_location("_world_stereo_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestImportWithoutComfyUI:
    def test_world_stereo_imports_without_folder_paths(self, monkeypatch, torch_module):
        class BlockFolderPaths(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "folder_paths":
                    raise ImportError("blocked folder_paths for smoke test")
                return None

        monkeypatch.setattr(sys, "meta_path", [BlockFolderPaths()] + sys.meta_path)
        sys.modules.pop("folder_paths", None)

        module = load_world_stereo_module()

        assert module.FOLDER_PATHS_AVAILABLE is False
        assert module._get_models_base() == os.path.join(PROJECT_ROOT, "models")


class TestBuildIntrinsics:
    def test_shape(self, torch_module):
        _build_intrinsics = load_world_stereo_module()._build_intrinsics
        K = _build_intrinsics(70.0, 768, 480)
        assert K.shape == (3, 3)

    def test_principal_point_at_center(self, torch_module):
        _build_intrinsics = load_world_stereo_module()._build_intrinsics
        K = _build_intrinsics(70.0, 768, 480)
        assert K[0, 2].item() == pytest.approx(384.0)
        assert K[1, 2].item() == pytest.approx(240.0)

    def test_focal_from_fov(self, torch_module):
        _build_intrinsics = load_world_stereo_module()._build_intrinsics
        K = _build_intrinsics(90.0, 200, 100)
        # fov=90 → fx = (200/2)/tan(45°) = 100
        assert K[0, 0].item() == pytest.approx(100.0, abs=1e-3)


class TestBuildTrajectory:
    def test_circular_shape(self, torch_module):
        module = load_world_stereo_module()
        if not module.PYTORCH3D_AVAILABLE:
            pytest.skip("pytorch3d is required for circular trajectories")
        _build_trajectory = module._build_trajectory
        c2ws, intrs = _build_trajectory("circular", 16, 1.0, 0.0, 0.0, 70.0, 768, 480)
        assert c2ws.shape == (16, 4, 4)
        assert intrs.shape == (16, 3, 3)

    def test_forward_shape(self, torch_module):
        _build_trajectory = load_world_stereo_module()._build_trajectory
        c2ws, intrs = _build_trajectory("forward", 8, 1.0, 0.5, 0.0, 70.0, 768, 480)
        assert c2ws.shape == (8, 4, 4)

    def test_zoom_in_shape(self, torch_module):
        _build_trajectory = load_world_stereo_module()._build_trajectory
        c2ws, intrs = _build_trajectory("zoom_in", 12, 1.0, 0.0, 0.0, 70.0, 768, 480)
        assert c2ws.shape == (12, 4, 4)

    def test_all_c2ws_are_valid_se3(self, torch_module):
        _build_trajectory = load_world_stereo_module()._build_trajectory
        c2ws, _ = _build_trajectory("forward", 8, 1.0, 0.05, 0.0, 70.0, 768, 480)
        # Bottom row must be [0,0,0,1]
        for i in range(8):
            bottom = c2ws[i, 3, :]
            assert bottom[3].item() == pytest.approx(1.0, abs=1e-5)
            assert bottom[0].item() == pytest.approx(0.0, abs=1e-5)

    def test_intrinsics_replicated_per_frame(self, torch_module):
        _build_trajectory = load_world_stereo_module()._build_trajectory
        c2ws, intrs = _build_trajectory("forward", 5, 1.0, 0.05, 0.0, 70.0, 768, 480)
        # All intrinsics frames must be identical
        for i in range(1, 5):
            assert torch_module.allclose(intrs[0], intrs[i])


class TestC2WToW2C:
    def test_identity_roundtrip(self, torch_module):
        _c2w_to_w2c = load_world_stereo_module()._c2w_to_w2c
        c2ws = torch_module.eye(4).unsqueeze(0).repeat(4, 1, 1)
        w2cs = _c2w_to_w2c(c2ws)
        assert torch_module.allclose(w2cs, c2ws, atol=1e-5)

    def test_shape_preserved(self, torch_module):
        _c2w_to_w2c = load_world_stereo_module()._c2w_to_w2c
        c2ws = torch_module.randn(10, 4, 4)
        # Make valid SE3
        c2ws[:, 3, :] = torch_module.tensor([0., 0., 0., 1.])
        w2cs = _c2w_to_w2c(c2ws)
        assert w2cs.shape == (10, 4, 4)
