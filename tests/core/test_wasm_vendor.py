"""The docs build must refresh cached WASM binaries after a MuJoCo upgrade."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("cached_version", [None, "3.9.0", "4.0.1"])
def test_vendor_cache_tracks_mujoco_version(tmp_path, monkeypatch, cached_version):
    path = Path(__file__).resolve().parents[2] / "scripts/dev/properdocs_hooks.py"
    spec = importlib.util.spec_from_file_location("properdocs_hooks", path)
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    monkeypatch.setattr(hooks, "_VENDOR_DIR", tmp_path)
    for name in ["mujoco/mujoco.wasm", "three/three.module.js"]:
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        target.touch()
    version_file = tmp_path / "mujoco/VERSION.txt"
    if cached_version is not None:
        version_file.write_text(cached_version)
    downloads = []
    monkeypatch.setattr(
        hooks, "_fetch_npm_files", lambda url, members: downloads.append(url)
    )

    hooks._ensure_vendor()

    assert len(downloads) == (0 if cached_version == hooks._MUJOCO_VERSION else 2)
    assert version_file.read_text().strip() == hooks._MUJOCO_VERSION
