from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np


class Retina:
    """Compound-eye preprocessing used by the v2 vision tutorials."""

    def __init__(
        self,
        ommatidia_id_map: np.ndarray | None = None,
        pale_type_mask: np.ndarray | None = None,
        distortion_coefficient: float = 3.8,
        zoom: float = 2.72,
        nrows: int = 512,
        ncols: int = 450,
    ) -> None:
        asset_dir = Path(str(files("flygym_demo.examples") / "assets/vision"))
        if ommatidia_id_map is None:
            ommatidia_id_map = np.load(asset_dir / "ommatidia_id_map.npy")
        if pale_type_mask is None:
            pale_type_mask = np.load(asset_dir / "pale_mask.npy").astype(int)
        self.ommatidia_id_map = ommatidia_id_map.astype(np.int16)
        unique, counts = np.unique(self.ommatidia_id_map, return_counts=True)
        self.num_pixels_per_ommatidia = counts[unique > 0]
        self.num_ommatidia_per_eye = len(self.num_pixels_per_ommatidia)
        self.pale_type_mask = pale_type_mask
        self.distortion_coefficient = distortion_coefficient
        self.zoom = zoom
        self.nrows = nrows
        self.ncols = ncols

    def raw_image_to_hex_pxls(self, raw_img: np.ndarray) -> np.ndarray:
        raw_img = np.asarray(raw_img)
        vals = np.zeros((self.num_ommatidia_per_eye, 2), dtype=np.float64)
        flat = raw_img.reshape((-1, 3))
        ids = self.ommatidia_id_map.ravel() - 1
        for ommatidium_id in range(self.num_ommatidia_per_eye):
            mask = ids == ommatidium_id
            if not np.any(mask):
                continue
            ch = int(self.pale_type_mask[ommatidium_id])
            vals[ommatidium_id, ch] = flat[mask, ch + 1].mean() / 255
        return vals

    def hex_pxls_to_human_readable(
        self, ommatidia_reading: np.ndarray, color_8bit: bool = False
    ) -> np.ndarray:
        arr = np.asarray(ommatidia_reading)
        if arr.shape[0] != self.num_ommatidia_per_eye:
            raise ValueError("First dimension must match number of ommatidia")
        dtype = np.uint8 if color_8bit else arr.dtype
        out = np.zeros((self.ommatidia_id_map.size, *arr.shape[1:]), dtype=dtype)
        if color_8bit:
            out += 255
            arr = arr * 255
        ids = self.ommatidia_id_map.ravel() - 1
        valid = ids >= 0
        out[valid] = arr[ids[valid]]
        return out.reshape((*self.ommatidia_id_map.shape, *ommatidia_reading.shape[1:]))

    def correct_fisheye(self, img: np.ndarray) -> np.ndarray:
        # Pure NumPy implementation kept intentionally simple for tutorial portability.
        img = np.asarray(img)
        nrows, ncols = img.shape[:2]
        rows, cols = np.indices((nrows, ncols))
        row_norm = ((2 * rows - nrows) / nrows) / self.zoom
        col_norm = ((2 * cols - ncols) / ncols) / self.zoom
        radius_sq = col_norm**2 + row_norm**2
        denom = 1 - self.distortion_coefficient * radius_sq + 1e-6
        src_rows = (((row_norm / denom + 1) * nrows) / 2).astype(int)
        src_cols = (((col_norm / denom + 1) * ncols) / 2).astype(int)
        out = np.zeros_like(img)
        valid = (0 <= src_rows) & (src_rows < nrows) & (0 <= src_cols) & (src_cols < ncols)
        out[valid] = img[src_rows[valid], src_cols[valid]]
        return out
