"""
pipeline/sat_cache.py
=====================
Persistent cross-run satellite tile cache for GEE imagery and T-SwinUNet
prediction rasters.

How it works
------------
* GPS points are snapped to a regular geographic grid whose cell spacing
  equals HEIGHT_BUFFER_M.  Any two GPS points that fall in the same cell
  share one 1 280 × 1 280 m satellite tile — GEE is called once per cell
  per year instead of once per point.

* The cache survives across pipeline runs (like the regional tile data).
  Re-running the pipeline on the same area skips both GEE downloads and
  GPU inference entirely.

* Both satellite TIFs and prediction rasters (_pred.tif) are cached.

Thread safety
-------------
A threading.Lock guards all index reads/writes.  The pipeline runs as a
single process so this is sufficient.

Cache layout
------------
<cache_dir>/
  sat_cache_index.json                   ← master JSON index
  lat31.4523_lon74.1875/
    2023/
      S1/  tile_00.tif … tile_11.tif     (zero-filled months included)
      S2/  tile_00.tif … tile_11.tif
      pred.tif                           (T-SwinUNet height raster)
  lat31.4523_lon74.1990/
    …

Index schema
------------
{
  "version": 1,
  "grid_step_m": 640.0,
  "tiles": {
    "lat31.4523_lon74.1875": {
      "center_lat": 31.4523,
      "center_lon": 74.1875,
      "bbox_wgs84": [west, south, east, north],
      "years": {
        "2023": {
          "sat_complete":  true,   // all 24 TIFs present on disk
          "n_s1_real": 10,
          "n_s2_real": 11,
          "pred_complete": true    // pred.tif present on disk
        }
      }
    }
  }
}
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_INDEX_NAME = "sat_cache_index.json"
_N_MONTHS   = 12


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _snap(lat: float, lon: float, buffer_m: float) -> Tuple[float, float]:
    """
    Snap (lat, lon) to the nearest node of a regular geographic grid whose
    cell spacing equals buffer_m.

    Latitude is snapped first.  The longitude step is then derived from the
    SNAPPED latitude, not the raw one.  This guarantees that any two nearby
    points which share the same snapped latitude will compute an identical
    step_lon and therefore an identical snapped longitude — even if their
    raw latitudes differ slightly.  Without this, multiplying different
    per-point step_lon values by the same large grid index (~11 000) would
    produce tile_id strings that diverge by tens of metres.
    """
    step_lat = buffer_m / 111_320.0
    s_lat    = round(lat / step_lat) * step_lat
    # Use s_lat (common for all nearby points) — not raw lat
    step_lon = buffer_m / (111_320.0 * math.cos(math.radians(s_lat)))
    s_lon    = round(lon / step_lon) * step_lon
    return s_lat, s_lon


def _tile_id(s_lat: float, s_lon: float) -> str:
    """Stable string key for the snapped grid node."""
    return f"lat{s_lat:.4f}_lon{s_lon:.4f}"


def _tile_bbox(
    s_lat: float,
    s_lon: float,
    buffer_m: float,
) -> Tuple[float, float, float, float]:
    """Return (west, south, east, north) bbox centred on the snapped node."""
    step_lat = buffer_m / 111_320.0
    step_lon = buffer_m / (111_320.0 * math.cos(math.radians(s_lat)))
    return (
        s_lon - step_lon,   # west
        s_lat - step_lat,   # south
        s_lon + step_lon,   # east
        s_lat + step_lat,   # north
    )


# ---------------------------------------------------------------------------
# SatelliteCache
# ---------------------------------------------------------------------------

class SatelliteCache:
    """
    Persistent disk cache for per-tile satellite data and prediction rasters.

    Parameters
    ----------
    cache_dir : Root directory for the cache.  Created on first use.
    buffer_m  : Cell size in metres (= HEIGHT_BUFFER_M from config).
    """

    def __init__(self, cache_dir: Path, buffer_m: float = 640.0) -> None:
        self.cache_dir   = Path(cache_dir)
        self.buffer_m    = buffer_m
        self._lock       = threading.Lock()
        self._index_path = self.cache_dir / _INDEX_NAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index      = self._load_index()

    # ------------------------------------------------------------------
    # Public query helpers
    # ------------------------------------------------------------------

    def get_tile_info(
        self, lat: float, lon: float
    ) -> Tuple[str, Tuple[float, float, float, float]]:
        """
        Return (tile_id, bbox_wgs84) for a GPS coordinate.
        bbox_wgs84 = (west, south, east, north).
        """
        s_lat, s_lon = _snap(lat, lon, self.buffer_m)
        tid  = _tile_id(s_lat, s_lon)
        bbox = _tile_bbox(s_lat, s_lon, self.buffer_m)
        return tid, bbox

    def is_sat_complete(self, tile_id: str, year: int) -> bool:
        """True if all 24 satellite TIFs for this tile/year are on disk."""
        with self._lock:
            return (
                self._index.get("tiles", {})
                .get(tile_id, {})
                .get("years", {})
                .get(str(year), {})
                .get("sat_complete", False)
            )

    def is_pred_complete(self, tile_id: str, year: int) -> bool:
        """True if the prediction raster for this tile/year is cached."""
        with self._lock:
            return (
                self._index.get("tiles", {})
                .get(tile_id, {})
                .get("years", {})
                .get(str(year), {})
                .get("pred_complete", False)
            )

    def get_sat_counts(self, tile_id: str, year: int) -> Tuple[int, int]:
        """Return (n_s1_real, n_s2_real) stored in the index."""
        with self._lock:
            entry = (
                self._index.get("tiles", {})
                .get(tile_id, {})
                .get("years", {})
                .get(str(year), {})
            )
            return entry.get("n_s1_real", 0), entry.get("n_s2_real", 0)

    # ------------------------------------------------------------------
    # Disk paths
    # ------------------------------------------------------------------

    def tile_sat_dir(self, tile_id: str, year: int) -> Path:
        return self.cache_dir / tile_id / str(year) / "sat"

    def tile_pred_path(self, tile_id: str, year: int) -> Path:
        return self.cache_dir / tile_id / str(year) / "pred.tif"

    # ------------------------------------------------------------------
    # Materialise cached sat data for the predictor
    # ------------------------------------------------------------------

    def prepare_working_dir(
        self,
        tile_id:  str,
        year:     int,
        point_id: str,
        tmp_dir:  Path,
    ) -> None:
        """
        Populate tmp_dir with copies of cached satellite files renamed to the
        convention expected by BHEPredictor:

            S1/img_{point_id}_{month:02d}.tif
            S2/img_{point_id}_{month:02d}.tif

        tmp_dir must already exist (e.g. a TemporaryDirectory).
        Files that are missing from the cache are skipped silently — the
        predictor treats absent months as zero-filled data.
        """
        src_sat = self.tile_sat_dir(tile_id, year)
        for sensor in ("S1", "S2"):
            dst_sensor = tmp_dir / sensor
            dst_sensor.mkdir(parents=True, exist_ok=True)
            for m in range(_N_MONTHS):
                src = src_sat / sensor / f"tile_{m:02d}.tif"
                if src.exists():
                    shutil.copy2(src, dst_sensor / f"img_{point_id}_{m:02d}.tif")

    # ------------------------------------------------------------------
    # Write to cache
    # ------------------------------------------------------------------

    def save_sat_from_dir(
        self,
        tile_id:    str,
        year:       int,
        src_dir:    Path,
        point_id:   str,
        n_s1_real:  int,
        n_s2_real:  int,
        bbox_wgs84: Tuple,
    ) -> None:
        """
        Copy satellite TIFs from a per-point download directory (where files
        are named img_{point_id}_MM.tif) into the tile cache (tile_MM.tif).

        Existing cache files are never overwritten.  Atomic index update.
        """
        dst_sat = self.tile_sat_dir(tile_id, year)
        for sensor in ("S1", "S2"):
            dst_sensor = dst_sat / sensor
            dst_sensor.mkdir(parents=True, exist_ok=True)
            for m in range(_N_MONTHS):
                src = src_dir / sensor / f"img_{point_id}_{m:02d}.tif"
                dst = dst_sensor / f"tile_{m:02d}.tif"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

        with self._lock:
            self._ensure_tile(tile_id, bbox_wgs84)
            yr = self._index["tiles"][tile_id]["years"].setdefault(str(year), {})
            yr["sat_complete"] = True
            yr["n_s1_real"]    = n_s1_real
            yr["n_s2_real"]    = n_s2_real
            self._save_index()

    def save_pred(
        self,
        tile_id:       str,
        year:          int,
        pred_tif_path: Path,
        bbox_wgs84:    Tuple,
    ) -> None:
        """Copy a prediction raster into the tile cache and update the index."""
        dst = self.tile_pred_path(tile_id, year)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if pred_tif_path.exists() and not dst.exists():
            shutil.copy2(pred_tif_path, dst)

        with self._lock:
            self._ensure_tile(tile_id, bbox_wgs84)
            self._index["tiles"][tile_id]["years"].setdefault(
                str(year), {}
            )["pred_complete"] = True
            self._save_index()

    # ------------------------------------------------------------------
    # Load from cache
    # ------------------------------------------------------------------

    def load_pred(self, tile_id: str, year: int) -> Optional[np.ndarray]:
        """Load and return the cached prediction raster as float32 (H, W)."""
        path = self.tile_pred_path(tile_id, year)
        if not path.exists():
            return None
        try:
            import tifffile
            arr = tifffile.imread(str(path)).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            return arr
        except Exception as exc:
            logger.warning(f"[sat_cache] Failed to read cached pred {path}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.warning(
                    f"[sat_cache] Could not load index ({exc}); starting fresh."
                )
        return {"version": 1, "grid_step_m": self.buffer_m, "tiles": {}}

    def _save_index(self) -> None:
        """Write index atomically.  Must be called while holding self._lock."""
        tmp = self._index_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._index, fh, indent=2)
        tmp.replace(self._index_path)

    def _ensure_tile(self, tile_id: str, bbox_wgs84: Tuple) -> None:
        """Create index entry for tile_id if it does not yet exist."""
        tiles = self._index.setdefault("tiles", {})
        if tile_id not in tiles:
            parts   = tile_id.split("_")          # ["lat31.4523", "lon74.1875"]
            c_lat   = float(parts[0][3:])
            c_lon   = float(parts[1][3:])
            tiles[tile_id] = {
                "center_lat": c_lat,
                "center_lon": c_lon,
                "bbox_wgs84": list(bbox_wgs84),
                "years":      {},
            }
