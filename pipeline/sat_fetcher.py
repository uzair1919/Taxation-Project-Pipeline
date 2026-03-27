"""
pipeline/sat_fetcher.py
========================
Downloads Sentinel-1 (GRD) and Sentinel-2 (SR) imagery via Google Earth Engine
for use as T-SwinUNet input.

Design principles
-----------------
1. Minimal restrictions — match what the original download_sat_data.py did.
   The T-SwinUNet model was trained on data fetched without strict cloud or
   orbit filters; adding them only causes data starvation.

2. Fallback chain per month:
     S2: try SR → try TOA (Level-1C) → None
     S1: try IW VV+VH any orbit → try EW → None

3. Non-zero validation — an image that rasterio reads as all-zeros is
   treated as failed and not counted.  Zero-filled placeholder files are
   marked with a sidecar ".zeros" file so resume logic does not count
   them as successful data.

4. Inference gate — fetch_for_point returns (n_s1_real, n_s2_real).
   The caller decides whether there are enough months to run the model.

5. Resume-safe — existing real TIFs are skipped.  Zero-filled TIFs
   are re-attempted on the next run (the .zeros sidecar tells us).
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
import rasterio
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)

IMG_SIZE = (128, 128)
N_MONTHS = 12

# Minimum fraction of pixels that must be non-zero for a downloaded
# image to be considered real data.  GEE nodata often comes back as
# exact zeros across the whole tile.
_MIN_NONZERO_FRACTION = 0.05   # 5% of pixels must be non-zero


# ---------------------------------------------------------------------------
# GEE initialisation
# ---------------------------------------------------------------------------

def _init_gee(project: Optional[str] = None) -> None:
    """Initialise GEE; authenticate interactively if credentials are missing."""
    import ee
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        logger.info("Google Earth Engine initialised.")
    except Exception:
        logger.warning(
            "\n══════════════════════════════════════════════════════════\n"
            "  GEE one-time authentication required.\n"
            "  A browser window will open — login and paste the code.\n"
            "══════════════════════════════════════════════════════════"
        )
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        logger.info("GEE authenticated and initialised.")


# ---------------------------------------------------------------------------
# Core download helper
# ---------------------------------------------------------------------------

def _download_image(image, bbox_gee, bands: list, timeout: int = 180) -> Optional[np.ndarray]:
    """
    Download one GEE Image for the given bbox as a (H, W, C) float32 array.

    Returns None if:
      - the HTTP request fails
      - the downloaded array is all-zeros or mostly-zeros (likely nodata)

    Parameters
    ----------
    image    : GEE Image object (already band-selected is OK, or we select here)
    bbox_gee : GEE Geometry
    bands    : list of band names to select
    timeout  : HTTP timeout in seconds
    """
    try:
        url = image.select(bands).getDownloadURL({
            "region":     bbox_gee,
            "dimensions": list(IMG_SIZE),  # [width, height] = [128, 128]
            "format":     "GEO_TIFF",
        })
    except Exception as exc:
        logger.debug(f"  getDownloadURL failed: {exc}")
        return None

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code != 200:
                logger.debug(
                    f"  GEE download HTTP {resp.status_code}: {resp.text[:120]}"
                )
                time.sleep(2 ** attempt)
                continue

            with rasterio.open(io.BytesIO(resp.content)) as src:
                arr = src.read()   # (C, H, W)

            arr = arr.transpose(1, 2, 0).astype(np.float32)  # → (H, W, C)

            # Validate: reject if the image is overwhelmingly zero
            nonzero_frac = np.count_nonzero(arr) / max(arr.size, 1)
            if nonzero_frac < _MIN_NONZERO_FRACTION:
                logger.debug(
                    f"  Downloaded image rejected: only "
                    f"{nonzero_frac*100:.1f}% non-zero pixels (likely nodata)"
                )
                return None

            return arr

        except Exception as exc:
            logger.debug(f"  GEE download attempt {attempt+1} exception: {exc}")
            time.sleep(2 ** attempt)

    return None


def _collection_first(collection) -> Optional[object]:
    """
    Return the first Image from a GEE ImageCollection, or None if empty.
    Uses a single getInfo() call — cheaper than size().getInfo() + toList().
    """
    try:
        info = collection.first().getInfo()
        if info is None:
            return None
        import ee
        return collection.first()
    except Exception:
        return None


def _month_range(year: int, month: int) -> Tuple[str, str]:
    """Return (start_date, end_date) strings for one month."""
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    return (
        f"{year}-{month:02d}-01",
        f"{year}-{month:02d}-{last_day:02d}",
    )


# ---------------------------------------------------------------------------
# S1 fetch — full fallback chain
# ---------------------------------------------------------------------------

def _fetch_s1_month(
    bbox_gee,
    bbox_wgs84: Tuple,
    year:       int,
    month:      int,
) -> Optional[np.ndarray]:
    """
    Fetch S1 for one month.  Returns (128, 128, 4) float32 or None.

    Strategy (in order, stops at first success):
      1. IW mode, VV+VH, any orbit direction
      2. EW mode, VV+VH, any orbit direction (fallback — less common)
      3. IW mode, any polarisation with VV (degraded — only VV available)

    All orbit directions are accepted — do NOT split by ASC/DSC before
    checking if any data exists.  If only one direction is available
    for a given month, its VV/VH is duplicated into all four channels.
    """
    import ee

    start, end = _month_range(year, month)

    def _try_collection(mode: str, pols: list) -> Optional[np.ndarray]:
        """Try to get S1 VV+VH from a specific mode."""
        col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(bbox_gee)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", mode))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", pols[0]))
        )
        if len(pols) > 1:
            col = col.filter(
                ee.Filter.listContains("transmitterReceiverPolarisation", pols[1])
            )
        # Sort by time (most recent first) and take best image
        col = col.sort("system:time_start", False)
        img = _collection_first(col)
        if img is None:
            return None
        return _download_image(img, bbox_gee, pols)

    # --- attempt 1: IW VV+VH any orbit ---
    arr = _try_collection("IW", ["VV", "VH"])
    if arr is not None:
        vv = arr[:, :, 0]
        vh = arr[:, :, 1]
        out = np.stack([vv, vh, vv, vh], axis=-1)  # ASC=DSC=available
        return out

    # --- attempt 2: EW VV+VH (broader search) ---
    arr = _try_collection("EW", ["VV", "VH"])
    if arr is not None:
        vv = arr[:, :, 0]
        vh = arr[:, :, 1]
        return np.stack([vv, vh, vv, vh], axis=-1)

    # --- attempt 3: IW VV only (degrade gracefully — duplicate VV into VH) ---
    arr = _try_collection("IW", ["VV"])
    if arr is not None:
        vv = arr[:, :, 0]
        return np.stack([vv, vv, vv, vv], axis=-1)

    return None


# ---------------------------------------------------------------------------
# S2 fetch — full fallback chain
# ---------------------------------------------------------------------------

def _fetch_s2_month(
    bbox_gee,
    bbox_wgs84: Tuple,
    year:       int,
    month:      int,
) -> Optional[np.ndarray]:
    """
    Fetch S2 for one month.  Returns (128, 128, 5) float32 or None.

    Strategy (in order):
      1. S2 SR Harmonised (Level-2A) — least-cloudy image, no cloud threshold
      2. S2 Level-1C TOA (if SR unavailable for this tile/date)
      3. Expand window to ±15 days either side of the month boundary
         and retry SR then TOA

    No cloud percentage threshold is applied — the model was trained
    on unfiltered data.  We just pick least-cloudy available.
    """
    import ee

    S2_BANDS = ["B2", "B3", "B4", "B8", "B11"]

    def _try_s2(collection_id: str, start: str, end: str) -> Optional[np.ndarray]:
        col = (
            ee.ImageCollection(collection_id)
            .filterBounds(bbox_gee)
            .filterDate(start, end)
            .sort("CLOUDY_PIXEL_PERCENTAGE")   # best first, no hard threshold
        )
        img = _collection_first(col)
        if img is None:
            return None
        return _download_image(img, bbox_gee, S2_BANDS)

    start, end = _month_range(year, month)

    # --- attempt 1: SR harmonised ---
    arr = _try_s2("COPERNICUS/S2_SR_HARMONIZED", start, end)
    if arr is not None:
        return arr[:, :, :5]

    # --- attempt 2: TOA Level-1C ---
    arr = _try_s2("COPERNICUS/S2_HARMONIZED", start, end)
    if arr is not None:
        return arr[:, :, :5]

    # --- attempt 3: expand window ±15 days ---
    import calendar
    last_day = calendar.monthrange(year, month)[1]

    # Expand: go 15 days back from start and 15 days forward from end
    from datetime import timedelta
    dt_start = datetime(year, month, 1) - timedelta(days=15)
    dt_end   = datetime(year, month, last_day) + timedelta(days=15)
    ext_start = dt_start.strftime("%Y-%m-%d")
    ext_end   = dt_end.strftime("%Y-%m-%d")

    arr = _try_s2("COPERNICUS/S2_SR_HARMONIZED", ext_start, ext_end)
    if arr is not None:
        logger.debug(f"  S2 {year}-{month:02d}: used expanded window ({ext_start}–{ext_end})")
        return arr[:, :, :5]

    arr = _try_s2("COPERNICUS/S2_HARMONIZED", ext_start, ext_end)
    if arr is not None:
        return arr[:, :, :5]

    return None


# ---------------------------------------------------------------------------
# GeoTIFF writer
# ---------------------------------------------------------------------------

def _save_geotiff(arr: np.ndarray, path: Path, bbox_wgs84: Tuple) -> None:
    """Save (H, W, C) float32 array as a georeferenced GeoTIFF."""
    h, w, c = arr.shape
    west, south, east, north = bbox_wgs84
    transform = from_bounds(west, south, east, north, w, h)
    with rasterio.open(
        path, "w",
        driver="GTiff", count=c, height=h, width=w,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        for i in range(c):
            dst.write(arr[:, :, i], i + 1)


def _mark_as_zeros(path: Path) -> None:
    """Write a sidecar file to indicate this TIF is zero-filled (no real data)."""
    path.with_suffix(".zeros").touch()


def _is_real_data(path: Path) -> bool:
    """Return True if this TIF contains real data (no .zeros sidecar)."""
    return path.exists() and not path.with_suffix(".zeros").exists()


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------

class SatelliteDataFetcher:
    """
    Downloads S1 + S2 imagery via GEE in the T-SwinUNet expected format.

    Parameters
    ----------
    gee_project : Optional GEE cloud project ID.
    verbose     : Log per-month status at INFO level.
    """

    def __init__(
        self,
        gee_project: Optional[str] = None,
        verbose:     bool          = True,
    ) -> None:
        self.verbose      = verbose
        self._gee_ready   = False
        self._gee_project = gee_project

    def _ensure_gee(self) -> None:
        if not self._gee_ready:
            _init_gee(self._gee_project)
            self._gee_ready = True

    def fetch_for_point(
        self,
        point_id:   str,
        bbox_wgs84: Tuple[float, float, float, float],
        year:       int,
        out_dir:    Path,
    ) -> Tuple[int, int]:
        """
        Download 12-month S1 and S2 stacks for one point.

        Files written:
            out_dir/S1/img_{point_id}_{t:02d}.tif  (4-channel, 128×128)
            out_dir/S2/img_{point_id}_{t:02d}.tif  (5-channel, 128×128)

        Zero-filled months (no GEE data) also write a .zeros sidecar file
        so that the caller can distinguish real from placeholder data.
        Placeholder files are re-attempted on the next run.

        Returns
        -------
        (n_s1_real, n_s2_real) — count of months with actual satellite data.
        Caller should check these against MIN_HEIGHT_MONTHS before running
        the model (all-zeros input produces meaningless output).
        """
        self._ensure_gee()

        s1_dir = (out_dir / "S1")
        s2_dir = (out_dir / "S2")
        s1_dir.mkdir(parents=True, exist_ok=True)
        s2_dir.mkdir(parents=True, exist_ok=True)

        bbox_gee = _gee_bbox(*bbox_wgs84)

        zeros_s1 = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 4), dtype=np.float32)
        zeros_s2 = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 5), dtype=np.float32)

        s1_real = s2_real = 0

        for month_idx in range(N_MONTHS):
            month = month_idx + 1
            tag   = f"{year}-{month:02d}"

            # ── S1 ────────────────────────────────────────────────────────
            s1_path = s1_dir / f"img_{point_id}_{month_idx:02d}.tif"

            if _is_real_data(s1_path):
                # Already downloaded successfully
                s1_real += 1
            elif s1_path.exists():
                # Zero-filled from a previous run — retry
                s1_path.unlink()
                s1_path.with_suffix(".zeros").unlink(missing_ok=True)
                arr = self._fetch_s1(bbox_gee, bbox_wgs84, year, month, point_id, tag)
                s1_real += self._save(arr, zeros_s1, s1_path, bbox_wgs84, "S1", tag, point_id)
            else:
                arr = self._fetch_s1(bbox_gee, bbox_wgs84, year, month, point_id, tag)
                s1_real += self._save(arr, zeros_s1, s1_path, bbox_wgs84, "S1", tag, point_id)

            # ── S2 ────────────────────────────────────────────────────────
            s2_path = s2_dir / f"img_{point_id}_{month_idx:02d}.tif"

            if _is_real_data(s2_path):
                s2_real += 1
            elif s2_path.exists():
                s2_path.unlink()
                s2_path.with_suffix(".zeros").unlink(missing_ok=True)
                arr = self._fetch_s2(bbox_gee, bbox_wgs84, year, month, point_id, tag)
                s2_real += self._save(arr, zeros_s2, s2_path, bbox_wgs84, "S2", tag, point_id)
            else:
                arr = self._fetch_s2(bbox_gee, bbox_wgs84, year, month, point_id, tag)
                s2_real += self._save(arr, zeros_s2, s2_path, bbox_wgs84, "S2", tag, point_id)

            time.sleep(0.5)   # GEE rate-limit buffer

        logger.info(
            f"[{point_id}/{year}] Satellite fetch complete — "
            f"S1: {s1_real}/{N_MONTHS}  S2: {s2_real}/{N_MONTHS} months real data"
        )
        return s1_real, s2_real

    # ------------------------------------------------------------------
    # Internal per-sensor fetch wrappers (for clean logging)
    # ------------------------------------------------------------------

    def _fetch_s1(self, bbox_gee, bbox_wgs84, year, month, point_id, tag):
        try:
            return _fetch_s1_month(bbox_gee, bbox_wgs84, year, month)
        except Exception as exc:
            logger.debug(f"  [{point_id}] S1 {tag} exception: {exc}")
            return None

    def _fetch_s2(self, bbox_gee, bbox_wgs84, year, month, point_id, tag):
        try:
            return _fetch_s2_month(bbox_gee, bbox_wgs84, year, month)
        except Exception as exc:
            logger.debug(f"  [{point_id}] S2 {tag} exception: {exc}")
            return None

    def _save(
        self,
        arr:        Optional[np.ndarray],
        zeros:      np.ndarray,
        path:       Path,
        bbox_wgs84: Tuple,
        sensor:     str,
        tag:        str,
        point_id:   str,
    ) -> int:
        """
        Save arr to path.  If arr is None, save zeros and mark with .zeros sidecar.
        Returns 1 if real data was saved, 0 if zero-filled.
        """
        if arr is not None:
            _save_geotiff(arr, path, bbox_wgs84)
            if self.verbose:
                logger.info(f"  [{point_id}] {sensor} {tag} ✓")
            return 1
        else:
            _save_geotiff(zeros, path, bbox_wgs84)
            _mark_as_zeros(path)
            if self.verbose:
                logger.info(f"  [{point_id}] {sensor} {tag} — no data (zero-filled)")
            return 0


def _gee_bbox(west: float, south: float, east: float, north: float):
    import ee
    return ee.Geometry.Rectangle([west, south, east, north])