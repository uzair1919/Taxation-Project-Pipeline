"""
pipeline/height_bridge.py
==========================
Integrates the T-SwinUNet building height estimation model with the
existing plot boundary + SAM segmentation pipeline.

Responsibilities
----------------
1. Compute the geographic bbox for each GPS point (buffer from config).
2. Fetch S1 + S2 satellite imagery for that bbox for each requested year.
3. Run BHEPredictor to produce a 128×128 height raster (_pred.tif).
4. For each PlotRecord (both stage1 and stage2):
   a. Project the plot's WKT polygon onto the 128×128 raster grid.
   b. Also use the SAM mask (if available) as a refined building footprint.
   c. Extract height pixel values within the mask and compute the median.
   d. Classify the median height into storeys using configurable thresholds.
5. Attach HeightResult(year→{height_m, height_class}) to each PlotRecord.

Key design decisions
--------------------
* One T-SwinUNet inference per GPS point per year — not per plot.
  The 128×128 raster covers the whole neighbourhood (~1280×1280m).
  Per-plot height is extracted by masking within that raster.

* SAM mask is preferred over polygon rasterisation when available
  (it gives the actual building footprint, not the plot boundary).
  Falls back to polygon rasterisation if SAM mask absent/failed.

* Satellite TIF files are written to a temp directory and deleted
  after inference unless SAVE_HEIGHT_SAT_DATA=True in config.

* The height raster (_pred.tif) is deleted unless
  SAVE_HEIGHT_PRED_TIFS=True in config.

* If satellite data cannot be fetched for a month, zeros are used
  (graceful degradation — model has seen missing data during training).

* If inference fails for a year, all plots get height_m=None for
  that year rather than crashing.

Data attached to PlotRecord
----------------------------
    rec.height_results : dict  { year (int) → HeightYearResult }

HeightYearResult fields
-----------------------
    height_m      : float | None  — mean height in metres within footprint
    height_class  : str | None    — storey class label e.g. "2-3 storeys"
    n_pixels      : int           — number of height-raster pixels used
    source        : str           — "sam_mask" | "polygon" | "failed"
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class attached to each PlotRecord
# ---------------------------------------------------------------------------

@dataclass
class HeightYearResult:
    """Height estimate for one plot for one year."""
    height_m:     Optional[float]   # median height in metres; None = inference failed
    height_class: Optional[str]     # storey label; None = inference failed
    n_pixels:     int               # pixels used for aggregation (0 = failed)
    source:       str               # "sam_mask" | "polygon" | "failed"


# ---------------------------------------------------------------------------
# Height classification
# ---------------------------------------------------------------------------

def classify_height(
    height_m: float,
    thresholds: List[Tuple[float, float, str]],
) -> str:
    """
    Map a height in metres to a storey class label.

    Parameters
    ----------
    height_m   : raw height value from recover_label (metres)
    thresholds : list of (min_m, max_m, label) tuples from config.
                 Ranges are [min_m, max_m).  The last range catches all
                 values above it (max_m is effectively infinity).

    Example config value:
        HEIGHT_STORY_THRESHOLDS = [
            (0,  3,  "0-1 storeys"),
            (3,  6,  "1-2 storeys"),
            (6,  9,  "2-3 storeys"),
            (9,  12, "3-4 storeys"),
            (12, 999,"4+ storeys"),
        ]
    """
    if not thresholds:
        return "unknown"
    for min_m, max_m, label in thresholds:
        if min_m <= height_m < max_m:
            return label
    # Return label of the last (highest) threshold for out-of-range values
    return thresholds[-1][2]


# ---------------------------------------------------------------------------
# Mask extraction from SAM RLE or polygon WKT
# ---------------------------------------------------------------------------



def _polygon_to_mask_128(
    polygon_wkt: str,
    bbox_wgs84:  Tuple[float, float, float, float],  # west, south, east, north
    img_size:    Tuple[int, int] = (128, 128),
) -> Optional[np.ndarray]:
    """
    Rasterise a WGS84 WKT polygon onto a 128×128 grid aligned with bbox_wgs84.
    Returns binary uint8 array or None.
    """
    from pipeline.utils.geo import parse_wkt_vertices

    vertices = parse_wkt_vertices(polygon_wkt)
    if not vertices:
        return None

    west, south, east, north = bbox_wgs84
    lon_range = east  - west
    lat_range = north - south
    h, w = img_size

    if lon_range <= 0 or lat_range <= 0:
        return None

    pts_px = []
    for lon, lat in vertices:
        px_x = int((lon - west)  / lon_range * w)
        px_y = int((north - lat) / lat_range * h)   # Y inverted
        px_x = max(0, min(w - 1, px_x))
        px_y = max(0, min(h - 1, px_y))
        pts_px.append([px_x, px_y])

    if len(pts_px) < 3:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    pts_arr = np.array(pts_px, dtype=np.int32)
    cv2.fillPoly(mask, [pts_arr], 1)
    return mask




def _extract_height_from_raster(
    pred_arr:    np.ndarray,    # (128, 128) height raster
    mask:        np.ndarray,    # (128, 128) binary mask
    aggregation: str = "median",
) -> Tuple[Optional[float], int]:
    """
    Aggregate height pixels within the binary mask.
    Returns (height_m, n_pixels) or (None, 0).

    aggregation — "median" (default, robust to edge/shadow outliers)
                  "mean"   (arithmetic average)
    """
    if mask is None or mask.sum() == 0:
        return None, 0

    agg = np.median if aggregation == "median" else np.mean

    vals = pred_arr[mask > 0]
    # Exclude zero-height pixels (likely no-data) if there are non-zero ones
    nonzero = vals[vals > 0]
    if len(nonzero) > 0:
        return float(agg(nonzero)), int(len(nonzero))
    elif len(vals) > 0:
        return float(agg(vals)), int(len(vals))
    return None, 0


# ---------------------------------------------------------------------------
# BHEPredictor wrapper that works in-process
# ---------------------------------------------------------------------------

def _load_predictor(
    model_path:  str,
    config_path: str,
    tswin_root:  str,
) -> object:
    """
    Load BHEPredictor from the T-SwinUNet library.

    Parameters
    ----------
    model_path  : path to model.pth
    config_path : path to exp3.yaml (or equivalent)
    tswin_root  : directory containing the 'libs' package
    """
    import sys
    if tswin_root not in sys.path:
        sys.path.insert(0, tswin_root)

    import yaml
    from omegaconf import OmegaConf

    with open(config_path, "r") as f:
        configs = OmegaConf.create(yaml.safe_load(f))

    from tswin_unet.libs.predict import BHEPredictor
    return BHEPredictor(model_path=model_path, configs=configs)


def _run_inference(
    predictor,
    data_dir: Path,
    point_id: str,
    out_dir:  Path,
) -> Optional[np.ndarray]:
    """
    Run predictor on one point's satellite data and return the
    height raster as a (128, 128) float32 numpy array.

    Returns None on any failure.
    """
    import sys
    subjects = [f"img_{point_id}.tif"]   # predictor strips extension → img_P0001

    # The predictor saves _pred.tif and _seg.tif but we only need _pred.tif
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        predictor.predict(str(data_dir), subjects, str(out_dir))
    except Exception as exc:
        logger.error(f"BHEPredictor.predict failed: {exc}")
        logger.error(traceback.format_exc())
        return None

    pred_path = out_dir / f"img_{point_id}_pred.tif"
    if not pred_path.exists():
        logger.warning(f"_pred.tif not found at {pred_path}")
        return None

    try:
        import tifffile
        arr = tifffile.imread(str(pred_path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]   # take first channel if (C, H, W)
        return arr
    except Exception as exc:
        logger.error(f"Failed to read {pred_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main HeightRunner class
# ---------------------------------------------------------------------------

class HeightRunner:
    """
    Orchestrates satellite fetch + T-SwinUNet inference + per-plot
    height extraction for all plots in a PointResult.

    Call run_on_point() after SAM has run (so SAM mask data is available).

    Parameters
    ----------
    cfg : unified config module (config.py)
    """

    def __init__(self, cfg) -> None:
        self.cfg        = cfg
        self._predictor = None   # lazy-loaded
        self._fetcher   = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_on_point(
        self,
        point_result,          # PointResult
        lat:      float,
        lon:      float,
        out_dir:  Path,
    ) -> None:
        """
        Attach HeightYearResult to every PlotRecord in point_result
        for every year in cfg.HEIGHT_YEARS.

        Results are attached in-place:
            rec.height_results = {year: HeightYearResult, ...}
        """
        years = list(getattr(self.cfg, "HEIGHT_YEARS", []))
        if not years:
            logger.info(f"[{point_result.point_id}] HEIGHT_YEARS empty — skipping height")
            return

        self._ensure_loaded()

        # Compute point bbox (buffer around GPS coordinate)
        buffer_m = float(getattr(self.cfg, "HEIGHT_BUFFER_M", 640.0))
        bbox_wgs84 = _compute_bbox(lat, lon, buffer_m)

        for year in years:
            logger.info(
                f"[{point_result.point_id}] Height estimation — year {year} …"
            )
            self._process_year(
                point_result = point_result,
                bbox_wgs84   = bbox_wgs84,
                year         = year,
                out_dir      = out_dir,
            )

    def close(self) -> None:
        """Release model GPU memory."""
        try:
            if self._predictor is not None:
                del self._predictor
                self._predictor = None
            try:
                import torch
                if torch.cuda.is_available():
                    import torch
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return

        cfg = self.cfg
        model_path  = str(getattr(cfg, "TSWIN_MODEL_PATH",  ""))
        config_path = str(getattr(cfg, "TSWIN_CONFIG_PATH", ""))
        tswin_root  = str(getattr(cfg, "TSWIN_ROOT",        ""))

        if not model_path or not config_path:
            raise ValueError(
                "TSWIN_MODEL_PATH and TSWIN_CONFIG_PATH must be set in config.py "
                "to use height estimation."
            )

        logger.info(f"Loading T-SwinUNet from {model_path} …")
        self._predictor = _load_predictor(model_path, config_path, tswin_root)
        logger.info("T-SwinUNet loaded.")

        # Satellite fetcher (Google Earth Engine)
        gee_project = getattr(cfg, "GEE_PROJECT", None)
        if gee_project:
            gee_project = str(gee_project) if gee_project else None

        from pipeline.sat_fetcher import SatelliteDataFetcher
        self._fetcher = SatelliteDataFetcher(
            gee_project = gee_project,
            verbose     = bool(getattr(cfg, "HEIGHT_VERBOSE", True)),
        )

    def _process_year(
        self,
        point_result,
        bbox_wgs84: Tuple,
        year:       int,
        out_dir:    Path,
    ) -> None:
        """Run one year of height estimation for all plots in point_result."""
        point_id = point_result.point_id
        save_sat = bool(getattr(self.cfg, "SAVE_HEIGHT_SAT_DATA",  False))
        save_pred = bool(getattr(self.cfg, "SAVE_HEIGHT_PRED_TIFS", False))

        # ── 1. Satellite data directory ───────────────────────────────────
        if save_sat:
            sat_dir = out_dir / "height" / str(year) / "sat"
        else:
            # Write to a temp dir; cleaned up at the end of this method
            _tmp_ctx = tempfile.TemporaryDirectory()
            sat_dir  = Path(_tmp_ctx.name)

        # ── 2. Fetch S1 + S2 ─────────────────────────────────────────────
        n_s1_real, n_s2_real = self._fetcher.fetch_for_point(
            point_id   = point_id,
            bbox_wgs84 = bbox_wgs84,
            year       = year,
            out_dir    = sat_dir,
        )

        min_months = int(getattr(self.cfg, "MIN_HEIGHT_MONTHS", 3))
        if n_s1_real < min_months or n_s2_real < min_months:
            logger.warning(
                f"[{point_id}/{year}] Insufficient satellite data "
                f"(S1: {n_s1_real}/{12}, S2: {n_s2_real}/{12}, "
                f"minimum required: {min_months}) — skipping inference."
            )
            # Mark all plots as failed for this year
            all_records = list(point_result.stage1_plots) + list(point_result.stage2_plots)
            for rec in all_records:
                if not hasattr(rec, "height_results") or rec.height_results is None:
                    rec.height_results = {}
                rec.height_results[year] = HeightYearResult(
                    height_m=None, height_class=None, n_pixels=0,
                    source="insufficient_data"
                )
            # Cleanup
            if not save_sat:
                try: _tmp_ctx.cleanup()
                except Exception: pass
            return

        # ── 3. Run inference ──────────────────────────────────────────────
        if save_pred:
            pred_out_dir = out_dir / "height" / str(year) / "pred"
        else:
            _tmp_pred = tempfile.TemporaryDirectory()
            pred_out_dir = Path(_tmp_pred.name)

        pred_arr = _run_inference(
            predictor = self._predictor,
            data_dir  = sat_dir,
            point_id  = point_id,
            out_dir   = pred_out_dir,
        )

        # ── 4. Attach results to each PlotRecord ──────────────────────────
        thresholds = list(getattr(self.cfg, "HEIGHT_STORY_THRESHOLDS", []))
        all_records = list(point_result.stage1_plots) + list(point_result.stage2_plots)

        for rec in all_records:
            if not hasattr(rec, "height_results") or rec.height_results is None:
                rec.height_results = {}

            if pred_arr is None:
                rec.height_results[year] = HeightYearResult(
                    height_m=None, height_class=None, n_pixels=0, source="failed"
                )
                continue

            aggregation = str(getattr(self.cfg, "HEIGHT_AGGREGATION", "median")).lower()
            height_m, n_px, source = self._extract_plot_height(
                rec, pred_arr, bbox_wgs84, aggregation
            )

            if height_m is not None:
                h_class = classify_height(height_m, thresholds)
            else:
                h_class = None

            rec.height_results[year] = HeightYearResult(
                height_m     = round(height_m, 2) if height_m is not None else None,
                height_class = h_class,
                n_pixels     = n_px,
                source       = source,
            )

        logger.info(
            f"[{point_id}/{year}] Height extraction complete — "
            f"{sum(1 for r in all_records if r.height_results.get(year) and r.height_results[year].height_m is not None)}"
            f"/{len(all_records)} plots with valid height"
        )

        # ── 5. Cleanup temp dirs ──────────────────────────────────────────
        if not save_sat:
            try: _tmp_ctx.cleanup()
            except Exception: pass
        if not save_pred:
            try: _tmp_pred.cleanup()
            except Exception: pass

    def _extract_plot_height(
        self,
        rec,             # PlotRecord
        pred_arr:    np.ndarray,
        bbox_wgs84:  Tuple,
        aggregation: str = "median",
    ) -> Tuple[Optional[float], int, str]:
        """
        Extract height for one plot.  Returns (height_m, n_pixels, source).

        Priority:
          1. SAM mask georeferenced polygon (best building footprint)
          2. Polygon WKT rasterised on 128×128 grid (fallback)
          3. None (both failed)
        """
        # Try the georeferenced SAM mask polygon first
        sam_result = getattr(rec, "sam_result", None)
        if sam_result is not None and getattr(sam_result, "status", "") == "success":
            mask_geo_wkt = getattr(sam_result, "mask_geo_wkt", "")
            if mask_geo_wkt:
                mask = _polygon_to_mask_128(mask_geo_wkt, bbox_wgs84)
                if mask is not None and mask.sum() > 0:
                    height_m, n_px = _extract_height_from_raster(pred_arr, mask, aggregation)
                    if height_m is not None:
                        return height_m, n_px, "sam_mask"

        # Fallback: rasterise the refined polygon WKT
        mask = _polygon_to_mask_128(rec.polygon_wkt, bbox_wgs84)
        if mask is not None and mask.sum() > 0:
            height_m, n_px = _extract_height_from_raster(pred_arr, mask, aggregation)
            if height_m is not None:
                return height_m, n_px, "polygon"

        return None, 0, "failed"


# ---------------------------------------------------------------------------
# Geographic helper
# ---------------------------------------------------------------------------

def _compute_bbox(
    lat: float, lon: float, buffer_m: float
) -> Tuple[float, float, float, float]:
    """
    Compute a square bbox (west, south, east, north) around (lat, lon)
    with a given buffer in metres.  Matches the logic in download_sat_data.py.
    """
    import math
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))

    buf_lat = buffer_m / m_per_deg_lat
    buf_lon = buffer_m / m_per_deg_lon

    return (
        lon - buf_lon,   # west
        lat - buf_lat,   # south
        lon + buf_lon,   # east
        lat + buf_lat,   # north
    )