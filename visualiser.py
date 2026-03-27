"""
visualiser.py
=============
Local Flask server for the pipeline output visualisation tool.

Usage
-----
    python visualiser.py --excel pipeline_output/final_dataset.xlsx
    python visualiser.py --excel pipeline_output/final_dataset.xlsx --output-dir pipeline_output
    # Open http://localhost:5000

Requirements
------------
    pip install flask pandas openpyxl numpy pillow matplotlib tifffile rasterio opencv-python
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# Import height thresholds from config so visualiser adapts automatically
# when the user edits HEIGHT_STORY_THRESHOLDS in config.py.
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from config import HEIGHT_STORY_THRESHOLDS as _HEIGHT_STORY_THRESHOLDS
except Exception:
    _HEIGHT_STORY_THRESHOLDS = [
        (0.0,   3.0,  "0-1 storeys"),
        (3.0,   6.0,  "1-2 storeys"),
        (6.0,   9.0,  "2-3 storeys"),
        (9.0,  12.0,  "3-4 storeys"),
        (12.0, 15.0,  "4-5 storeys"),
        (15.0, 999.0, "5+ storeys"),
    ]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("visualiser")

app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_EXCEL_PATH:  Optional[Path] = None
_OUTPUT_DIR:  Optional[Path] = None
_DF_POINTS:   Optional[pd.DataFrame] = None
_DF_STAGE1:   Optional[pd.DataFrame] = None
_DF_STAGE2:   Optional[pd.DataFrame] = None
_HEIGHT_YEARS: list = []
_ESRI_FETCHER = None


# ---------------------------------------------------------------------------
# Error handler — always return JSON so the frontend never gets HTML
# ---------------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_excel(excel_path: Path) -> None:
    global _DF_POINTS, _DF_STAGE1, _DF_STAGE2, _HEIGHT_YEARS

    logger.info(f"Loading Excel: {excel_path}")
    _DF_POINTS = pd.read_excel(excel_path, sheet_name="points")
    _DF_STAGE1 = pd.read_excel(excel_path, sheet_name="plots_stage1")
    _DF_STAGE2 = pd.read_excel(excel_path, sheet_name="plots_stage2")

    years = set()
    for col in _DF_STAGE1.columns:
        m = re.match(r"height_m_(\d+)", col)
        if m:
            years.add(int(m.group(1)))
    _HEIGHT_YEARS = sorted(years)
    logger.info(
        f"Loaded {len(_DF_POINTS)} points, "
        f"{len(_DF_STAGE1)} stage1 plots, "
        f"{len(_DF_STAGE2)} stage2 plots, "
        f"height years: {_HEIGHT_YEARS}"
    )


def _get_esri():
    global _ESRI_FETCHER
    if _ESRI_FETCHER is None:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from sam_utils.esri_tile_fetcher import ESRITileFetcher
            cache = Path(__file__).parent / "esri_vis_cache"
            _ESRI_FETCHER = ESRITileFetcher(cache_dir=cache)
            logger.info("ESRITileFetcher loaded.")
        except Exception as exc:
            logger.warning(f"Could not load ESRITileFetcher: {exc}")
    return _ESRI_FETCHER


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _decode_rle(rle: str, h: int, w: int) -> Optional[np.ndarray]:
    """Decode pipeline RLE format: 'val:count,val:count,...'"""
    if not rle or not isinstance(rle, str) or str(rle).strip() in ("nan", "None", ""):
        return None
    try:
        flat = []
        for part in str(rle).split(","):
            val_s, cnt_s = part.split(":")
            flat.extend([int(val_s)] * int(cnt_s))
        arr = np.array(flat, dtype=np.uint8)
        if arr.size != h * w:
            return None
        return arr.reshape(h, w)
    except Exception:
        return None


def _wkt_to_coords(wkt: str) -> Optional[list]:
    """Parse WKT POLYGON → [[lon,lat], ...] (no closing duplicate)."""
    if not wkt or str(wkt).strip() in ("nan", "None", ""):
        return None
    m = re.search(r"POLYGON\s*\(\((.*?)\)\)", str(wkt), re.IGNORECASE)
    if not m:
        return None
    coords = []
    for pair in m.group(1).split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            coords.append([float(parts[0]), float(parts[1])])
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords if len(coords) >= 3 else None


def _coords_bbox(coords: list) -> Tuple[float, float, float, float]:
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


def _geo_to_px(lon: float, lat: float,
               west: float, south: float, east: float, north: float,
               w: int, h: int) -> Tuple[int, int]:
    px = int((lon - west)  / max(east - west,   1e-9) * w)
    py = int((north - lat) / max(north - south,  1e-9) * h)
    return int(np.clip(px, 0, w-1)), int(np.clip(py, 0, h-1))


def _img_to_b64(arr: np.ndarray) -> str:
    from PIL import Image as PILImage
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _apply_colormap(arr: np.ndarray, vmin: float = 0, vmax: float = 60,
                    cmap_name: str = "inferno") -> np.ndarray:
    """Apply a matplotlib colormap → (H, W, 3) uint8."""
    import matplotlib.colors as mcolors
    import matplotlib.cm as _cm
    try:
        import matplotlib
        cmap = matplotlib.colormaps[cmap_name]
    except Exception:
        cmap = _cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = (cmap(norm(arr)) * 255).astype(np.uint8)
    return rgba[:, :, :3]


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if (f != f) else round(f, 3)
    except Exception:
        return None


def _safe_str(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v).strip()
    return None if s in ("nan", "None", "") else s


def _build_height_class_colors() -> dict:
    """Generate a green→yellow→red palette for each height class in config."""
    import matplotlib
    n = len(_HEIGHT_STORY_THRESHOLDS)
    try:
        cmap = matplotlib.colormaps["RdYlGn"].reversed()
    except Exception:
        import matplotlib.cm as _cm
        cmap = _cm.get_cmap("RdYlGn_r")
    colors = {}
    for i, (_, _, label) in enumerate(_HEIGHT_STORY_THRESHOLDS):
        t = i / max(n - 1, 1)
        r, g, b, _ = cmap(t)
        colors[label] = (int(r * 255), int(g * 255), int(b * 255))
    return colors

def _height_vis_max() -> float:
    """Colormap upper limit = lower bound of the last (open-ended) class.
    Concentrates contrast in the low-storey range where most plots fall."""
    if len(_HEIGHT_STORY_THRESHOLDS) >= 2:
        return float(_HEIGHT_STORY_THRESHOLDS[-1][0])
    return 60.0

HEIGHT_CLASS_COLORS = _build_height_class_colors()
_HEIGHT_VIS_MAX     = _height_vis_max()

def _class_color(label: Optional[str]) -> Tuple[int, int, int]:
    if not label or str(label).strip() in ("nan", "None", ""):
        return (150, 150, 150)
    return HEIGHT_CLASS_COLORS.get(str(label), (180, 80, 220))


# ---------------------------------------------------------------------------
# TIF loaders
# ---------------------------------------------------------------------------

def _load_pred_tif(point_id: str, year: int) -> Optional[np.ndarray]:
    """Load 128×128 height prediction raster."""
    if not _OUTPUT_DIR:
        return None
    candidates = list(_OUTPUT_DIR.glob(
        f"{point_id}_*/height/{year}/pred/img_{point_id}_pred.tif"
    ))
    if not candidates:
        return None
    try:
        import tifffile
        arr = tifffile.imread(str(candidates[0])).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        return arr
    except Exception as exc:
        logger.debug(f"pred tif read failed: {exc}")
        return None


def _load_sat_rgb(point_id: str, year: int) -> Optional[Tuple[np.ndarray, Tuple]]:
    """
    Load S2 RGB image.
    Returns (rgb_array, bbox_wgs84) or None.
    The bbox is read from the GeoTIFF geotransform so it is accurate.
    """
    if not _OUTPUT_DIR:
        return None
    # Try months 3-9 (spring/summer — least cloud in Pakistan)
    for month in [3, 4, 5, 6, 7, 8, 9, 2, 10, 1, 11, 12]:
        candidates = list(_OUTPUT_DIR.glob(
            f"{point_id}_*/height/{year}/sat/S2/img_{point_id}_{month:02d}.tif"
        ))
        if not candidates:
            continue
        try:
            import rasterio
            with rasterio.open(str(candidates[0])) as src:
                arr = src.read()          # (C, H, W)
                bounds = src.bounds       # left, bottom, right, top
                bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)

            if arr.shape[0] < 3:
                continue

            # arr is (C, H, W) where C = [B2, B3, B4, B8, B11]
            # True colour: R=B4(ch2), G=B3(ch1), B=B2(ch0)
            r = arr[2].astype(np.float32)
            g = arr[1].astype(np.float32)
            b = arr[0].astype(np.float32)

            # Check if this month has real data (not all zeros)
            if r.max() < 1 and g.max() < 1 and b.max() < 1:
                continue

            # Percentile stretch for good contrast regardless of DN range
            p2  = np.percentile(r[r > 0], 2)  if (r > 0).any() else 0
            p98 = np.percentile(r[r > 0], 98) if (r > 0).any() else 1
            if p98 <= p2:
                p2, p98 = r.min(), max(r.max(), p2 + 1)

            def stretch(ch):
                lo = np.percentile(ch[ch > 0], 2) if (ch > 0).any() else 0
                hi = np.percentile(ch[ch > 0], 98) if (ch > 0).any() else 1
                if hi <= lo:
                    hi = lo + 1
                return np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

            rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
            return rgb, bbox

        except Exception as exc:
            logger.debug(f"sat rgb read failed month {month}: {exc}")
            continue
    return None


def _point_bbox_from_config(lat_c: float, lon_c: float, buffer_m: float = 640.0
                             ) -> Tuple[float, float, float, float]:
    """Compute the geographic bbox the height model was run over."""
    buf_lat = buffer_m / 111320.0
    buf_lon = buffer_m / (111320.0 * math.cos(math.radians(lat_c)))
    return lon_c - buf_lon, lat_c - buf_lat, lon_c + buf_lon, lat_c + buf_lat


def _get_point_centre(point_id: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) for a point_id from the points DataFrame."""
    if _DF_POINTS is None:
        return None
    row = _DF_POINTS[_DF_POINTS["point_id"].astype(str) == str(point_id)]
    if row.empty:
        return None
    return float(row.iloc[0]["latitude"]), float(row.iloc[0]["longitude"])


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------

def _render_on_background(
    canvas:         np.ndarray,        # (H, W, 3) uint8 — modified in-place clone
    west:  float, south: float, east: float, north: float,
    polygon_coords: Optional[list],
    sam_mask_wkt:   Optional[str],
    height_class:   Optional[str],
    show_polygon:   bool = True,
    show_sam_mask:  bool = True,
) -> np.ndarray:
    """
    Overlay SAM mask and polygon boundary onto canvas.
    Returns the modified canvas (uint8 RGB).

    FIX: The old code had a broken assignment that tried to convert a
    multi-element numpy array to a Python scalar.  The blending is now
    done entirely with vectorised numpy operations — no scalar conversion.
    """
    import cv2

    h, w = canvas.shape[:2]
    out = canvas.copy()

    # ── SAM mask overlay ────────────────────────────────────────────────
    if show_sam_mask:
        mask = None

        # Rasterise the SAM mask WKT polygon on the canvas grid
        if sam_mask_wkt and str(sam_mask_wkt).strip() not in ("nan", "None", ""):
            bbox_coords = _wkt_to_coords(str(sam_mask_wkt))
            if bbox_coords:
                pts = np.array(
                    [_geo_to_px(lon, lat, west, south, east, north, w, h)
                     for lon, lat in bbox_coords],
                    dtype=np.int32
                )
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 1)

        if mask is not None and mask.sum() > 0:
            # Resize mask to canvas size if needed
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

            color = np.array(_class_color(height_class), dtype=np.float32)
            m = (mask > 0)

            # Fully vectorised alpha blend — no Python scalar conversion
            alpha = 0.45
            out_f = out.astype(np.float32)
            out_f[m] = out_f[m] * (1.0 - alpha) + color * alpha
            out = np.clip(out_f, 0, 255).astype(np.uint8)

            # Bright outline around the mask region
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            border = cv2.dilate(mask, kernel) - mask
            border_color = np.clip(color * 1.4, 0, 255).astype(np.uint8)
            out[border > 0] = border_color

    # ── Polygon boundary ─────────────────────────────────────────────────
    if show_polygon and polygon_coords:
        pts = np.array(
            [_geo_to_px(lon, lat, west, south, east, north, w, h)
             for lon, lat in polygon_coords],
            dtype=np.int32
        )
        cv2.polylines(out, [pts], isClosed=True, color=(0, 220, 255), thickness=2)

    return out


# ---------------------------------------------------------------------------
# View builders
# ---------------------------------------------------------------------------

def _build_esri_bg(west, south, east, north, zoom, out_size):
    """Fetch ESRI tiles and return (H, W, 3) uint8 resized to out_size."""
    import cv2
    esri = _get_esri()
    if esri is None:
        return None
    try:
        pil_img = esri.fetch_bbox(west, south, east, north, zoom=zoom)
        arr = np.array(pil_img)[:, :, :3]
        return cv2.resize(arr, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    except Exception as exc:
        logger.warning(f"ESRI fetch failed: {exc}")
        return None


def _build_height_bg(point_id, year, out_size, poly_bbox_wgs84=None):
    """
    Return (bg_array, bbox_wgs84) for the height raster view.
    Always shows the full point neighbourhood (128×128 raster).
    If poly_bbox_wgs84 is given, also returns the coordinates so the
    caller can draw overlays in the correct geographic space.
    """
    import cv2
    pred = _load_pred_tif(point_id, year)
    if pred is None:
        return None, None

    # Get point centre from points table for accurate bbox
    centre = _get_point_centre(point_id)
    if centre:
        lat_c, lon_c = centre
    else:
        # Fallback: estimate from polygon coords
        if poly_bbox_wgs84:
            pw, ps, pe, pn = poly_bbox_wgs84
            lat_c = (ps + pn) / 2
            lon_c = (pw + pe) / 2
        else:
            return None, None

    bbox = _point_bbox_from_config(lat_c, lon_c)
    coloured = _apply_colormap(pred, vmin=0, vmax=_HEIGHT_VIS_MAX, cmap_name="inferno")
    bg = cv2.resize(coloured, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return bg, bbox


def _build_sat_bg(point_id, year, out_size):
    """
    Return (bg_array, bbox_wgs84) for the GEE satellite view.
    Uses the geotransform from the saved GeoTIFF for accurate bbox.
    """
    import cv2
    result = _load_sat_rgb(point_id, year)
    if result is None:
        return None, None
    rgb, bbox = result
    bg = cv2.resize(rgb, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return bg, bbox


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    html_path = Path(__file__).parent / "visualiser.html"
    return html_path.read_text(encoding="utf-8")


@app.route("/api/meta")
def api_meta():
    return jsonify({
        "height_years": _HEIGHT_YEARS,
        "n_points":     len(_DF_POINTS),
    })


@app.route("/api/points")
def api_points():
    rows = []
    for _, r in _DF_POINTS.iterrows():
        rows.append({
            "point_id":       str(r.get("point_id", "")),
            "name":           str(r.get("name", "")),
            "latitude":       _safe_float(r.get("latitude")) or 0.0,
            "longitude":      _safe_float(r.get("longitude")) or 0.0,
            "n_stage1_plots": int(r.get("n_stage1_plots", 0)),
            "n_stage2_plots": int(r.get("n_stage2_plots", 0)),
            "n_clusters":     int(r.get("n_clusters", 0)),
        })
    return jsonify(rows)


@app.route("/api/plots/<point_id>")
def api_plots(point_id: str):
    year = int(request.args.get("year", _HEIGHT_YEARS[0] if _HEIGHT_YEARS else 0))

    def _row(row, stage):
        d = {
            "point_id":     str(row.get("point_id", "")),
            "stage":        stage,
            "plot_index":   int(row.get("plot_index", 0)),
            "cluster_id":   (int(row["cluster_id"])
                             if "cluster_id" in row and not pd.isna(row.get("cluster_id"))
                             else None),
            "polygon_wkt":  str(row.get("polygon_wkt", "")),
            "sam_status":   str(row.get("sam_status", "")),
            "sam_score":    _safe_float(row.get("sam_score")),
            "sam_iou":      _safe_float(row.get("sam_iou")),
            "sam_area_m2":  _safe_float(row.get("sam_area_m2")),
            "sam_mask_wkt": str(row.get("sam_mask_wkt", "")),
            "sam_bbox_wkt": str(row.get("sam_bbox_wkt", "")),
        }
        if year:
            d["height_m"]     = _safe_float(row.get(f"height_m_{year}"))
            d["height_class"] = _safe_str(row.get(f"height_class_{year}"))
            d["height_src"]   = _safe_str(row.get(f"height_src_{year}"))
        return d

    s1 = _DF_STAGE1[_DF_STAGE1["point_id"].astype(str) == str(point_id)]
    s2 = _DF_STAGE2[_DF_STAGE2["point_id"].astype(str) == str(point_id)]
    return jsonify({
        "stage1": [_row(r, "stage1") for _, r in s1.iterrows()],
        "stage2": [_row(r, "stage2") for _, r in s2.iterrows()],
    })


@app.route("/api/available/<point_id>")
def api_available(point_id: str):
    result = {"pred_tifs": {}, "sat_tifs": {}}
    if _OUTPUT_DIR:
        for year in _HEIGHT_YEARS:
            result["pred_tifs"][year] = bool(list(_OUTPUT_DIR.glob(
                f"{point_id}_*/height/{year}/pred/img_{point_id}_pred.tif"
            )))
            result["sat_tifs"][year] = bool(list(_OUTPUT_DIR.glob(
                f"{point_id}_*/height/{year}/sat/S2/img_{point_id}_??.tif"
            )))
    return jsonify(result)


@app.route("/api/render/plot")
def api_render_plot():
    """
    Render one plot. Returns {"image": "<base64 png>"}.

    Views
    -----
    esri   — ESRI satellite tiles (always available, live fetch)
    height — T-SwinUNet pred.tif colourised; full point neighbourhood shown
    sat    — GEE S2 RGB from saved TIFs; full point neighbourhood shown
    """
    point_id   = request.args.get("point_id",  "")
    stage      = request.args.get("stage",      "stage1")
    plot_index = int(request.args.get("plot_index", 0))
    cluster_id = request.args.get("cluster_id", None)
    year       = int(request.args.get("year",   _HEIGHT_YEARS[0] if _HEIGHT_YEARS else 0))
    view       = request.args.get("view",       "esri")
    show_poly  = request.args.get("show_polygon",  "1") == "1"
    show_mask  = request.args.get("show_sam_mask", "1") == "1"
    zoom       = int(request.args.get("zoom", 19))
    pad        = float(request.args.get("pad", 0.0003))
    OUT        = 500   # output image size

    # Look up row
    df   = _DF_STAGE1 if stage == "stage1" else _DF_STAGE2
    sel  = df["point_id"].astype(str) == str(point_id)
    sel &= df["plot_index"].astype(int) == plot_index
    if cluster_id is not None and "cluster_id" in df.columns:
        sel &= df["cluster_id"].astype(str) == str(cluster_id)
    rows = df[sel]
    if rows.empty:
        return jsonify({"error": f"plot {point_id}/{stage}/{plot_index} not found"}), 404
    row = rows.iloc[0]

    polygon_coords = _wkt_to_coords(str(row.get("polygon_wkt", "")))
    if not polygon_coords:
        return jsonify({"error": "invalid polygon WKT"}), 400

    poly_bbox = _coords_bbox(polygon_coords)   # (west, south, east, north)

    height_class = _safe_str(row.get(f"height_class_{year}", None))
    sam_mask_wkt = str(row.get("sam_mask_wkt", ""))

    # ── Build background + determine render bbox ──────────────────────────
    bg      = None
    bg_bbox = None   # (west, south, east, north) that bg image covers

    if view == "height":
        bg, bg_bbox = _build_height_bg(point_id, year, OUT, poly_bbox)

    elif view == "sat":
        bg, bg_bbox = _build_sat_bg(point_id, year, OUT)

    # ESRI is the default AND the fallback for height/sat when files missing
    if bg is None:
        pw, ps, pe, pn = poly_bbox
        esri_west  = pw - pad
        esri_south = ps - pad
        esri_east  = pe + pad
        esri_north = pn + pad
        bg      = _build_esri_bg(esri_west, esri_south, esri_east, esri_north, zoom, OUT)
        bg_bbox = (esri_west, esri_south, esri_east, esri_north)

    if bg is None:
        bg      = np.full((OUT, OUT, 3), 40, dtype=np.uint8)
        pw, ps, pe, pn = poly_bbox
        bg_bbox = (pw - pad, ps - pad, pe + pad, pn + pad)

    # ── Overlay features ──────────────────────────────────────────────────
    bw, bs, be, bn = bg_bbox
    rendered = _render_on_background(
        canvas        = bg,
        west=bw, south=bs, east=be, north=bn,
        polygon_coords = polygon_coords,
        sam_mask_wkt   = sam_mask_wkt,
        height_class   = height_class,
        show_polygon   = show_poly,
        show_sam_mask  = show_mask,
    )

    return jsonify({"image": _img_to_b64(rendered)})


@app.route("/api/render/overview")
def api_render_overview():
    """
    Render all plots for a point on one background.
    Supports esri / height / sat views same as single-plot render.
    """
    import cv2

    point_id   = request.args.get("point_id", "")
    stage      = request.args.get("stage", "both")
    year       = int(request.args.get("year", _HEIGHT_YEARS[0] if _HEIGHT_YEARS else 0))
    view       = request.args.get("view", "esri")
    zoom       = int(request.args.get("zoom", 18))
    show_poly  = request.args.get("show_polygon",    "1") == "1"
    show_mask  = request.args.get("show_sam_mask",   "1") == "1"
    show_val   = request.args.get("show_height_val", "1") == "1"
    OUT        = 900

    # Collect all plot rows
    dfs = []
    if stage in ("stage1", "both"):
        s1 = _DF_STAGE1[_DF_STAGE1["point_id"].astype(str) == str(point_id)]
        dfs.append(("stage1", s1))
    if stage in ("stage2", "both"):
        s2 = _DF_STAGE2[_DF_STAGE2["point_id"].astype(str) == str(point_id)]
        dfs.append(("stage2", s2))

    all_coords = []
    for _, df in dfs:
        for _, row in df.iterrows():
            c = _wkt_to_coords(str(row.get("polygon_wkt", "")))
            if c:
                all_coords.extend(c)

    if not all_coords:
        return jsonify({"error": "no valid polygons for this point"}), 400

    pw, ps, pe, pn = _coords_bbox(all_coords)
    pad   = 0.001
    ow, os_, oe, on_ = pw - pad, ps - pad, pe + pad, pn + pad

    # ── Background ─────────────────────────────────────────────────────────
    bg = bg_bbox = None

    if view == "height":
        bg, bg_bbox = _build_height_bg(point_id, year, OUT, (pw, ps, pe, pn))

    elif view == "sat":
        bg, bg_bbox = _build_sat_bg(point_id, year, OUT)

    if bg is None:
        bg      = _build_esri_bg(ow, os_, oe, on_, zoom, OUT)
        bg_bbox = (ow, os_, oe, on_)

    if bg is None:
        bg      = np.full((OUT, OUT, 3), 30, dtype=np.uint8)
        bg_bbox = (ow, os_, oe, on_)

    bw, bs, be, bn = bg_bbox
    h, w = bg.shape[:2]

    # ── Draw all plots ──────────────────────────────────────────────────────
    for stg, df in dfs:
        for _, row in df.iterrows():
            coords = _wkt_to_coords(str(row.get("polygon_wkt", "")))
            if not coords:
                continue

            hclass   = _safe_str(row.get(f"height_class_{year}", None))
            height_m = _safe_float(row.get(f"height_m_{year}", None))
            color    = _class_color(hclass)

            pts = np.array(
                [_geo_to_px(lon, lat, bw, bs, be, bn, w, h) for lon, lat in coords],
                dtype=np.int32
            )

            # SAM mask overlay if requested
            if show_mask:
                mask = None
                mwkt = str(row.get("sam_mask_wkt", ""))

                if mwkt.strip() not in ("nan", "None", ""):
                    bc = _wkt_to_coords(mwkt)
                    if bc:
                        bp = np.array([_geo_to_px(lon, lat, bw, bs, be, bn, w, h)
                                       for lon, lat in bc], dtype=np.int32)
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask, [bp], 1)

                if mask is not None and mask.sum() > 0:
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                          interpolation=cv2.INTER_NEAREST)
                    alpha = 0.35
                    c_arr = np.array(color, dtype=np.float32)
                    m = (mask > 0)
                    bg_f = bg.astype(np.float32)
                    bg_f[m] = bg_f[m] * (1 - alpha) + c_arr * alpha
                    bg = np.clip(bg_f, 0, 255).astype(np.uint8)

            # Polygon outline
            if show_poly:
                outline_color = (0, 220, 255) if stg == "stage1" else (255, 200, 0)
                cv2.polylines(bg, [pts], isClosed=True, color=outline_color, thickness=1)

            # Height label at centroid
            if show_val and height_m is not None:
                cx = int(np.mean([p[0] for p in pts]))
                cy = int(np.mean([p[1] for p in pts]))
                label = f"{height_m:.0f}m"
                font  = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.28
                (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
                cv2.rectangle(bg, (cx - tw//2 - 2, cy - th - 2),
                              (cx + tw//2 + 2, cy + 2), (0, 0, 0), -1)
                cv2.putText(bg, label, (cx - tw//2, cy),
                            font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Legend
    legend_y = 10
    for label, col in HEIGHT_CLASS_COLORS.items():
        cv2.rectangle(bg, (w - 145, legend_y), (w - 125, legend_y + 12), col, -1)
        cv2.rectangle(bg, (w - 145, legend_y), (w - 125, legend_y + 12), (200, 200, 200), 1)
        cv2.putText(bg, label, (w - 120, legend_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 230, 230), 1)
        legend_y += 17

    return jsonify({"image": _img_to_b64(bg)})


@app.route("/api/colorbar")
def api_colorbar():
    """Return inferno colorbar PNG for the height raster scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(4, 0.45))
    fig.subplots_adjust(bottom=0.5)
    norm = mcolors.Normalize(vmin=0, vmax=_HEIGHT_VIS_MAX)
    try:
        cmap = matplotlib.colormaps["inferno"]
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("inferno")
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="horizontal"
    )
    cb.set_label(f"Height (m)  [0 – {_HEIGHT_VIS_MAX:.0f} m]", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    fig.patch.set_facecolor("#1a1a2e")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return jsonify({"image": base64.b64encode(buf.getvalue()).decode()})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    global _EXCEL_PATH, _OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Pipeline Output Visualiser")
    parser.add_argument("--excel",      required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--port",       type=int, default=5000)
    parser.add_argument("--host",       default="127.0.0.1")
    args = parser.parse_args()

    _EXCEL_PATH = Path(args.excel).resolve()
    if not _EXCEL_PATH.exists():
        print(f"Error: Excel file not found: {_EXCEL_PATH}")
        sys.exit(1)

    _OUTPUT_DIR = (
        Path(args.output_dir).resolve() if args.output_dir
        else _EXCEL_PATH.parent
    )

    sys.path.insert(0, str(_EXCEL_PATH.parent.parent))
    sys.path.insert(0, str(Path(__file__).parent))

    _load_excel(_EXCEL_PATH)

    print(f"\n{'='*60}")
    print(f"  Pipeline Output Visualiser")
    print(f"  Excel  : {_EXCEL_PATH}")
    print(f"  OutDir : {_OUTPUT_DIR}")
    print(f"  Points : {len(_DF_POINTS)}")
    print(f"  Open   : http://{args.host}:{args.port}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()