"""
pipeline/output_writer.py
==========================
Writes the final clean Excel workbook from a list of PointResult objects.

Sheet layout
------------
  plots_stage1  — one row per stage-1 plot + SAM output + height columns
  plots_stage2  — one row per stage-2 plot + SAM output + height columns
  points        — one row per input GPS point (summary)
  debug_info    — only present when debug_mode=True

Height columns (dynamically added per year in HEIGHT_YEARS):
  height_m_{year}     — mean height in metres within SAM/polygon footprint
  height_class_{year} — storey class label e.g. "2-3 storeys"
  height_src_{year}   — "sam_mask" | "polygon" | "failed"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def _sam(rec, attr, default=None):
    sr = getattr(rec, "sam_result", None)
    if sr is None:
        return default
    return getattr(sr, attr, default)


def _height(rec, year: int, attr: str, default=None):
    hr = getattr(rec, "height_results", None)
    if not hr:
        return default
    yr = hr.get(year)
    if yr is None:
        return default
    return getattr(yr, attr, default)


def _polygon_area_m2(wkt: str) -> Optional[float]:
    """
    Compute the geodesic area of a WGS84 WKT polygon in square metres.
    Uses pyproj.Geod (WGS84 ellipsoid) — accurate regardless of location.
    Returns None on any parse/computation failure.
    """
    if not wkt:
        return None
    try:
        from shapely import wkt as shapely_wkt
        from pyproj import Geod
        geod    = Geod(ellps="WGS84")
        polygon = shapely_wkt.loads(wkt)
        area, _ = geod.geometry_area_perimeter(polygon)
        return abs(area)
    except Exception:
        return None


def _covered_pct(rec) -> Optional[float]:
    """
    Percentage of the plot boundary area covered by the SAM mask.
    = sam_area_m2 / plot_area_m2 × 100
    Returns None if either area is missing or plot area is zero.
    """
    sam_area  = _sam(rec, "sam_area_m2")
    plot_area = _polygon_area_m2(getattr(rec, "polygon_wkt", ""))
    if sam_area is None or plot_area is None or plot_area == 0:
        return None
    return round(sam_area / plot_area * 100, 2)


# ---------------------------------------------------------------------------
# Static column definitions
# ---------------------------------------------------------------------------

_SAM_COLS = [
    ("sam_status",         lambda r: _sam(r, "status")),
    ("sam_score",          lambda r: _sam(r, "sam_score")),
    ("sam_iou",            lambda r: _sam(r, "sam_iou")),
    ("sam_area_m2",        lambda r: _sam(r, "sam_area_m2")),
    ("covered_pct",        lambda r: _covered_pct(r)),
    ("sam_bbox_wkt",       lambda r: _sam(r, "sam_bbox_wkt")),
    ("sam_mask_wkt",       lambda r: _sam(r, "mask_geo_wkt")),
    ("sam_mask_path",      lambda r: _sam(r, "mask_path")),
    ("sam_rotation_deg",   lambda r: _sam(r, "rotation_angle_deg")),
    ("sam_error",          lambda r: _sam(r, "error")),
]

_STAGE1_BASE = [
    ("point_id",      lambda r: r.point_id),
    ("plot_index",    lambda r: r.plot_index),
    ("polygon_wkt",   lambda r: r.polygon_wkt),
    ("plot_area_m2",  lambda r: _polygon_area_m2(getattr(r, "polygon_wkt", ""))),
]

_STAGE2_BASE = [
    ("point_id",      lambda r: r.point_id),
    ("cluster_id",    lambda r: r.cluster_id),
    ("plot_index",    lambda r: r.plot_index),
    ("polygon_wkt",   lambda r: r.polygon_wkt),
    ("plot_area_m2",  lambda r: _polygon_area_m2(getattr(r, "polygon_wkt", ""))),
]

_POINTS_COLS = [
    ("point_id",       lambda p: p.point_id),
    ("name",           lambda p: p.name),
    ("latitude",       lambda p: p.latitude),
    ("longitude",      lambda p: p.longitude),
    ("status",         lambda p: p.status),
    ("n_stage1_plots", lambda p: p.n_stage1_plots),
    ("n_clusters",     lambda p: p.n_clusters),
    ("n_stage2_plots", lambda p: p.n_stage2_plots),
    ("error",          lambda p: p.error or ""),
]


def _build_height_cols(years: List[int]) -> list:
    """
    Build dynamic height column definitions for the requested years.
    Columns per year (in order):
        height_m_{year}     — mean height in metres
        height_class_{year} — storey class label
        height_src_{year}   — "sam_mask" | "polygon" | "failed"
    """
    cols = []
    for year in sorted(years):
        y = year
        cols.append((f"height_m_{y}",     lambda r, yr=y: _height(r, yr, "height_m")))
        cols.append((f"height_class_{y}", lambda r, yr=y: _height(r, yr, "height_class")))
        cols.append((f"height_src_{y}",   lambda r, yr=y: _height(r, yr, "source")))
    return cols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_final_excel(
    point_results: List,
    output_path:   Path,
    debug_mode:    bool = False,
    height_years:  Optional[List[int]] = None,
) -> None:
    """
    Build and save the final clean Excel workbook.

    If the output file already exists (e.g. from a previous run), the new
    rows are MERGED with the existing ones rather than overwriting them.
    Any point_id present in both the existing file and the current batch is
    replaced by the new data (handles re-runs after failures).

    Parameters
    ----------
    point_results : list of PointResult objects processed in this run
    output_path   : destination .xlsx file
    debug_mode    : if True, add a debug_info sheet
    height_years  : years for which height columns are included; None = omit
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError(
            "openpyxl is required.  Install with:  pip install openpyxl"
        ) from None

    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_stage1 = [rec for pr in point_results for rec in pr.stage1_plots]
    all_stage2 = [rec for pr in point_results for rec in pr.stage2_plots]

    n_failed = sum(1 for pr in point_results if getattr(pr, "status", "") == "error")
    if n_failed:
        failed_ids = [getattr(pr, "point_id", "?") for pr in point_results
                      if getattr(pr, "status", "") == "error"]
        logger.warning(
            f"{n_failed}/{len(point_results)} point(s) failed refinement and produced no plots: "
            + ", ".join(failed_ids)
        )

    h_cols      = _build_height_cols(height_years or [])
    stage1_cols = _STAGE1_BASE + _SAM_COLS + h_cols
    stage2_cols = _STAGE2_BASE + _SAM_COLS + h_cols

    df_stage1_new = (
        _records_to_df(all_stage1, stage1_cols)
        if all_stage1
        else pd.DataFrame(columns=[c[0] for c in stage1_cols])
    )
    df_stage2_new = (
        _records_to_df(all_stage2, stage2_cols)
        if all_stage2
        else pd.DataFrame(columns=[c[0] for c in stage2_cols])
    )
    df_points_new = _records_to_df(point_results, _POINTS_COLS)

    # ── Merge with existing Excel (if present) ────────────────────────────
    # Drop rows for any point_id being written now (handles re-runs), then
    # prepend the surviving old rows so output is always cumulative.
    new_ids = {getattr(pr, "point_id", None) for pr in point_results}

    df_stage1 = df_stage1_new
    df_stage2 = df_stage2_new
    df_points = df_points_new
    df_debug_existing = pd.DataFrame()

    if output_path.exists():
        try:
            with pd.ExcelFile(output_path, engine="openpyxl") as xf:
                def _load_existing(sheet: str) -> pd.DataFrame:
                    if sheet not in xf.sheet_names:
                        return pd.DataFrame()
                    df = pd.read_excel(xf, sheet_name=sheet, dtype=str)
                    if "point_id" in df.columns:
                        df = df[~df["point_id"].isin(new_ids)]
                    return df

                ex_s1     = _load_existing("plots_stage1")
                ex_s2     = _load_existing("plots_stage2")
                ex_pts    = _load_existing("points")
                ex_debug  = _load_existing("debug_info")

            def _merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
                if existing.empty:
                    return new
                return pd.concat([existing, new], ignore_index=True)

            df_stage1        = _merge(ex_s1,  df_stage1_new)
            df_stage2        = _merge(ex_s2,  df_stage2_new)
            df_points        = _merge(ex_pts, df_points_new)
            df_debug_existing = ex_debug

            kept = len(ex_s1) + len(ex_s2)
            logger.info(
                f"Merging with existing Excel: "
                f"kept {kept} rows from previous run(s), "
                f"adding {len(all_stage1) + len(all_stage2)} new rows"
            )
        except Exception as exc:
            logger.warning(
                f"Could not read existing Excel for merge ({exc}); "
                "writing current batch only — previous data may be lost."
            )

    if df_stage1.empty and df_stage2.empty:
        logger.error(
            "OUTPUT IS EMPTY — no plot records exist for any point. "
            "All points likely failed during refinement. "
            "Check the pipeline.log and per-point error.txt files for the root cause."
        )

    logger.info(
        f"Writing Excel → {output_path}  "
        f"({len(df_stage1)} stage1 rows, {len(df_stage2)} stage2 rows, "
        f"{len(df_points)} points)"
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_stage1.to_excel(writer, sheet_name="plots_stage1", index=False)
        df_stage2.to_excel(writer, sheet_name="plots_stage2", index=False)
        df_points.to_excel(writer, sheet_name="points",       index=False)

        if debug_mode:
            df_debug_new = _build_debug_df(point_results)
            df_debug = (
                pd.concat([df_debug_existing, df_debug_new], ignore_index=True)
                if not df_debug_existing.empty and not df_debug_new.empty
                else (df_debug_new if not df_debug_new.empty else df_debug_existing)
            )
            if not df_debug.empty:
                df_debug.to_excel(writer, sheet_name="debug_info", index=False)

    size_kb = output_path.stat().st_size // 1024
    logger.info(f"Excel written: {output_path} ({size_kb} KB)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _records_to_df(records: list, col_defs: list):
    import pandas as pd
    rows = []
    for rec in records:
        row = {}
        for col_name, extractor in col_defs:
            try:
                row[col_name] = extractor(rec)
            except Exception:
                row[col_name] = None
        rows.append(row)
    return pd.DataFrame(rows, columns=[c[0] for c in col_defs])


def _build_debug_df(point_results: list):
    import pandas as pd
    rows = []
    for pr in point_results:
        if pr.debug_point is None:
            continue
        s1 = pr.debug_point.get("stage1") or {}
        if s1:
            rows.append({
                "point_id":   pr.point_id,
                "debug_type": "stage1_summary",
                **{f"s1_{k}": v for k, v in s1.items()},
            })
        for cl in (pr.debug_point.get("clusters") or []):
            rows.append({
                "point_id":   pr.point_id,
                "debug_type": "stage2_cluster",
                **{f"s2_{k}": v for k, v in cl.items()},
            })
    if not rows:
        return pd.DataFrame()
    all_keys = sorted({k for r in rows for k in r})
    return pd.DataFrame([{k: r.get(k) for k in all_keys} for r in rows])