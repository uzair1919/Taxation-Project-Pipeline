"""
stage1.py
=========
Stage 1 — Global alignment.

Takes the raw GeoJSON plots from the score pipeline, translates and
optionally scales the entire set to minimise road overlap, and returns
the corrected GeoDataFrame plus alignment stats.

Public API
----------
    run_stage1(plot_geojson, geobounds, streets_shp, params, verbose)
        → (result_gdf, original_geoms, streets_utm, road_surface, stats)

    save_stage1_plots(original_geoms, result_gdf, streets_utm, road_surface,
                      stats, out_dir)

    write_stage1_geojson(original_geoms, result_gdf, point_meta, out_path)
        → GeoJSON file path
"""

import os
import socket
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from shapely.affinity import translate, scale
import contextily as cx
from pyproj import Transformer

from refinement_utils.config import (
    UTM_EPSG, WGS84_EPSG, WEB_MERCATOR_EPSG,
    ESRI_URL, STAGE1, VIZ, 
)
from refinement_utils.road_utils import build_road_surface

warnings.filterwarnings("ignore")

_UTM = UTM_EPSG


# ===========================================================================
# Core alignment
# ===========================================================================

def run_stage1(
    plot_geojson,
    geobounds,
    streets_shp,
    params: dict = None,
    verbose: bool = True,
):
    """
    Align all plots in plot_geojson to the street network.

    Parameters
    ----------
    plot_geojson : dict | GeoDataFrame
        Raw plot polygons from the score pipeline (WGS84).
    geobounds : dict
        {"north", "south", "east", "west"} bounding box of the pipeline context.
    streets_shp : str
        Absolute path to the OSM roads shapefile.
    params : dict, optional
        Override any STAGE1 config keys.
    verbose : bool

    Returns
    -------
    result_gdf      : GeoDataFrame (WGS84) — corrected plots
    original_geoms  : list of Shapely UTM geoms — before correction
    streets_utm     : GeoDataFrame (UTM) — loaded road segments
    road_surface    : Shapely geometry (UTM) — buffered road union
    stats           : dict — alignment statistics + scale_anchor
    """
    cfg = {**STAGE1, **(params or {})}

    n, s, e, w = (geobounds["north"], geobounds["south"],
                  geobounds["east"],  geobounds["west"])

    # ── Load and project plots ─────────────────────────────────────────────
    if isinstance(plot_geojson, dict):
        plots_gdf = gpd.GeoDataFrame.from_features(plot_geojson["features"])
    elif not isinstance(plot_geojson, gpd.GeoDataFrame):
        plots_gdf = gpd.GeoDataFrame(plot_geojson)
    else:
        plots_gdf = plot_geojson.copy()

    plots_utm = (
        plots_gdf
        .set_crs(epsg=WGS84_EPSG, allow_override=True)
        .to_crs(epsg=_UTM)
    )
    plots_utm = plots_utm[
        plots_utm.geometry.is_valid & ~plots_utm.geometry.is_empty
    ].reset_index(drop=True)

    if plots_utm.empty:
        raise ValueError("Stage 1: no valid plot geometries found.")

    # ── Load streets ───────────────────────────────────────────────────────
    streets_raw = gpd.read_file(streets_shp, bbox=(w, s, e, n))
    if streets_raw.empty:
        raise ValueError("Stage 1: no streets found within the bounding box.")
    streets_utm  = streets_raw.to_crs(epsg=_UTM)
    road_surface = build_road_surface(streets_utm)

    # ── Loss function ──────────────────────────────────────────────────────
    prox_w = cfg["proximity_weight"]

    def compute_loss(geom_list):
        n_overlap, prox_dists = 0, []
        for geom in geom_list:
            if geom.intersects(road_surface):
                n_overlap += 1
            else:
                prox_dists.append(geom.distance(road_surface))
        mean_prox = float(np.mean(prox_dists)) if prox_dists else 0.0
        return float(n_overlap) + prox_w * mean_prox, n_overlap, mean_prox

    # ── Translation search ─────────────────────────────────────────────────
    def run_step_search(geom_list, label="step"):
        current = list(geom_list)
        tdx = tdy = 0.0
        step = cfg["initial_step_m"]
        itr  = 0
        while step >= cfg["min_step_m"]:
            cur_loss = compute_loss(current)[0]
            best = cur_loss; bdx = bdy = 0.0; bdir = None
            for dx, dy, nm in [(step,0,"+E"),(-step,0,"-E"),(0,step,"+N"),(0,-step,"-N")]:
                tl = compute_loss([translate(g, xoff=dx, yoff=dy) for g in current])[0]
                if tl < best:
                    best, bdx, bdy, bdir = tl, dx, dy, nm
            if bdir:
                current = [translate(g, xoff=bdx, yoff=bdy) for g in current]
                tdx += bdx; tdy += bdy; itr += 1
                if verbose:
                    l, no, mp = compute_loss(current)
                    print(f"    [{label}] step={step:.3f}m {bdir:4s} "
                          f"overlap={no} prox={mp:.2f}m loss={l:.4f}")
            else:
                step /= 2.0
        return current, tdx, tdy, itr

    # ── Scale search ───────────────────────────────────────────────────────
    def run_stretch_search(geom_list, anchor):
        current = list(geom_list)
        ax, ay  = anchor.x, anchor.y
        cum_sx = cum_sy = 1.0
        ss  = cfg["initial_scale_step"]
        itr = 0
        while ss >= cfg["min_scale_step"]:
            cur_loss = compute_loss(current)[0]
            best = cur_loss; bg = None; bl = None; bdsx = bdsy = 1.0
            for sx, sy, lbl in [
                (1+ss, 1.0, "+X"), (1-ss, 1.0, "-X"),
                (1.0, 1+ss, "+Y"), (1.0, 1-ss, "-Y"),
            ]:
                if not (cfg["min_scale_factor"] <= cum_sx*sx <= cfg["max_scale_factor"]):
                    continue
                if not (cfg["min_scale_factor"] <= cum_sy*sy <= cfg["max_scale_factor"]):
                    continue
                trial = [scale(g, xfact=sx, yfact=sy, origin=(ax, ay)) for g in current]
                if not all(g.is_valid and not g.is_empty for g in trial):
                    continue
                tl = compute_loss(trial)[0]
                if tl < best:
                    best, bg, bl, bdsx, bdsy = tl, trial, lbl, sx, sy
            if bg:
                current = bg; cum_sx *= bdsx; cum_sy *= bdsy; itr += 1
                if verbose:
                    l, no, mp = compute_loss(current)
                    print(f"    [stretch] ss={ss:.4f} {bl} "
                          f"sx={cum_sx:.4f} sy={cum_sy:.4f} "
                          f"overlap={no} prox={mp:.2f}m loss={l:.4f}")
            else:
                ss /= 2.0
        return current, cum_sx, cum_sy, itr

    # ── Run ────────────────────────────────────────────────────────────────
    original_geoms = plots_utm.geometry.tolist()

    # The scale anchor is stored in stats so stage 2 can re-apply the
    # same transform to tile-fetched plots.
    scale_anchor = unary_union(original_geoms).centroid

    loss_before, n_before, p_before = compute_loss(original_geoms)
    if verbose:
        print(f"  [S1] Initial: overlap={n_before} prox={p_before:.2f}m "
              f"loss={loss_before:.4f}")
        print("  [S1] Phase 1: step search")

    current_geoms, total_dx, total_dy, s1 = run_step_search(original_geoms, "step")
    _, n_p1, _ = compute_loss(current_geoms)

    total_sx = total_sy = 1.0
    s2 = s3 = 0
    pre_stretch = list(current_geoms)

    if n_p1 > 0:
        if verbose: print(f"  [S1] Phase 2: stretch ({n_p1} overlaps remain)")
        current_geoms, total_sx, total_sy, s2 = run_stretch_search(
            current_geoms, scale_anchor
        )
        loss_p2, _, _ = compute_loss(current_geoms)
        loss_p1, _, _ = compute_loss(pre_stretch)
        if loss_p2 < loss_p1:
            if verbose: print("  [S1] Phase 3: post-stretch step search")
            current_geoms, dx2, dy2, s3 = run_step_search(current_geoms, "post-stretch")
            total_dx += dx2; total_dy += dy2
        else:
            if verbose: print("  [S1] Stretch did not improve — reverting")
            current_geoms = pre_stretch
            total_sx = total_sy = 1.0
    else:
        if verbose: print("  [S1] No overlaps — skipping stretch")

    loss_final, n_final, p_final = compute_loss(current_geoms)

    if loss_final > loss_before:
        if verbose: print("  [S1] WARNING: reverting — final worse than original")
        current_geoms = original_geoms
        total_dx = total_dy = 0.0
        total_sx = total_sy = 1.0
        loss_final, n_final, p_final = loss_before, n_before, p_before

    if verbose:
        print(f"  [S1] Final: overlap={n_final} prox={p_final:.2f}m "
              f"loss={loss_before:.4f}→{loss_final:.4f}")
        print(f"  [S1] dx={total_dx:.2f}m dy={total_dy:.2f}m "
              f"sx={total_sx:.4f} sy={total_sy:.4f}")

    result_gdf = gpd.GeoDataFrame(
        plots_utm.drop(columns="geometry", errors="ignore"),
        geometry=current_geoms, crs=_UTM,
    ).to_crs(epsg=WGS84_EPSG)

    stats = {
        "total_dx_m":        total_dx,
        "total_dy_m":        total_dy,
        "total_sx":          total_sx,
        "total_sy":          total_sy,
        "scale_anchor_x":    scale_anchor.x,
        "scale_anchor_y":    scale_anchor.y,
        "n_overlap_before":  n_before,
        "n_overlap_after":   n_final,
        "prox_before_m":     p_before,
        "prox_after_m":      p_final,
        "loss_before":       loss_before,
        "loss_after":        loss_final,
        "step_iterations":   s1 + s3,
        "stretch_iterations": s2,
        "n_plots":           len(original_geoms),
    }

    return result_gdf, original_geoms, streets_utm, road_surface, stats


# ===========================================================================
# GeoJSON output
# ===========================================================================

def write_stage1_geojson(original_geoms_utm, result_gdf, point_meta, out_path):
    """
    Write a GeoJSON file with both pre- and post-stage-1 plot polygons.

    Each feature carries:
        point_id, region_name, tile_names,
        stage       : "stage1_before" | "stage1_after"
        plot_index  : integer index within the pipeline output

    Parameters
    ----------
    original_geoms_utm : list of Shapely UTM geoms (before alignment)
    result_gdf         : GeoDataFrame WGS84 (after alignment)
    point_meta         : dict with keys: point_id, region_name, tile_names,
                         target_lat, target_lon
    out_path           : str — path to write .geojson
    """
    features = []

    # Before
    orig_gdf_wgs = gpd.GeoDataFrame(
        geometry=original_geoms_utm, crs=_UTM
    ).to_crs(WGS84_EPSG)
    for i, geom in enumerate(orig_gdf_wgs.geometry):
        if geom is None or geom.is_empty:
            continue
        features.append({
            "type": "Feature",
            "geometry": geom.__geo_interface__,
            "properties": {
                "point_id":    point_meta["point_id"],
                "region_name": point_meta.get("region_name", ""),
                "target_lat":  point_meta["target_lat"],
                "target_lon":  point_meta["target_lon"],
                "stage":       "stage1_before",
                "plot_index":  i,
            },
        })

    # After
    for i, geom in enumerate(result_gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        features.append({
            "type": "Feature",
            "geometry": geom.__geo_interface__,
            "properties": {
                "point_id":    point_meta["point_id"],
                "region_name": point_meta.get("region_name", ""),
                "target_lat":  point_meta["target_lat"],
                "target_lon":  point_meta["target_lon"],
                "stage":       "stage1_after",
                "plot_index":  i,
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        import json
        json.dump(geojson, f, indent=2)

    return out_path


# ===========================================================================
# Visualisation
# ===========================================================================

def _fetch_esri(merc_w, merc_s, merc_e, merc_n, verbose=True):
    zoom = VIZ["satellite_zoom"]
    orig = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    img = extent = None
    try:
        for z in (zoom, zoom - 1, zoom - 2):
            try:
                img_raw, ext = cx.bounds2img(
                    merc_w, merc_s, merc_e, merc_n,
                    ll=False, source=ESRI_URL, zoom=z
                )
                if img_raw.shape[2] == 4:
                    img_raw = img_raw[:, :, :3]
                img, extent = img_raw, ext
                if verbose:
                    print(f"    ESRI tiles zoom={z} {img.shape[1]}x{img.shape[0]}px")
                break
            except Exception as exc:
                if verbose:
                    print(f"    zoom={z} failed: {exc}")
    finally:
        socket.setdefaulttimeout(orig)
    return img, extent


def _utm_geom_to_wgs84_patch(geom_utm, **kwargs):
    gw = gpd.GeoDataFrame(geometry=[geom_utm], crs=_UTM).to_crs(WGS84_EPSG).geometry.iloc[0]
    if gw.is_empty:
        return []
    polys = [gw] if gw.geom_type == "Polygon" else list(gw.geoms)
    return [MplPolygon(np.array(p.exterior.coords), closed=True, **kwargs)
            for p in polys]


def save_stage1_plots(original_geoms, result_gdf, streets_utm,
                      road_surface, stats, out_dir, verbose=True):
    """
    Save two stage-1 diagnostic plots into out_dir:
        stage1_utm.png        — UTM coordinate space (fast, no satellite)
        stage1_satellite.png  — ESRI satellite background
    """
    os.makedirs(out_dir, exist_ok=True)
    _save_utm_plot(original_geoms, result_gdf, streets_utm, road_surface,
                   stats, os.path.join(out_dir, "stage1_utm.png"))
    _save_satellite_plot(original_geoms, result_gdf, streets_utm, road_surface,
                         stats, os.path.join(out_dir, "stage1_satellite.png"),
                         verbose=verbose)


def _save_utm_plot(original_geoms, result_gdf, streets_utm,
                   road_surface, stats, out_path):
    result_utm = result_gdf.to_crs(epsg=_UTM)
    road_gdf   = gpd.GeoDataFrame(geometry=[road_surface], crs=_UTM)
    orig_gdf   = gpd.GeoDataFrame(geometry=original_geoms, crs=_UTM)
    b   = orig_gdf.total_bounds
    pad = max(b[2] - b[0], b[3] - b[1]) * 0.08

    fig, axes = plt.subplots(1, 2, figsize=VIZ["stage1_figsize"])
    for ax in axes:
        road_gdf.plot(ax=ax, color="salmon", alpha=0.4)
        streets_utm.plot(ax=ax, color="red", linewidth=0.8, alpha=0.8)
        ax.set_xlim(b[0] - pad, b[2] + pad)
        ax.set_ylim(b[1] - pad, b[3] + pad)
        ax.set_aspect("equal")

    orig_gdf.plot(ax=axes[0], facecolor="steelblue", edgecolor="navy",
                  linewidth=0.4, alpha=0.6)
    axes[0].set_title(
        f"BEFORE\nOverlapping: {stats['n_overlap_before']}/{stats['n_plots']} | "
        f"Loss: {stats['loss_before']:.3f}", fontsize=12
    )
    result_utm.plot(ax=axes[1], facecolor="limegreen", edgecolor="darkgreen",
                    linewidth=0.4, alpha=0.6)
    axes[1].set_title(
        f"AFTER  dx={stats['total_dx_m']:.1f}m dy={stats['total_dy_m']:.1f}m "
        f"sx={stats['total_sx']:.3f} sy={stats['total_sy']:.3f}\n"
        f"Overlapping: {stats['n_overlap_after']}/{stats['n_plots']} | "
        f"Loss: {stats['loss_after']:.3f}", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=VIZ["plot_dpi"], bbox_inches="tight")
    plt.close(fig)


def _save_satellite_plot(original_geoms, result_gdf, streets_utm,
                         road_surface, stats, out_path, verbose=True):
    _utm_to_3857 = Transformer.from_crs(f"EPSG:{_UTM}", f"EPSG:{WEB_MERCATOR_EPSG}", always_xy=True)
    _utm_to_wgs  = Transformer.from_crs(f"EPSG:{_UTM}", f"EPSG:{WGS84_EPSG}", always_xy=True)

    orig_gdf = gpd.GeoDataFrame(geometry=original_geoms, crs=_UTM)
    b   = orig_gdf.total_bounds
    pad = max(b[2] - b[0], b[3] - b[1]) * 0.12
    minx, miny, maxx, maxy = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad

    merc_w, merc_s = _utm_to_3857.transform(minx, miny)
    merc_e, merc_n = _utm_to_3857.transform(maxx, maxy)
    wgs_w,  wgs_s  = _utm_to_wgs.transform(minx, miny)
    wgs_e,  wgs_n  = _utm_to_wgs.transform(maxx, maxy)

    img, extent = _fetch_esri(merc_w, merc_s, merc_e, merc_n, verbose=verbose)
    if extent is not None:
        _3857_wgs = Transformer.from_crs(f"EPSG:{WEB_MERCATOR_EPSG}", f"EPSG:{WGS84_EPSG}", always_xy=True)
        ew, es = _3857_wgs.transform(extent[0], extent[2])
        ee, en = _3857_wgs.transform(extent[1], extent[3])
    else:
        ew, es, ee, en = wgs_w, wgs_s, wgs_e, wgs_n

    road_wgs = gpd.GeoDataFrame(geometry=[road_surface], crs=_UTM).to_crs(WGS84_EPSG)
    result_geoms_utm = result_gdf.to_crs(_UTM).geometry.tolist()

    fig, axes = plt.subplots(1, 2, figsize=VIZ["stage2_figsize"])
    for ax in axes:
        if img is not None:
            ax.imshow(img, extent=(ew, ee, es, en),
                      aspect="auto", origin="upper",
                      interpolation="lanczos", zorder=0)
        road_wgs.plot(ax=ax, color="red", alpha=0.30, zorder=1)
        streets_utm.to_crs(WGS84_EPSG).plot(ax=ax, color="red",
                                             linewidth=0.5, alpha=0.6, zorder=2)
        ax.set_xlim(wgs_w, wgs_e); ax.set_ylim(wgs_s, wgs_n)
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)

    for g in original_geoms:
        for p in _utm_geom_to_wgs84_patch(
            g, facecolor=(0.27, 0.51, 0.71, 0.35),
            edgecolor=(0.08, 0.20, 0.45, 1.0), linewidth=0.8, zorder=3
        ):
            axes[0].add_patch(p)
    axes[0].set_title(
        f"BEFORE\nOverlapping: {stats['n_overlap_before']}/{stats['n_plots']} | "
        f"Loss: {stats['loss_before']:.3f}", fontsize=12
    )

    for g in result_geoms_utm:
        for p in _utm_geom_to_wgs84_patch(
            g, facecolor=(0.0, 0.78, 0.2, 0.35),
            edgecolor=(0.0, 0.45, 0.1, 1.0), linewidth=0.8, zorder=3
        ):
            axes[1].add_patch(p)
    axes[1].set_title(
        f"AFTER  dx={stats['total_dx_m']:.1f}m dy={stats['total_dy_m']:.1f}m "
        f"sx={stats['total_sx']:.3f} sy={stats['total_sy']:.3f}\n"
        f"Overlapping: {stats['n_overlap_after']}/{stats['n_plots']} | "
        f"Loss: {stats['loss_after']:.3f}", fontsize=12
    )
    axes[1].legend(handles=[
        mpatches.Patch(facecolor=(0.27, 0.51, 0.71, 0.4),
                       edgecolor=(0.08, 0.20, 0.45), label="Original"),
        mpatches.Patch(facecolor=(0.0, 0.78, 0.2, 0.4),
                       edgecolor=(0.0, 0.45, 0.1),   label="Adjusted"),
        mpatches.Patch(facecolor="red", alpha=0.35, label="Road surface"),
    ], loc="lower right", fontsize=9, framealpha=0.85)
    fig.text(0.01, 0.005, "Satellite © Esri, Maxar, Earthstar Geographics",
             fontsize=6, color="grey")
    plt.tight_layout()
    plt.savefig(out_path, dpi=VIZ["plot_dpi"], bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  [S1] Satellite plot saved -> {out_path}")