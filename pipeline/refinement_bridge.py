"""
pipeline/refinement_bridge.py
==============================
Wraps stage1 + stage2 refinement and returns PointResult objects.

Changes in this version
-----------------------
* Respects cfg.SAVE_REFINEMENT_PLOTS — diagnostic PNGs are optional.
* Respects cfg.SAVE_GEOJSON — GeoJSON files are optional.
* Respects cfg.REFINEMENT_VERBOSE — suppresses sub-pipeline stdout when False.
* PointResult and PlotRecord are defined here; sam_result field is included.
"""

from __future__ import annotations

import json
import os
import traceback
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import geopandas as gpd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlotRecord:
    """One refined plot polygon — one row in the final Excel."""

    point_id:    str
    stage:       str            # "stage1" | "stage2"
    cluster_id:  Optional[int]  # None for stage1
    plot_index:  int
    polygon_wkt: str            # WGS84 POLYGON WKT

    debug_stats: Optional[dict] = field(default=None, repr=False)

    # Attached by sam_bridge after SAM2 inference; None until then.
    sam_result:  Optional[object] = field(default=None, repr=False)


@dataclass
class PointResult:
    """Everything produced by the refinement pipeline for one GPS point."""

    point_id:       str
    name:           str
    latitude:       float
    longitude:      float
    status:         str          # "ok" | "error"
    error:          Optional[str]

    n_stage1_plots: int
    n_clusters:     int
    n_stage2_plots: int

    stage1_plots:   List[PlotRecord]
    stage2_plots:   List[PlotRecord]

    debug_point:    Optional[dict] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _geoms_to_wkt(geoms_utm, utm_epsg: int, wgs84_epsg: int) -> list:
    from pipeline.utils.geo import shapely_geom_to_wkt
    if not geoms_utm:
        return []
    gdf = gpd.GeoDataFrame(geometry=geoms_utm, crs=utm_epsg).to_crs(wgs84_epsg)
    return [shapely_geom_to_wkt(g) for g in gdf.geometry]


def _s1_debug(stats: dict, n: int) -> dict:
    return {
        "n_plots":          n,
        "n_overlap_before": stats.get("n_overlap_before"),
        "n_overlap_after":  stats.get("n_overlap_after"),
        "loss_before":      round(stats.get("loss_before", 0), 4),
        "loss_after":       round(stats.get("loss_after",  0), 4),
        "dx_m":             round(stats.get("total_dx_m", 0), 3),
        "dy_m":             round(stats.get("total_dy_m", 0), 3),
        "sx":               round(stats.get("total_sx", 1), 5),
        "sy":               round(stats.get("total_sy", 1), 5),
    }


def _cluster_debug(cr) -> dict:
    s = cr.stats
    return {
        "cluster_id":       cr.cluster_id,
        "n_plots_in":       s.get("cluster_size"),
        "n_plots_out":      s.get("n_plots_refined"),
        "loss_before":      round(s["loss_before"], 4) if s.get("loss_before") is not None else None,
        "loss_after":       round(s["loss_after"],  4) if s.get("loss_after")  is not None else None,
        "rdx_m":            round(s.get("rdx_m", 0), 3),
        "rdy_m":            round(s.get("rdy_m", 0), 3),
        "centroid_lat":     s.get("centroid_lat"),
        "centroid_lon":     s.get("centroid_lon"),
        "dist_to_target_m": round(s["dist_to_target_m"], 1) if s.get("dist_to_target_m") is not None else None,
        "error":            s.get("error"),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_refinement(
    lat: float,
    lon: float,
    point_id: str,
    name: str,
    out_dir: str,
    run_s1: bool = True,
    run_s2: bool = True,
    stage1_params: Optional[dict] = None,
    stage2_params: Optional[dict] = None,
    cfg=None,
    verbose: bool = True,
    debug_mode: bool = False,
) -> PointResult:
    """
    Run stage-1 and/or stage-2 for one GPS point.
    Returns PointResult; never raises.
    """
    # ── Resolve config values ────────────────────────────────────────────
    from refinement_utils.config import (
        STREETS_SHP       as _STREETS,
        UTM_EPSG          as _UTM,   # authoritative UTM EPSG — used by stage2 internally
        WGS84_EPSG        as _WGS,
        TILE_OVERLAP_PERCENT,
        STAGE1            as _S1,
        STAGE2            as _S2,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    streets_shp = str(getattr(cfg, "STREETS_SHP", _STREETS) if cfg else _STREETS)
    utm_epsg    = int(getattr(cfg, "UTM_EPSG",    _UTM)     if cfg else _UTM)
    wgs84_epsg  = int(getattr(cfg, "WGS84_EPSG",  _WGS)     if cfg else _WGS)
    s1_cfg      = {**_S1, **(stage1_params or {})}
    s2_cfg      = {**_S2, **(stage2_params or {})}

    save_plots  = bool(getattr(cfg, "SAVE_REFINEMENT_PLOTS", True) if cfg else True)
    save_geojson = bool(getattr(cfg, "SAVE_GEOJSON",         True) if cfg else True)
    # verbose flag from caller overrides cfg (CLI --quiet takes precedence)
    verbose = verbose and bool(getattr(cfg, "REFINEMENT_VERBOSE", True) if cfg else True)

    streets_abs = os.path.abspath(streets_shp) if not os.path.isabs(streets_shp) else streets_shp

    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    original_cwd = os.getcwd()

    # Survivors
    stage1_plots:   List[PlotRecord] = []
    stage2_plots:   List[PlotRecord] = []
    cluster_debug:  List[dict]       = []
    s1_debug_data:  Optional[dict]   = None
    s1_stats:       dict             = {}
    result_gdf                       = None
    original_geoms                   = []
    error_msg:      Optional[str]    = None

    try:
        from refinement_utils.stage1 import (
            run_stage1, save_stage1_plots, write_stage1_geojson,
        )
        from refinement_utils.stage2 import (
            run_stage2, save_stage2_plots, write_stage2_geojson,
        )
        from refinement_utils.extract_streets import extract_urban_streets_local
        from map_tile_utils import region_manager, get_relevant_tiles, edge_detection
        from map_tile_utils.edge_detection import exec as edge_detect

        # ── Region lookup ────────────────────────────────────────────────
        region_details = region_manager.lookup_coordinate(lat, lon)
        region_name    = region_details["region"]["name"]
        tile_csv       = os.path.abspath(region_details["region"]["csv_path"])
        tile_folder    = os.path.abspath(region_details["region"]["tile_folder"])

        point_meta = {
            "point_id":    point_id,
            "region_name": region_name,
            "target_lat":  lat,
            "target_lon":  lon,
        }

        # ── Stage 1 ──────────────────────────────────────────────────────
        if run_s1:
            if verbose:
                print(f"  [{point_id}] Fetching context tiles …")

            context_img = os.path.join(out_dir, "context.png")
            geobounds = get_relevant_tiles.exec(
                lat, lon,
                CSV_PATH        = tile_csv,
                TILE_FOLDER     = tile_folder,
                OVERLAP_PERCENT = TILE_OVERLAP_PERCENT,
                OUTPUT_NAME     = context_img,
            )

            if verbose:
                print(f"  [{point_id}] Detecting edges …")
            _, plot_geojson = edge_detection.exec(geobounds, context_img)

            if verbose:
                print(f"  [{point_id}] Extracting streets …")
            extract_urban_streets_local(
                geobounds["north"], geobounds["south"],
                geobounds["east"],  geobounds["west"],
                streets_abs,
            )

            if verbose:
                print(f"  [{point_id}] Stage 1: global alignment …")
            result_gdf, original_geoms, streets_utm, road_surface, s1_stats = \
                run_stage1(
                    plot_geojson = plot_geojson,
                    geobounds    = geobounds,
                    streets_shp  = streets_abs,
                    params       = s1_cfg,
                    verbose      = verbose,
                )

            # Optional diagnostic plots
            if save_plots:
                save_stage1_plots(
                    original_geoms, result_gdf, streets_utm,
                    road_surface, s1_stats, plots_dir, verbose=verbose,
                )

            # Optional GeoJSON
            s1_geojson_path = os.path.join(out_dir, "stage1_plots.geojson")
            if save_geojson:
                write_stage1_geojson(
                    original_geoms_utm = original_geoms,
                    result_gdf         = result_gdf,
                    point_meta         = point_meta,
                    out_path           = s1_geojson_path,
                )
                if verbose:
                    print(f"  [{point_id}] stage1_plots.geojson → {s1_geojson_path}")

            # Build PlotRecords — result_gdf is already WGS84
            from pipeline.utils.geo import shapely_geom_to_wkt
            for idx, geom in enumerate(result_gdf.geometry):
                wkt = shapely_geom_to_wkt(geom)
                if not wkt:
                    continue
                dbg = _s1_debug(s1_stats, len(result_gdf)) if debug_mode else None
                stage1_plots.append(PlotRecord(
                    point_id=point_id, stage="stage1",
                    cluster_id=None, plot_index=idx,
                    polygon_wkt=wkt, debug_stats=dbg,
                ))

            if debug_mode:
                s1_debug_data = _s1_debug(s1_stats, len(result_gdf))

        else:
            # stage2-only: load previous stage-1 GeoJSON
            s1_geojson_path = os.path.join(out_dir, "stage1_plots.geojson")
            if not os.path.exists(s1_geojson_path):
                raise FileNotFoundError(
                    f"--stage2-only requires {s1_geojson_path}"
                )
            if verbose:
                print(f"  [{point_id}] Loading stage-1 from previous run …")
            gdf_all        = gpd.read_file(s1_geojson_path)
            after          = gdf_all[gdf_all["stage"] == "stage1_after"].reset_index(drop=True)
            before         = gdf_all[gdf_all["stage"] == "stage1_before"].reset_index(drop=True)
            result_gdf     = after.to_crs(wgs84_epsg)
            # Again use _UTM from refinement_utils.config, not our utm_epsg variable,
            # to match the CRS that stage2 will use internally.
            original_geoms = (
                before.to_crs(_UTM).geometry.tolist() if not before.empty else []
            )
            from shapely.ops import unary_union
            if original_geoms:
                anchor = unary_union(original_geoms).centroid
            else:
                from pyproj import Transformer
                from shapely.geometry import Point
                t = Transformer.from_crs(f"EPSG:{wgs84_epsg}", f"EPSG:{utm_epsg}", always_xy=True)
                anchor = Point(*t.transform(lon, lat))
            s1_stats = {
                "total_dx_m": 0.0, "total_dy_m": 0.0,
                "total_sx": 1.0,   "total_sy": 1.0,
                "scale_anchor_x": anchor.x, "scale_anchor_y": anchor.y,
                "n_plots": len(after),
            }
            from pipeline.utils.geo import shapely_geom_to_wkt
            for idx, geom in enumerate(result_gdf.geometry):
                wkt = shapely_geom_to_wkt(geom)
                if wkt:
                    stage1_plots.append(PlotRecord(
                        point_id=point_id, stage="stage1",
                        cluster_id=None, plot_index=idx, polygon_wkt=wkt,
                    ))

        # ── Stage 2 ──────────────────────────────────────────────────────
        if run_s2 and result_gdf is not None:
            if verbose:
                print(f"  [{point_id}] Stage 2: per-cluster refinement …")

            diag_dir = os.path.abspath(
                os.path.join(plots_dir, "stitch_diagnostics")
            ) if save_plots else None

            cluster_results = run_stage2(
                lat=lat, lon=lon,
                result_gdf     = result_gdf,
                original_geoms = original_geoms,
                stats          = s1_stats,
                streets_shp    = streets_abs,
                tile_csv       = tile_csv,
                tile_folder    = tile_folder,
                edge_detector_fn = edge_detect,
                params         = s2_cfg,
                diag_dir       = diag_dir,
                verbose        = verbose,
            )

            if save_plots:
                save_stage2_plots(cluster_results, point_id, plots_dir, verbose=verbose)

            if save_geojson:
                s2_geojson_path = os.path.join(out_dir, "stage2_plots.geojson")
                write_stage2_geojson(
                    cluster_results = cluster_results,
                    point_meta      = point_meta,
                    out_path        = s2_geojson_path,
                )
                if verbose:
                    print(f"  [{point_id}] stage2_plots.geojson → {s2_geojson_path}")

            # Build PlotRecords
            # IMPORTANT: cr.refined_geoms are in the UTM CRS used internally
            # by stage2.py (_UTM = refinement_utils.config.UTM_EPSG).
            # We must use that same value here — NOT our config.py utm_epsg —
            # otherwise GeoDataFrame labels the geometries with the wrong CRS
            # and to_crs() produces completely wrong WGS84 coordinates.
            from pipeline.utils.geo import shapely_geom_to_wkt
            for cr in cluster_results:
                if not cr.refined_geoms:
                    continue
                refined_wgs = (
                    gpd.GeoDataFrame(geometry=cr.refined_geoms, crs=_UTM)
                    .to_crs(wgs84_epsg)
                )
                dbg = _cluster_debug(cr) if debug_mode else None
                for idx_in_cluster, geom in enumerate(refined_wgs.geometry):
                    wkt = shapely_geom_to_wkt(geom)
                    if not wkt:
                        continue
                    stage2_plots.append(PlotRecord(
                        point_id=point_id, stage="stage2",
                        cluster_id=cr.cluster_id,
                        plot_index=idx_in_cluster,
                        polygon_wkt=wkt, debug_stats=dbg,
                    ))
                if debug_mode and dbg:
                    cluster_debug.append(dbg)

    except Exception as exc:
        error_msg = str(exc)
        print(f"\n  [{point_id}] REFINEMENT ERROR: {exc}")
        print(traceback.format_exc())
        with open(os.path.join(out_dir, "error.txt"), "w") as f:
            f.write(traceback.format_exc())

    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
        try:
            os.chdir(original_cwd)
        except Exception:
            pass

    # ── metadata.json (always written) ──────────────────────────────────
    n_clusters = len({r.cluster_id for r in stage2_plots if r.cluster_id is not None})
    metadata = {
        "point_id":       point_id,
        "name":           name,
        "latitude":       lat,
        "longitude":      lon,
        "status":         "error" if error_msg else "ok",
        "error":          error_msg,
        "n_stage1_plots": len(stage1_plots),
        "n_clusters":     n_clusters,
        "n_stage2_plots": len(stage2_plots),
    }
    if debug_mode and s1_debug_data:
        metadata["stage1_debug"] = s1_debug_data
    if debug_mode and cluster_debug:
        metadata["cluster_debug"] = cluster_debug
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    debug_point = (
        {"stage1": s1_debug_data, "clusters": cluster_debug}
        if debug_mode else None
    )

    return PointResult(
        point_id       = point_id,
        name           = name,
        latitude       = lat,
        longitude      = lon,
        status         = "error" if error_msg else "ok",
        error          = error_msg,
        n_stage1_plots = len(stage1_plots),
        n_clusters     = n_clusters,
        n_stage2_plots = len(stage2_plots),
        stage1_plots   = stage1_plots,
        stage2_plots   = stage2_plots,
        debug_point    = debug_point,
    )