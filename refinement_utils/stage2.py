
"""
stage2.py
=========
Stage 2 — Per-cluster refinement.

Takes the stage-1 corrected plots, identifies ALL disconnected BFS clusters,
optionally expands each cluster's neighbourhood with tile-fetched plots,
then runs the full refinement pipeline (intersection clearing → directional
stretch → fine translation) independently on EVERY cluster.

Key design decisions
--------------------
* Tile expansion is done ONCE for the whole stage-1 bounding box, not once
  per cluster.  This avoids running the expensive edge detector many times.
  The full set of tile-fetched plots is then merged and deduplicated with the
  stage-1 plots, and clusters are identified on the merged set.

* Each cluster gets its own localised road surface loaded from the OSM
  shapefile (with a configurable buffer).  Loading roads globally would be
  wasteful and would mix road segments from one cluster's refinement into
  another cluster's loss function.

* Cluster IDs are assigned by distance of cluster centroid to the target
  (lat, lon): cluster_0 is always the closest cluster to the target point.

Public API
----------
    run_stage2(lat, lon, result_gdf, original_geoms, stats,
               streets_shp, tile_csv, tile_folder, edge_detector_fn,
               params, verbose)
        → list of ClusterResult(namedtuple)

    write_stage2_geojson(cluster_results, stage1_result_gdf, point_meta, out_path)
        → GeoJSON file path

    save_stage2_plots(cluster_results, point_id, out_dir)
"""

import json
import os
import shutil
import socket
import tempfile
import traceback
import warnings
from collections import namedtuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
from shapely.geometry import Point, box as sbox
from shapely.ops import unary_union
from shapely.affinity import translate, scale
import contextily as cx
from pyproj import Transformer

from refinement_utils.config import (
    UTM_EPSG, WGS84_EPSG, WEB_MERCATOR_EPSG,
    ESRI_URL, STAGE2, TILE_OVERLAP_PERCENT, VIZ,
)
from refinement_utils.road_utils import build_road_surface, load_roads_for_bbox

warnings.filterwarnings("ignore")

_UTM = UTM_EPSG

# Returned by run_stage2 for each cluster
ClusterResult = namedtuple("ClusterResult", [
    "cluster_id",           # int, 0 = nearest to target
    "stage1_geoms",         # list of Shapely UTM geoms (before refinement)
    "refined_geoms",        # list of Shapely UTM geoms (after refinement)
    "stats",                # dict with loss_before, loss_after, rdx, rdy, etc.
    "roads_utm",            # GeoDataFrame UTM — roads used for this cluster
    "road_surface",         # Shapely geometry (UTM)
])


# ===========================================================================
# Tile helpers
# ===========================================================================

def _load_tile_index(tile_csv: str) -> dict:
    """Returns {(r, c): {name, north, south, east, west}}"""
    tiles = {}
    df = pd.read_csv(tile_csv)
    for _, row in df.iterrows():
        name = str(row["Tile_Name"])
        try:
            parts = name.replace(".png", "").split("_")
            r = int(parts[1][1:])
            c = int(parts[2][1:])
        except Exception:
            continue
        tiles[(r, c)] = {
            "name":  name,
            "north": float(row["TL_Lat"]),
            "south": float(row["BR_Lat"]),
            "east":  float(row["BR_Lon"]),
            "west":  float(row["TL_Lon"]),
        }
    return tiles


def _find_home_tile(lat, lon, tile_index):
    for (r, c), info in tile_index.items():
        if (info["south"] <= lat <= info["north"] and
                info["west"] <= lon <= info["east"]):
            return r, c
    return None, None


def _get_closest_four_tiles(lat, lon, tile_index):
    home_r = home_c = home_bounds = None
    for (r, c), bounds in tile_index.items():
        if (bounds["south"] <= lat <= bounds["north"] and
                bounds["west"] <= lon <= bounds["east"]):
            home_r, home_c, home_bounds = r, c, bounds
            break
    if home_r is None:
        return None

    mid_lat = (home_bounds["north"] + home_bounds["south"]) / 2
    mid_lon = (home_bounds["east"]  + home_bounds["west"])  / 2
    row_mod = -1 if lat > mid_lat else 1
    col_mod =  1 if lon > mid_lon else -1

    rows = sorted([home_r, home_r + row_mod])
    cols = sorted([home_c, home_c + col_mod])

    selected = []
    for r in rows:
        for c in cols:
            if (r, c) not in tile_index:
                return None
            selected.append((r, c))
    return selected


def _get_stitched_bounds(indices, tile_index):
    tl = tile_index[indices[0]]
    br = tile_index[indices[3]]
    return {
        "north": float(f"{tl['north']:.8f}"),
        "south": float(f"{br['south']:.8f}"),
        "east":  float(f"{br['east']:.8f}"),
        "west":  float(f"{tl['west']:.8f}"),
    }


def _stitch_four_tiles(indices, tile_index, tile_folder,
                       overlap_percent=TILE_OVERLAP_PERCENT):
    tile_names = [tile_index[idx]["name"] for idx in indices]
    imgs = []
    for name in tile_names:
        path = os.path.join(tile_folder, name)
        if not os.path.exists(path):
            return None, None
        imgs.append(Image.open(path).convert("RGB"))

    tile_w, tile_h = imgs[0].size
    overlap_x = int(tile_w * overlap_percent)
    overlap_y = int(tile_h * overlap_percent)
    stitched = Image.new("RGB", (tile_w * 2 - overlap_x, tile_h * 2 - overlap_y))
    for img, pos in zip(imgs, [
        (0,                  0),
        (tile_w - overlap_x, 0),
        (0,                  tile_h - overlap_y),
        (tile_w - overlap_x, tile_h - overlap_y),
    ]):
        stitched.paste(img, pos)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    stitched.save(tmp.name, quality=95)
    tmp.close()
    return tmp.name, _get_stitched_bounds(indices, tile_index)


def _apply_stage1_transform(geoms_utm, stats):
    """Re-apply the stage-1 (dx, dy, sx, sy) transform to a list of UTM geoms."""
    dx = stats["total_dx_m"]
    dy = stats["total_dy_m"]
    sx = stats["total_sx"]
    sy = stats["total_sy"]
    ax = stats["scale_anchor_x"]
    ay = stats["scale_anchor_y"]
    out = []
    for g in geoms_utm:
        g2 = scale(g, xfact=sx, yfact=sy, origin=(ax, ay))
        out.append(translate(g2, xoff=dx, yoff=dy))
    return out


def _deduplicate(geom_list, distance_m):
    """
    Remove duplicate geoms whose centroids are within distance_m of each other.
    Uses an STRtree for O(n log n) performance instead of O(n²).
    """
    if not geom_list:
        return []
    from shapely.strtree import STRtree
    cents = [g.centroid for g in geom_list]
    tree = STRtree(cents)
    used = set()
    keep = []
    for i, c in enumerate(cents):
        if i in used:
            continue
        keep.append(i)
        for j in tree.query(c.buffer(distance_m)):
            if j != i and j not in used and c.distance(cents[j]) < distance_m:
                used.add(j)
    return [geom_list[i] for i in keep]


def _dedup_by_overlap(geom_list, overlap_threshold=0.60):
    """
    Remove polygons that are substantially covered by a larger polygon.

    When the edge detector is run on overlapping 2x2 tile stitch windows
    (stride-1 enumeration), plots that fall in the shared overlap zone
    between two adjacent windows are detected redundantly.  This pass
    discards the smaller/duplicate detection by checking what fraction of
    each polygon's area is already covered by a previously-kept larger one.

    Sort order: largest first, so the authoritative (largest) detection is
    always kept and the redundant smaller one is discarded.

    Uses a lazily-rebuilt STRtree so the spatial index is only reconstructed
    when the kept set actually grows.

    Parameters
    ----------
    geom_list         : list of Shapely Polygon / MultiPolygon (UTM)
    overlap_threshold : fraction of smaller polygon covered by a larger one
                        that triggers removal (default 0.60)
    Returns
    -------
    list of Shapely geometries (subset of geom_list, original order preserved)
    """
    if not geom_list:
        return []
    from shapely.strtree import STRtree

    indexed = sorted(enumerate(geom_list), key=lambda x: x[1].area, reverse=True)
    kept_geoms        = []
    kept_orig_indices = []
    _tree      = None
    _tree_size = 0

    for orig_i, geom in indexed:
        if not geom.is_valid or geom.is_empty:
            continue
        is_dup = False
        if kept_geoms:
            if _tree is None or len(kept_geoms) != _tree_size:
                _tree      = STRtree(kept_geoms)
                _tree_size = len(kept_geoms)
            for ci in _tree.query(geom):
                try:
                    inter = geom.intersection(kept_geoms[ci]).area
                except Exception:
                    continue
                if geom.area > 0 and (inter / geom.area) > overlap_threshold:
                    is_dup = True
                    break
        if not is_dup:
            kept_geoms.append(geom)
            kept_orig_indices.append(orig_i)

    kept_set = set(kept_orig_indices)
    return [g for i, g in enumerate(geom_list) if i in kept_set]


def _filter_connecting_plots(extra_geoms, pipeline_geoms, connect_gap_m):
    """
    From extra_geoms (tile-fetched), keep only those that are within
    connect_gap_m of at least one plot in pipeline_geoms.

    These are the only plots that can extend an existing stage-1 cluster
    across a tile boundary.  Plots that are further away would form brand-new
    isolated clusters — which is not what tile expansion is for.

    Parameters
    ----------
    extra_geoms    : list of Shapely UTM geoms from adjacent tiles
    pipeline_geoms : list of Shapely UTM geoms from stage-1 output
    connect_gap_m  : maximum distance (metres) for a plot to be considered
                     "connecting" to an existing cluster

    Returns list of filtered extra_geoms.
    """
    if not extra_geoms or not pipeline_geoms:
        return []

    # Build a single union of all pipeline plots, buffered by connect_gap_m.
    # Any extra plot whose centroid falls inside this buffer is a neighbour
    # of an existing cluster and is therefore kept.
    pipeline_union_buffered = unary_union(pipeline_geoms).buffer(connect_gap_m)

    kept = []
    for g in extra_geoms:
        if g.centroid.within(pipeline_union_buffered):
            kept.append(g)
    return kept


def _run_one_group(group, tile_index, tile_folder, edge_detector_fn,
                   diag_dir, group_idx, verbose):
    """
    Stitch one 4-tile group, run the edge detector, return raw GeoJSON features.
    Cleans up the temp file.  Returns (list_of_features, stitch_path_saved).
    """
    indices = list(group)
    names   = [tile_index[i]["name"] for i in indices if i in tile_index]
    stitch_path, geo_bounds = _stitch_four_tiles(indices, tile_index, tile_folder)
    if stitch_path is None:
        if verbose:
            print(f"    [tile-expand] Skipping {names} — missing tiles")
        return [], None

    if diag_dir:
        diag_name = f"stitch_{group_idx:03d}_" + "_".join(
            n.replace(".png", "") for n in names
        ) + ".png"
        try:
            shutil.copy2(stitch_path, os.path.join(diag_dir, diag_name))
        except Exception:
            pass

    features = []
    try:
        if verbose:
            print(f"    [tile-expand] Edge detect: {names}")
        result = edge_detector_fn(geo_bounds, stitch_path)
        _, geojson = result if isinstance(result, tuple) else (None, result)
        if geojson and geojson.get("features"):
            features = geojson["features"]
            if verbose:
                print(f"      -> {len(features)} plots")
    except Exception as exc:
        if verbose:
            print(f"    [tile-expand] Edge detect failed for {names}: {exc}")
            traceback.print_exc()
    finally:
        try:
            os.unlink(stitch_path)
        except Exception:
            pass

    return features, names


def _coverage_bbox_utm(tl_r_min, tl_c_min, tl_r_max, tl_c_max,
                       tile_index, _wgs_to_utm):
    """
    Return the UTM bounding box (utm_w, utm_s, utm_e, utm_n) of all tiles
    covered by windows whose top-left corners span
    [tl_r_min..tl_r_max] x [tl_c_min..tl_c_max].

    Covered tiles are [tl_r_min .. tl_r_max+1] x [tl_c_min .. tl_c_max+1].
    """
    north = south = east = west = None
    for r in range(tl_r_min, tl_r_max + 2):
        for c in range(tl_c_min, tl_c_max + 2):
            if (r, c) not in tile_index:
                continue
            t = tile_index[(r, c)]
            north = t["north"] if north is None else max(north, t["north"])
            south = t["south"] if south is None else min(south, t["south"])
            east  = t["east"]  if east  is None else max(east,  t["east"])
            west  = t["west"]  if west  is None else min(west,  t["west"])
    if None in (north, south, east, west):
        return None
    utm_w, utm_s = _wgs_to_utm.transform(west,  south)
    utm_e, utm_n = _wgs_to_utm.transform(east,  north)
    return utm_w, utm_s, utm_e, utm_n


def _touched_edges(geoms_utm, utm_w, utm_s, utm_e, utm_n, inset_m):
    """
    Return a set of direction strings {'N','S','E','W'} for each coverage
    edge that has at least one plot within inset_m of it.

    A plot "touches" the North edge if its bounding box top (maxy) is within
    inset_m of utm_n, and similarly for the other three edges.

    Using per-plot bbox bounds (minx/miny/maxx/maxy) is intentional: it is
    fast (no full geometry ops) and conservative — it may trigger an expansion
    where none is strictly needed, but it will never miss a genuine edge touch.
    """
    touched = set()
    for g in geoms_utm:
        minx, miny, maxx, maxy = g.bounds
        if maxy >= utm_n - inset_m:
            touched.add("N")
        if miny <= utm_s + inset_m:
            touched.add("S")
        if maxx >= utm_e - inset_m:
            touched.add("E")
        if minx <= utm_w + inset_m:
            touched.add("W")
        if len(touched) == 4:   # all edges already triggered — short-circuit
            break
    return touched


def fetch_all_tile_plots(
    stage1_geoms_utm,
    tile_csv, tile_folder, edge_detector_fn, stats,
    center_lat, center_lon,
    expansion_rings=4,
    diag_dir=None,
    verbose=True,
):
    """
    Run the edge detector on 2x2 tile stitch windows using demand-driven,
    directional ring expansion.

    Algorithm
    ---------
    1. Start with the single 2x2 window containing (center_lat, center_lon).
    2. Run the edge detector; collect detected plots (UTM).
    3. Check each of the 4 coverage edges independently: does any plot's bbox
       come within boundary_inset_m of that edge?
    4. For each triggered edge, add one column/row of new windows on that side.
       Directions are processed in order N→S→E→W and the coverage bbox is
       updated in-place after each direction, so corner windows are naturally
       included whenever two adjacent edges both trigger (e.g. if N triggers
       and shifts tl_r_min, the subsequent E expansion uses the already-shifted
       row range and therefore covers the NE corner window automatically).
    5. Repeat steps 3-4 until no edge triggers or expansion_rings hard cap hit.

    Stride-1 guarantee
    ------------------
    Adjacent windows always share a full tile column (horizontal neighbours)
    or tile row (vertical neighbours).  New windows added on expansion share
    one tile with the existing coverage border, so no plot can be split across
    any seam.  Corner windows are covered by the in-place bbox update described
    in step 4.

    Parameters
    ----------
    stage1_geoms_utm : kept for API compatibility; not used for expansion.
    tile_csv, tile_folder, edge_detector_fn, stats : standard pipeline args.
    center_lat, center_lon : target coordinate — determines the initial window.
    expansion_rings  : hard cap on expansion iterations (default 4).
    diag_dir         : save stitched tile debug images here (None = skip).
    verbose          : bool.

    Config keys read from STAGE2
    ----------------------------
    boundary_inset_m         : metres inset from edge to trigger expansion (default 20).
    dedup_distance_m         : centroid dedup radius.
    tile_overlap_dedup_threshold : overlap fraction for polygon dedup (default 0.60).

    Returns list of Shapely UTM geoms (stage-1 transform applied), or None.
    """
    tile_index     = _load_tile_index(tile_csv)
    dedup_m        = STAGE2["dedup_distance_m"]
    inset_m        = STAGE2.get("boundary_inset_m", 20.0)
    overlap_thresh = STAGE2.get("tile_overlap_dedup_threshold", 0.60)

    _wgs_to_utm = Transformer.from_crs(
        f"EPSG:{WGS84_EPSG}", f"EPSG:{_UTM}", always_xy=True
    )

    if diag_dir:
        os.makedirs(diag_dir, exist_ok=True)

    # ── Step 1: initial 2x2 window around the target point ──────────────────
    initial_group = _get_closest_four_tiles(center_lat, center_lon, tile_index)
    if initial_group is None:
        if verbose:
            print("    [tile-expand] Target point not in any tile — skipping")
        return None

    # Coverage tracked as bbox of top-left corners of all run windows.
    # initial_group = [(tl_r, tl_c), (tl_r, tl_c+1), (tl_r+1, tl_c), (tl_r+1, tl_c+1)]
    tl_r_min = tl_r_max = initial_group[0][0]
    tl_c_min = tl_c_max = initial_group[0][1]

    run_groups    = set()    # windows already submitted to the edge detector
    all_features  = []
    group_counter = [0]      # mutable for use inside nested function

    def _submit(quad):
        """Run quad through the edge detector if not already done."""
        q = tuple(quad)
        if q in run_groups:
            return
        run_groups.add(q)
        feats, _ = _run_one_group(
            q, tile_index, tile_folder, edge_detector_fn,
            diag_dir, group_counter[0], verbose
        )
        group_counter[0] += 1
        all_features.extend(feats)

    def _add_windows(r_lo, r_hi, c_lo, c_hi):
        """Submit all valid stride-1 windows with TL in [r_lo..r_hi] x [c_lo..c_hi]."""
        for tl_r in range(r_lo, r_hi + 1):
            for tl_c in range(c_lo, c_hi + 1):
                quad = (
                    (tl_r,     tl_c),
                    (tl_r,     tl_c + 1),
                    (tl_r + 1, tl_c),
                    (tl_r + 1, tl_c + 1),
                )
                if all(t in tile_index for t in quad):
                    _submit(quad)

    # ── Step 2: run the initial window ──────────────────────────────────────
    _submit(tuple(initial_group))

    rings_used = 0

    # ── Steps 3-5: demand-driven directional expansion ───────────────────────
    for ring in range(1, expansion_rings + 1):

        if not all_features:
            break

        # Parse all collected plots into UTM geoms, then apply the stage-1
        # transform so the boundary check is done in corrected (road-aligned)
        # space.  This matters because stage-1 can shift plots by several
        # metres; checking in raw space would give a boundary distance that is
        # off by that shift amount and could cause missed or spurious expansions.
        gdf_check = gpd.GeoDataFrame.from_features(all_features)
        gdf_check = gdf_check.set_crs(epsg=WGS84_EPSG, allow_override=True)
        gdf_check = gdf_check[
            gdf_check.geometry.notna() &
            gdf_check.geometry.is_valid &
            ~gdf_check.geometry.is_empty
        ]
        if gdf_check.empty:
            break
        # Raw UTM → apply stage-1 transform → corrected positions
        raw_geoms_utm      = gdf_check.to_crs(_UTM).geometry.tolist()
        corrected_geoms_utm = _apply_stage1_transform(raw_geoms_utm, stats)

        # Coverage bbox in UTM (pre-transform tile bounds), then shift by the
        # stage-1 translation so it matches the corrected plot positions.
        # Scale is near-1 so shifting the bbox corners by (dx, dy) is a
        # sufficient approximation; we add the inset on top of this.
        bbox = _coverage_bbox_utm(
            tl_r_min, tl_c_min, tl_r_max, tl_c_max, tile_index, _wgs_to_utm
        )
        if bbox is None:
            break
        dx = stats["total_dx_m"]
        dy = stats["total_dy_m"]
        utm_w = bbox[0] + dx
        utm_s = bbox[1] + dy
        utm_e = bbox[2] + dx
        utm_n = bbox[3] + dy

        triggered = _touched_edges(corrected_geoms_utm, utm_w, utm_s, utm_e, utm_n, inset_m)

        if not triggered:
            if verbose:
                print(f"    [tile-expand] Ring {ring}: no edges touched "
                      f"— stopping expansion")
            break

        if verbose:
            print(f"    [tile-expand] Ring {ring}: edges touched={sorted(triggered)}")

        rings_used = ring

        # Process N→S→E→W.  Update tl_r/c_min/max in-place after each
        # direction so that subsequent directions automatically cover corners.
        #
        # Example — N and E both triggered:
        #   N processed first: tl_r_min decrements by 1, new top row of windows
        #   added spanning [new_tl_r_min, new_tl_r_min] x [tl_c_min..tl_c_max].
        #   E processed next: tl_c_max increments by 1, new right column added
        #   spanning [tl_r_min..tl_r_max] x [new_tl_c_max, new_tl_c_max].
        #   Because tl_r_min was already decremented, the NE corner window
        #   (new_tl_r_min, new_tl_c_max) is included in the E column — no
        #   special corner logic needed.

        if "N" in triggered:
            new_tl_r_min = tl_r_min - 1
            _add_windows(new_tl_r_min, new_tl_r_min, tl_c_min, tl_c_max)
            tl_r_min = new_tl_r_min

        if "S" in triggered:
            new_tl_r_max = tl_r_max + 1
            _add_windows(new_tl_r_max, new_tl_r_max, tl_c_min, tl_c_max)
            tl_r_max = new_tl_r_max

        if "E" in triggered:
            new_tl_c_max = tl_c_max + 1
            # Row range uses already-updated tl_r_min/max → corners covered
            _add_windows(tl_r_min, tl_r_max, new_tl_c_max, new_tl_c_max)
            tl_c_max = new_tl_c_max

        if "W" in triggered:
            new_tl_c_min = tl_c_min - 1
            # Row range uses already-updated tl_r_min/max → corners covered
            _add_windows(tl_r_min, tl_r_max, new_tl_c_min, new_tl_c_min)
            tl_c_min = new_tl_c_min

    if verbose:
        grid_r = tl_r_max - tl_r_min + 2
        grid_c = tl_c_max - tl_c_min + 2
        print(f"    [tile-expand] {len(run_groups)} windows run "
              f"(rings_used={rings_used}/{expansion_rings}, "
              f"final grid={grid_r}x{grid_c} tiles)")

    if not all_features:
        if verbose:
            print("    [tile-expand] No plots found")
        return None

    gdf = gpd.GeoDataFrame.from_features(all_features)
    gdf = gdf.set_crs(epsg=WGS84_EPSG, allow_override=True)
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    if gdf.empty:
        return None

    gdf_utm   = gdf.to_crs(_UTM).reset_index(drop=True)
    raw_geoms = gdf_utm.geometry.tolist()
    deduped   = _deduplicate(raw_geoms, dedup_m)
    if verbose:
        print(f"    [tile-expand] {len(raw_geoms)} raw -> {len(deduped)} after centroid-dedup")

    deduped = _dedup_by_overlap(deduped, overlap_threshold=overlap_thresh)
    if verbose:
        print(f"    [tile-expand] {len(deduped)} after overlap-dedup "
              f"(threshold={overlap_thresh:.0%})")

    return _apply_stage1_transform(deduped, stats)


# ===========================================================================
# BFS cluster detection
# ===========================================================================

def find_all_clusters(geoms_utm, gap_m=4.0):
    """
    BFS-partition ALL geoms into disconnected clusters.

    Uses an STRtree spatial index so each BFS expansion is sub-linear
    rather than scanning all remaining candidates.

    Returns list of lists of indices, sorted by cluster size descending.
    """
    from shapely.strtree import STRtree
    if not geoms_utm:
        return []
    buffered  = [g.buffer(gap_m / 2) for g in geoms_utm]
    tree      = STRtree(buffered)
    unvisited = set(range(len(geoms_utm)))
    clusters  = []

    while unvisited:
        seed = next(iter(unvisited))
        unvisited.discard(seed)
        visited = {seed}
        queue   = [seed]
        while queue:
            cur = queue.pop(0)
            for j in tree.query(buffered[cur]):
                if j in unvisited and buffered[cur].intersects(buffered[j]):
                    visited.add(j)
                    unvisited.discard(j)
                    queue.append(j)
        clusters.append(sorted(visited))

    clusters.sort(key=lambda c: len(c), reverse=True)
    return clusters


# ===========================================================================
# Per-cluster refinement helpers
# ===========================================================================

def _refinement_loss(geom_list, road_surface):
    dists = []
    for geom in geom_list:
        if geom.buffer(-0.01).intersects(road_surface):
            dists.append(-1.0)
        else:
            dists.append(geom.distance(road_surface))
    if any(d < 0 for d in dists):
        return 1e9, dists
    return float(sum(dists)), dists


def _clear_road_intersections(geom_list, road_surface, cfg, verbose=True):
    """
    Translate cluster as a rigid body to clear road overlaps.
    Returns (cleared_geoms, total_dx, total_dy, was_needed).
    """
    def overlap_area(glist):
        total = 0.0
        for g in glist:
            try:
                inter = g.intersection(road_surface)
                if not inter.is_empty:
                    total += inter.area
            except Exception:
                pass
        return total

    current = list(geom_list)
    init_area = overlap_area(current)
    if init_area == 0.0:
        return current, 0.0, 0.0, False

    n_i = sum(1 for g in geom_list if g.buffer(-0.01).intersects(road_surface))
    if verbose:
        print(f"      [clear] {n_i}/{len(geom_list)} plots intersect roads "
              f"({init_area:.2f}m2) — clearing...")

    tdx = tdy = 0.0
    step = cfg["clear_initial_step"]
    itr = 0
    best_area  = init_area
    best_geoms = list(current)

    while step >= cfg["clear_min_step"]:
        if overlap_area(current) == 0.0:
            break
        if abs(tdx) + abs(tdy) > cfg["clear_max_move"]:
            if verbose:
                print(f"      [clear] WARNING: hit {cfg['clear_max_move']}m cap")
            break

        cur_area = overlap_area(current)
        best_local = cur_area; bdx = bdy = 0.0; bdir = None
        for dx, dy, nm in [(step,0,"+E"),(-step,0,"-E"),(0,step,"+N"),(0,-step,"-N")]:
            ta = overlap_area([translate(g, xoff=dx, yoff=dy) for g in current])
            if ta < best_local:
                best_local, bdx, bdy, bdir = ta, dx, dy, nm

        if bdir:
            current = [translate(g, xoff=bdx, yoff=bdy) for g in current]
            tdx += bdx; tdy += bdy; itr += 1
            remaining = overlap_area(current)
            if verbose:
                print(f"      [clear] step={step:.3f}m {bdir} "
                      f"area={remaining:.3f}m2 moved=({tdx:.2f},{tdy:.2f})m")
            if remaining < best_area:
                best_area = remaining; best_geoms = list(current)
            if remaining == 0.0:
                break
        else:
            step /= 2.0

    final = overlap_area(current)
    if final > 0.0:
        if verbose:
            print(f"      [clear] Could not fully clear. Best={best_area:.3f}m2")
        current = best_geoms

    if verbose:
        print(f"      [clear] Done in {itr} steps. "
              f"Moved dx={tdx:.2f}m dy={tdy:.2f}m")
    return current, tdx, tdy, True


def _directional_gap(geom_list, direction, road_surface):
    cu = unary_union(geom_list)
    if cu.intersects(road_surface):
        return 0.0
    b = cu.bounds
    pd_ = 200.0
    if   direction == "N": probe = sbox(b[0], b[3], b[2], b[3] + pd_)
    elif direction == "S": probe = sbox(b[0], b[1] - pd_, b[2], b[1])
    elif direction == "E": probe = sbox(b[2], b[1], b[2] + pd_, b[3])
    else:                  probe = sbox(b[0] - pd_, b[1], b[0], b[3])
    road_in_dir = road_surface.intersection(probe)
    if road_in_dir.is_empty:
        return None
    return float(cu.distance(road_in_dir))


def _refine_step_search(geom_list, road_surface, cfg, verbose=True):
    current = list(geom_list)
    tdx = tdy = 0.0
    step = cfg["refine_step_m"]
    itr  = 0
    while step >= cfg["min_refine_step"]:
        cur_loss, _ = _refinement_loss(current, road_surface)
        best = cur_loss; bdx = bdy = 0.0; bdir = None
        for dx, dy, nm in [(step,0,"+E"),(-step,0,"-E"),(0,step,"+N"),(0,-step,"-N")]:
            tl, _ = _refinement_loss(
                [translate(g, xoff=dx, yoff=dy) for g in current], road_surface
            )
            if tl < best:
                best, bdx, bdy, bdir = tl, dx, dy, nm
        if bdir:
            current = [translate(g, xoff=bdx, yoff=bdy) for g in current]
            tdx += bdx; tdy += bdy; itr += 1
            if verbose:
                l, _ = _refinement_loss(current, road_surface)
                print(f"      [trans] step={step:.3f}m {bdir} loss={l:.2f}")
        else:
            step /= 2.0
    return current, tdx, tdy, itr


def _refine_stretch_search(geom_list, road_surface, cfg, verbose=True):
    current = list(geom_list)
    itr = 0

    bounded = {}; gaps = {}
    for d in ("N", "S", "E", "W"):
        gap = _directional_gap(current, d, road_surface)
        bounded[d] = gap is not None; gaps[d] = gap
        if verbose:
            if gap is None: print(f"      [stretch] {d} = open (no road)")
            else:           print(f"      [stretch] {d} gap = {gap:.2f}m")

    dirs_with_roads = [d for d in ("N", "S", "E", "W") if bounded[d]]
    if not dirs_with_roads:
        if verbose: print("      [stretch] No bounded directions — nothing to stretch")
        return current, 1.0, 1.0, 0

    dir_step     = {d: cfg["refine_scale_step"] for d in dirs_with_roads}
    min_sf       = cfg["min_scale_factor"]
    max_sf       = cfg["max_scale_factor"]
    any_progress = True

    while any_progress:
        any_progress = False
        for direction in dirs_with_roads:
            ss = dir_step[direction]
            if ss < cfg["min_refine_scale"]:
                continue

            b = unary_union(current).bounds
            if   direction == "N": ax_, ay_ = (b[0]+b[2])/2, b[1]; sx, sy = 1.0, 1.0+ss
            elif direction == "S": ax_, ay_ = (b[0]+b[2])/2, b[3]; sx, sy = 1.0, 1.0+ss
            elif direction == "E": ax_, ay_ = b[0], (b[1]+b[3])/2; sx, sy = 1.0+ss, 1.0
            else:                  ax_, ay_ = b[2], (b[1]+b[3])/2; sx, sy = 1.0+ss, 1.0

            # Estimate cumulative scale vs original cluster bbox
            b0 = unary_union(geom_list).bounds
            cum_sx = max(b[2]-b[0], 1e-6) / max(b0[2]-b0[0], 1e-6)
            cum_sy = max(b[3]-b[1], 1e-6) / max(b0[3]-b0[1], 1e-6)

            if direction in ("E", "W") and not (min_sf <= cum_sx*(1+ss) <= max_sf):
                dir_step[direction] /= 2.0; continue
            if direction in ("N", "S") and not (min_sf <= cum_sy*(1+ss) <= max_sf):
                dir_step[direction] /= 2.0; continue

            trial = [scale(g, xfact=sx, yfact=sy, origin=(ax_, ay_)) for g in current]
            if not all(g.is_valid and not g.is_empty for g in trial):
                dir_step[direction] /= 2.0; continue
            if any(g.buffer(-0.01).intersects(road_surface) for g in trial):
                dir_step[direction] /= 2.0; continue

            loss_before, _ = _refinement_loss(current, road_surface)
            loss_after,  _ = _refinement_loss(trial,   road_surface)
            if loss_after < loss_before:
                current = trial; itr += 1; any_progress = True
                new_gap = _directional_gap(current, direction, road_surface)
                if verbose:
                    print(f"      [stretch-{direction}] ss={ss:.5f} "
                          f"gap {gaps[direction]:.2f}m -> {new_gap:.2f}m "
                          f"loss={loss_after:.2f}m")
                gaps[direction] = new_gap if new_gap is not None else 0.0
                if new_gap is not None and new_gap < 0.05:
                    dir_step[direction] = 0.0
            else:
                dir_step[direction] /= 2.0

    if verbose:
        lf, _ = _refinement_loss(current, road_surface)
        print(f"      [stretch] Done: {itr} steps, loss={lf:.2f}m")
    return current, 1.0, 1.0, itr


def _refine_one_cluster(cluster_geoms, stage1_geoms, road_surface, roads_utm,
                        streets_shp, cfg, verbose=True):
    """
    Run the full refinement pipeline on one cluster.

    Returns (refined_geoms, stats_dict)
    """
    # Stage-1 snapshot for loss baseline
    stage1_snapshot = list(stage1_geoms)

    # Pre-processing: clear intersections
    n_intersecting = sum(
        1 for g in cluster_geoms if g.buffer(-0.01).intersects(road_surface)
    )
    pre_clear_dx = pre_clear_dy = 0.0

    if n_intersecting > 0:
        if verbose:
            print(f"      Pre-processing: clearing {n_intersecting} intersection(s)...")
        cluster_geoms, pre_clear_dx, pre_clear_dy, _ = _clear_road_intersections(
            cluster_geoms, road_surface, cfg, verbose=verbose
        )
        still = sum(1 for g in cluster_geoms if g.buffer(-0.01).intersects(road_surface))
        if still > 0 and verbose:
            print(f"      [clear] {still} plot(s) still intersecting — best-effort")
    else:
        if verbose:
            print("      No road intersections — skipping pre-processing")

    loss_init, _ = _refinement_loss(cluster_geoms, road_surface)
    if verbose:
        print(f"      Baseline loss (post-clear): {loss_init:.2f}m")

    # Directional stretch
    if verbose:
        print("      Directional stretch...")
    pre_stretch = list(cluster_geoms)
    refined, rsx, rsy, ri2 = _refine_stretch_search(
        cluster_geoms, road_surface, cfg, verbose=verbose
    )
    l_post_stretch, _ = _refinement_loss(refined,     road_surface)
    l_pre_stretch,  _ = _refinement_loss(pre_stretch, road_surface)
    if l_post_stretch >= l_pre_stretch:
        if verbose: print("      Stretch did not improve — reverting")
        refined = pre_stretch; rsx = rsy = 1.0

    # Post-stretch fine translation
    if verbose:
        print("      Post-stretch fine translation...")
    refined, rdx, rdy, ri3 = _refine_step_search(
        refined, road_surface, cfg, verbose=verbose
    )

    loss_final, _ = _refinement_loss(refined, road_surface)
    if loss_final > loss_init:
        if verbose:
            print("      WARNING: refinement made things worse — reverting to cleared baseline")
        refined = cluster_geoms; rdx = rdy = 0.0; rsx = rsy = 1.0
        loss_final, _ = _refinement_loss(refined, road_surface)

    # Fold clearing translation into rdx/rdy
    rdx += pre_clear_dx
    rdy += pre_clear_dy

    loss_stage1, _ = _refinement_loss(stage1_snapshot, road_surface)

    stats = {
        "rdx_m":            rdx,
        "rdy_m":            rdy,
        "rsx":              rsx,
        "rsy":              rsy,
        "loss_before":      loss_stage1,
        "loss_after":       loss_final,
        "cluster_size":     len(cluster_geoms),
        "translate_iters":  ri3,
        "stretch_iters":    ri2,
        "n_plots_refined":  len(refined),
    }
    return refined, stats


# ===========================================================================
# Main stage-2 entry point
# ===========================================================================

def run_stage2(
    lat, lon,
    result_gdf,         # stage-1 output: GeoDataFrame WGS84
    original_geoms,     # stage-1 input:  list of UTM geoms (pre-stage-1)
    stats,              # stage-1 stats dict (contains scale_anchor etc.)
    streets_shp,        # absolute path to OSM shapefile
    tile_csv,           # path to tile index CSV
    tile_folder,        # folder containing tile PNGs
    edge_detector_fn,   # callable (geo_bounds, img_path) → (df, geojson)
    params=None,        # override STAGE2 config keys
    diag_dir=None,      # directory to save stitched tile debug images
    verbose=True,
):
    """
    Identify all BFS clusters in the merged (stage-1 + tile-expanded) plot set
    and refine each one independently.

    Parameters
    ----------
    lat, lon            : target coordinate (used to rank clusters by proximity
                          and as the primary tile expansion seed)
    result_gdf          : GeoDataFrame WGS84 — stage-1 corrected plots
    original_geoms      : list of UTM geoms — before stage-1 (for baseline loss)
    stats               : dict from run_stage1 — must contain scale_anchor_x/y
    streets_shp         : str — absolute path to OSM roads shapefile
    tile_csv, tile_folder, edge_detector_fn : for tile-based plot expansion
    params              : dict — override any STAGE2 config key
    diag_dir            : str | None — save stitched tile images here
    verbose             : bool

    Returns
    -------
    list of ClusterResult (ordered by cluster_id, i.e. cluster_0 is closest
    to the target point)
    """
    cfg = {**STAGE2, **(params or {})}

    # ── Step 1: Build merged plot set ─────────────────────────────────────
    if verbose:
        print("\n  [S2] Step 1: Building merged plot set...")

    pipeline_utm = result_gdf.to_crs(_UTM).geometry.tolist()

    extra_utm = None
    if tile_csv and tile_folder and edge_detector_fn:
        try:
            extra_utm = fetch_all_tile_plots(
                stage1_geoms_utm=pipeline_utm,
                tile_csv=tile_csv,
                tile_folder=tile_folder,
                edge_detector_fn=edge_detector_fn,
                stats=stats,
                center_lat=lat,
                center_lon=lon,
                expansion_rings=cfg["expansion_rings"],
                diag_dir=diag_dir,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                print(f"    [S2] Tile expansion failed: {exc} — using pipeline plots only")
                traceback.print_exc()

    all_utm = list(pipeline_utm)
    if extra_utm:
        # Only keep tile-fetched plots that spatially connect to an existing
        # stage-1 cluster (i.e. are within connect_gap_m of a pipeline plot).
        # Plots deeper in the adjacent tile that don't border any current
        # cluster would otherwise create spurious new clusters.
        connect_gap = cfg.get("tile_connect_gap_m", cfg["gap_threshold_m"])
        connecting = _filter_connecting_plots(extra_utm, pipeline_utm, connect_gap)
        if verbose:
            print(f"    [S2] Tile-fetched plots: {len(extra_utm)} total, "
                  f"{len(connecting)} connect to existing clusters "
                  f"(connect_gap={connect_gap}m), "
                  f"{len(extra_utm) - len(connecting)} discarded")
        all_utm.extend(connecting)

    # Deduplicate merged set — centroid proximity
    all_utm = _deduplicate(all_utm, cfg["dedup_distance_m"])

    # Remove small plots that are substantially covered by a larger plot.
    # This eliminates text-cutout fragments and other sub-plot artifacts that
    # survive from the original edge detector run into the pipeline_utm set.
    # _dedup_by_overlap is already applied inside fetch_all_tile_plots on the
    # tile-fetched plots, but pipeline_utm (the stage-1 output) has never had
    # this filter applied — so we run it here on the fully merged set.
    overlap_thresh = cfg.get("tile_overlap_dedup_threshold", 0.60)
    n_before = len(all_utm)
    all_utm = _dedup_by_overlap(all_utm, overlap_threshold=overlap_thresh)
    if verbose:
        print(f"    [S2] Total plots after merge + dedup: {len(all_utm)} "
              f"(overlap-dedup removed {n_before - len(all_utm)})")

    # ── Step 2: Find all clusters ─────────────────────────────────────────
    if verbose:
        print(f"  [S2] Step 2: Finding all BFS clusters "
              f"(gap={cfg['gap_threshold_m']}m)...")

    raw_clusters = find_all_clusters(all_utm, gap_m=cfg["gap_threshold_m"])
    if verbose:
        print(f"    [S2] Found {len(raw_clusters)} clusters: "
              + ", ".join(f"C{i}({len(c)} plots)" for i, c in enumerate(raw_clusters)))

    # ── Step 3: Rank clusters by distance to target ───────────────────────
    _wgs_to_utm = Transformer.from_crs(
        f"EPSG:{WGS84_EPSG}", f"EPSG:{_UTM}", always_xy=True
    )
    tx, ty = _wgs_to_utm.transform(lon, lat)
    target_pt = Point(tx, ty)

    def cluster_dist(indices):
        cents = [all_utm[i].centroid for i in indices]
        cu    = unary_union([all_utm[i] for i in indices]).centroid
        return cu.distance(target_pt)

    ranked_clusters = sorted(raw_clusters, key=cluster_dist)

    # ── Step 4: Refine each cluster ───────────────────────────────────────
    results = []
    _utm_to_wgs = Transformer.from_crs(f"EPSG:{_UTM}", f"EPSG:{WGS84_EPSG}", always_xy=True)

    for cluster_id, indices in enumerate(ranked_clusters):
        cluster_geoms = [all_utm[i] for i in indices]
        n_plots       = len(cluster_geoms)
        centroid      = unary_union(cluster_geoms).centroid
        centroid_lon, centroid_lat = _utm_to_wgs.transform(centroid.x, centroid.y)
        dist_to_target = centroid.distance(target_pt)

        if verbose:
            print(f"\n  [S2] Cluster {cluster_id}: {n_plots} plots, "
                  f"centroid=({centroid_lat:.5f},{centroid_lon:.5f}), "
                  f"dist_to_target={dist_to_target:.0f}m")

        # Load roads local to this cluster
        try:
            cb = unary_union(cluster_geoms).bounds
            wgs_w, wgs_s = _utm_to_wgs.transform(
                cb[0] - cfg["search_buffer_m"], cb[1] - cfg["search_buffer_m"]
            )
            wgs_e, wgs_n = _utm_to_wgs.transform(
                cb[2] + cfg["search_buffer_m"], cb[3] + cfg["search_buffer_m"]
            )
            roads_utm, road_surface = load_roads_for_bbox(
                streets_shp, (wgs_w, wgs_s, wgs_e, wgs_n), _UTM
            )
            if verbose:
                l0, _ = _refinement_loss(cluster_geoms, road_surface)
                print(f"    Roads: {len(roads_utm)} segments, "
                      f"initial loss: {l0:.2f}m "
                      f"({'overlaps' if l0 >= 1e9 else 'clear'})")
        except Exception as exc:
            if verbose:
                print(f"    [S2] Road loading failed for cluster {cluster_id}: {exc}")
            results.append(ClusterResult(
                cluster_id=cluster_id,
                stage1_geoms=cluster_geoms,
                refined_geoms=cluster_geoms,   # unchanged
                stats={"error": str(exc), "cluster_size": n_plots},
                roads_utm=None,
                road_surface=None,
            ))
            continue

        # Run refinement
        try:
            refined_geoms, refine_stats = _refine_one_cluster(
                cluster_geoms=list(cluster_geoms),
                stage1_geoms=cluster_geoms,
                road_surface=road_surface,
                roads_utm=roads_utm,
                streets_shp=streets_shp,
                cfg=cfg,
                verbose=verbose,
            )
            refine_stats["cluster_id"]       = cluster_id
            refine_stats["centroid_lat"]     = centroid_lat
            refine_stats["centroid_lon"]     = centroid_lon
            refine_stats["dist_to_target_m"] = dist_to_target

            if verbose:
                print(f"    Cluster {cluster_id} done: "
                      f"loss {refine_stats['loss_before']:.2f}m "
                      f"-> {refine_stats['loss_after']:.2f}m "
                      f"rdx={refine_stats['rdx_m']:.2f}m "
                      f"rdy={refine_stats['rdy_m']:.2f}m")

            results.append(ClusterResult(
                cluster_id=cluster_id,
                stage1_geoms=cluster_geoms,
                refined_geoms=refined_geoms,
                stats=refine_stats,
                roads_utm=roads_utm,
                road_surface=road_surface,
            ))
        except Exception as exc:
            if verbose:
                print(f"    [S2] Refinement failed for cluster {cluster_id}: {exc}")
                traceback.print_exc()
            results.append(ClusterResult(
                cluster_id=cluster_id,
                stage1_geoms=cluster_geoms,
                refined_geoms=cluster_geoms,
                stats={"error": str(exc), "cluster_size": n_plots},
                roads_utm=roads_utm,
                road_surface=road_surface,
            ))

    if verbose:
        print(f"\n  [S2] All {len(results)} clusters processed.")

    return results


# ===========================================================================
# GeoJSON output
# ===========================================================================

def write_stage2_geojson(cluster_results, point_meta, out_path):
    """
    Write a single GeoJSON file with ALL plots from ALL clusters.

    Each feature carries:
        point_id, region_name, tile_names, target_lat, target_lon,
        stage         : "stage2_before" | "stage2_after"
        cluster_id    : int
        plot_index    : int (within cluster)
        loss_before   : float (cluster-level)
        loss_after    : float (cluster-level)
        rdx_m, rdy_m  : float (cluster-level translation applied)

    Parameters
    ----------
    cluster_results : list of ClusterResult
    point_meta      : dict with point_id, region_name, tile_names,
                      target_lat, target_lon
    out_path        : str
    """
    features = []
    _utm_to_wgs = Transformer.from_crs(
        f"EPSG:{_UTM}", f"EPSG:{WGS84_EPSG}", always_xy=True
    )

    for cr in cluster_results:
        cid   = cr.cluster_id
        stats = cr.stats
        base_props = {
            "point_id":    point_meta["point_id"],
            "region_name": point_meta.get("region_name", ""),
            "target_lat":  point_meta["target_lat"],
            "target_lon":  point_meta["target_lon"],
            "cluster_id":  cid,
            "loss_before": stats.get("loss_before"),
            "loss_after":  stats.get("loss_after"),
            "rdx_m":       stats.get("rdx_m"),
            "rdy_m":       stats.get("rdy_m"),
        }

        # Before (stage-1 cluster)
        stage1_wgs = gpd.GeoDataFrame(
            geometry=cr.stage1_geoms, crs=_UTM
        ).to_crs(WGS84_EPSG)
        for i, geom in enumerate(stage1_wgs.geometry):
            if geom is None or geom.is_empty:
                continue
            features.append({
                "type": "Feature",
                "geometry": geom.__geo_interface__,
                "properties": {
                    **base_props,
                    "stage":      "stage2_before",
                    "plot_index": i,
                },
            })

        # After (refined cluster)
        refined_wgs = gpd.GeoDataFrame(
            geometry=cr.refined_geoms, crs=_UTM
        ).to_crs(WGS84_EPSG)
        for i, geom in enumerate(refined_wgs.geometry):
            if geom is None or geom.is_empty:
                continue
            features.append({
                "type": "Feature",
                "geometry": geom.__geo_interface__,
                "properties": {
                    **base_props,
                    "stage":      "stage2_after",
                    "plot_index": i,
                },
            })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
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


def save_stage2_plots(cluster_results, point_id, out_dir, verbose=True):
    """
    Save one satellite comparison plot per cluster into out_dir/stage2_clusters/.
        stage2_cluster_<id>.png
    Also save a combined overview plot showing all clusters at once:
        stage2_overview.png
    """
    cluster_dir = os.path.join(out_dir, "stage2_clusters")
    os.makedirs(cluster_dir, exist_ok=True)

    _utm_to_3857 = Transformer.from_crs(f"EPSG:{_UTM}", f"EPSG:{WEB_MERCATOR_EPSG}", always_xy=True)
    _utm_to_wgs  = Transformer.from_crs(f"EPSG:{_UTM}", f"EPSG:{WGS84_EPSG}", always_xy=True)

    for cr in cluster_results:
        if cr.roads_utm is None:
            continue   # cluster failed road loading — skip plot

        _save_cluster_plot(cr, _utm_to_3857, _utm_to_wgs,
                           os.path.join(cluster_dir,
                                        f"stage2_cluster_{cr.cluster_id:03d}.png"),
                           verbose=verbose)

    # Overview: all clusters before/after on one canvas
    _save_overview_plot(cluster_results, _utm_to_3857, _utm_to_wgs,
                        os.path.join(out_dir, "stage2_overview.png"),
                        point_id=point_id, verbose=verbose)


def _save_cluster_plot(cr, _utm_to_3857, _utm_to_wgs, out_path, verbose=True):
    before_geoms = cr.stage1_geoms
    after_geoms  = cr.refined_geoms
    roads_utm    = cr.roads_utm
    road_surface = cr.road_surface
    stats        = cr.stats

    b_gdf = gpd.GeoDataFrame(geometry=before_geoms, crs=_UTM)
    b     = b_gdf.total_bounds
    pad   = max(b[2]-b[0], b[3]-b[1]) * VIZ["bbox_pad_fraction"]
    minx, miny, maxx, maxy = b[0]-pad, b[1]-pad, b[2]+pad, b[3]+pad

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

    def draw(ax, geoms, fc, ec):
        for g in geoms:
            for p in _utm_geom_to_wgs84_patch(g, facecolor=fc, edgecolor=ec,
                                               linewidth=0.9, zorder=3):
                ax.add_patch(p)

    fig, axes = plt.subplots(1, 2, figsize=VIZ["stage2_figsize"])
    for ax in axes:
        if img is not None:
            ax.imshow(img, extent=(ew, ee, es, en),
                      aspect="auto", origin="upper",
                      interpolation="lanczos", zorder=0)
        road_wgs.plot(ax=ax, color="red", alpha=0.30, zorder=1)
        roads_utm.to_crs(WGS84_EPSG).plot(ax=ax, color="red",
                                          linewidth=0.6, alpha=0.55, zorder=2)
        ax.set_xlim(wgs_w, wgs_e); ax.set_ylim(wgs_s, wgs_n)
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)

    draw(axes[0], before_geoms,
         fc=(0.27, 0.51, 0.71, 0.35), ec=(0.08, 0.20, 0.45, 1.0))
    axes[0].set_title(
        f"Cluster {cr.cluster_id} — BEFORE refinement\n"
        f"Total dist to roads: {stats.get('loss_before', '?'):.1f}m  |  "
        f"{stats.get('cluster_size', '?')} plots",
        fontsize=11
    )
    draw(axes[1], after_geoms,
         fc=(0.0, 0.78, 0.2, 0.35), ec=(0.0, 0.45, 0.1, 1.0))
    axes[1].set_title(
        f"Cluster {cr.cluster_id} — AFTER refinement  "
        f"rdx={stats.get('rdx_m', 0):.1f}m  rdy={stats.get('rdy_m', 0):.1f}m\n"
        f"Total dist to roads: {stats.get('loss_after', '?'):.1f}m",
        fontsize=11
    )
    axes[1].legend(handles=[
        mpatches.Patch(facecolor=(0.27,0.51,0.71,0.4),
                       edgecolor=(0.08,0.20,0.45), label="Before refinement"),
        mpatches.Patch(facecolor=(0.0,0.78,0.2,0.4),
                       edgecolor=(0.0,0.45,0.1),   label="After refinement"),
        mpatches.Patch(facecolor="red", alpha=0.35, label="Road surface"),
    ], loc="lower right", fontsize=9, framealpha=0.85)
    fig.text(0.01, 0.005, "Satellite © Esri, Maxar, Earthstar Geographics",
             fontsize=6, color="grey")
    plt.tight_layout()
    plt.savefig(out_path, dpi=VIZ["plot_dpi"], bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    [S2] Cluster {cr.cluster_id} plot -> {out_path}")


def _save_overview_plot(cluster_results, _utm_to_3857, _utm_to_wgs,
                        out_path, point_id="", verbose=True):
    """
    All clusters on one satellite canvas — stage-1 on left, refined on right.
    """
    valid = [cr for cr in cluster_results if cr.roads_utm is not None]
    if not valid:
        return

    all_before = [g for cr in valid for g in cr.stage1_geoms]
    all_after  = [g for cr in valid for g in cr.refined_geoms]

    b_gdf = gpd.GeoDataFrame(geometry=all_before, crs=_UTM)
    b     = b_gdf.total_bounds
    pad   = max(b[2]-b[0], b[3]-b[1]) * 0.08
    minx, miny, maxx, maxy = b[0]-pad, b[1]-pad, b[2]+pad, b[3]+pad

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

    # Build a combined road surface for the overview
    all_roads = pd.concat([cr.roads_utm for cr in valid], ignore_index=True)
    all_road_surface = build_road_surface(all_roads)
    road_wgs = gpd.GeoDataFrame(geometry=[all_road_surface], crs=_UTM).to_crs(WGS84_EPSG)

    # Assign a distinct colour per cluster
    colours = plt.cm.tab10.colors
    n_total_before = sum(len(cr.stage1_geoms)  for cr in valid)
    n_total_after  = sum(len(cr.refined_geoms) for cr in valid)

    fig, axes = plt.subplots(1, 2, figsize=(26, 13))
    for ax in axes:
        if img is not None:
            ax.imshow(img, extent=(ew, ee, es, en),
                      aspect="auto", origin="upper",
                      interpolation="lanczos", zorder=0)
        road_wgs.plot(ax=ax, color="red", alpha=0.25, zorder=1)
        all_roads.to_crs(WGS84_EPSG).plot(ax=ax, color="red",
                                           linewidth=0.5, alpha=0.5, zorder=2)
        ax.set_xlim(wgs_w, wgs_e); ax.set_ylim(wgs_s, wgs_n)
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)

    for cr in valid:
        col = colours[cr.cluster_id % len(colours)]
        fc  = (*col[:3], 0.35)
        ec  = (*col[:3], 1.0)
        for g in cr.stage1_geoms:
            for p in _utm_geom_to_wgs84_patch(g, facecolor=fc, edgecolor=ec,
                                               linewidth=0.8, zorder=3):
                axes[0].add_patch(p)
        for g in cr.refined_geoms:
            for p in _utm_geom_to_wgs84_patch(g, facecolor=fc, edgecolor=ec,
                                               linewidth=0.8, zorder=3):
                axes[1].add_patch(p)

    axes[0].set_title(
        f"{point_id} — Stage 2 BEFORE (all clusters)\n"
        f"{len(valid)} clusters, {n_total_before} plots",
        fontsize=12
    )
    axes[1].set_title(
        f"{point_id} — Stage 2 AFTER (all clusters)\n"
        f"{n_total_after} plots refined",
        fontsize=12
    )
    legend_patches = [
        mpatches.Patch(color=colours[cr.cluster_id % len(colours)],
                       label=f"Cluster {cr.cluster_id} ({len(cr.refined_geoms)} plots)")
        for cr in valid
    ]
    legend_patches.append(mpatches.Patch(facecolor="red", alpha=0.35, label="Roads"))
    axes[1].legend(handles=legend_patches, loc="lower right",
                   fontsize=8, framealpha=0.85)
    fig.text(0.01, 0.005, "Satellite © Esri, Maxar, Earthstar Geographics",
             fontsize=6, color="grey")
    plt.tight_layout()
    plt.savefig(out_path, dpi=VIZ["plot_dpi"], bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    [S2] Overview plot -> {out_path}")