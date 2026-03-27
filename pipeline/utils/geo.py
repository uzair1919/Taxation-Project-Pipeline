"""
pipeline/utils/geo.py
=====================
Lightweight geographic utility helpers used across all pipeline stages.

All functions are pure (no side effects) and dependency-minimal.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# CRS constants (kept in sync with refinement_utils.config)
# ---------------------------------------------------------------------------
WGS84_EPSG = 4326
UTM_EPSG = 32642  # UTM zone 42N — covers Pakistan; override via config if needed


# ---------------------------------------------------------------------------
# UTM ↔ WGS84 conversion helpers
# ---------------------------------------------------------------------------

def utm_geoms_to_wgs84(
    geoms_utm: list,
    utm_epsg: int = UTM_EPSG,
) -> gpd.GeoDataFrame:
    """
    Convert a list of Shapely UTM geometries to a WGS84 GeoDataFrame.

    Parameters
    ----------
    geoms_utm : list of Shapely geometries in UTM CRS
    utm_epsg  : EPSG code of the source UTM CRS

    Returns
    -------
    GeoDataFrame with CRS EPSG:4326
    """
    if not geoms_utm:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84_EPSG)
    return (
        gpd.GeoDataFrame(geometry=geoms_utm, crs=utm_epsg)
        .to_crs(epsg=WGS84_EPSG)
    )


def shapely_geom_to_wkt(geom) -> str:
    """
    Return a clean WKT POLYGON string for a Shapely geometry.

    Handles Polygon and MultiPolygon (uses the largest ring for Multi).
    Returns empty string for None / empty geoms.
    """
    if geom is None or geom.is_empty:
        return ""
    if geom.geom_type == "MultiPolygon":
        # Use the largest sub-polygon
        geom = max(geom.geoms, key=lambda p: p.area)
    if geom.geom_type != "Polygon":
        return ""
    coords = list(geom.exterior.coords)
    if not coords:
        return ""
    pts = ", ".join(f"{lon:.8f} {lat:.8f}" for lon, lat in coords)
    return f"POLYGON (({pts}))"


def parse_wkt_vertices(wkt: str) -> Optional[List[List[float]]]:
    """
    Parse a WKT POLYGON string → list of [lon, lat] vertex pairs.

    Returns None on any parse failure.  The closing duplicate vertex
    is NOT included (consistent with Shapely's exterior.coords[:-1]).

    Example
    -------
    >>> parse_wkt_vertices("POLYGON ((74.19 31.45, 74.20 31.45, ...))")
    [[74.19, 31.45], [74.20, 31.45], ...]
    """
    if not wkt or str(wkt).strip() in ("", "nan", "None"):
        return None
    import re
    match = re.search(r"POLYGON\s*\(\((.*?)\)\)", str(wkt), re.IGNORECASE)
    if not match:
        return None
    try:
        vertices = []
        for pair in match.group(1).split(","):
            parts = pair.strip().split()
            if len(parts) >= 2:
                vertices.append([float(parts[0]), float(parts[1])])
        # Drop duplicate closing vertex if present
        if len(vertices) > 1 and vertices[0] == vertices[-1]:
            vertices = vertices[:-1]
        return vertices if vertices else None
    except (ValueError, IndexError):
        return None


def wkt_to_bbox(wkt: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Return (west, south, east, north) bounding box from a WKT polygon.
    Returns None if WKT is invalid.
    """
    vertices = parse_wkt_vertices(wkt)
    if not vertices:
        return None
    lons = [v[0] for v in vertices]
    lats = [v[1] for v in vertices]
    return (min(lons), min(lats), max(lons), max(lats))


def bbox_corners_to_wkt(corners: List[List[float]]) -> str:
    """
    Convert a list of [lon, lat] corner points (4-5 points) to a WKT POLYGON.
    Closes the ring automatically if the last point != first point.

    Used for storing SAM's aligned_bbox_geo in the output Excel.
    """
    if not corners:
        return ""
    pts = list(corners)
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    coord_str = ", ".join(f"{lon:.8f} {lat:.8f}" for lon, lat in pts)
    return f"POLYGON (({coord_str}))"


def geo_to_pixel(
    lon: float, lat: float,
    ctx_west: float, ctx_south: float,
    ctx_east: float, ctx_north: float,
    img_w: int, img_h: int,
) -> Tuple[int, int]:
    """
    Convert a geographic (lon, lat) point to (px_x, px_y) pixel coordinates
    within a context image defined by its geographic bounding box.

    Y is inverted: north → top of image (px_y = 0).
    """
    px_x = int((lon - ctx_west) / (ctx_east - ctx_west) * img_w)
    px_y = int((ctx_north - lat) / (ctx_north - ctx_south) * img_h)
    px_x = max(0, min(img_w - 1, px_x))
    px_y = max(0, min(img_h - 1, px_y))
    return px_x, px_y


def compute_context_bbox(
    wkt_list: List[str],
    pad_fraction: float = 0.05,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Union bounding box of all valid WKT polygons, expanded by pad_fraction.

    Returns (west, south, east, north) or None if no valid polygons.
    """
    all_lons: List[float] = []
    all_lats: List[float] = []
    for wkt in wkt_list:
        verts = parse_wkt_vertices(wkt)
        if verts:
            all_lons.extend(v[0] for v in verts)
            all_lats.extend(v[1] for v in verts)
    if not all_lons:
        return None
    west  = min(all_lons)
    east  = max(all_lons)
    south = min(all_lats)
    north = max(all_lats)
    dx = east  - west
    dy = north - south
    return (
        west  - dx * pad_fraction,
        south - dy * pad_fraction,
        east  + dx * pad_fraction,
        north + dy * pad_fraction,
    )