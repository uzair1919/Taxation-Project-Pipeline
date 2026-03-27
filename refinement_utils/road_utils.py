"""
road_utils.py
=============
Shared road-width and road-surface helpers used by both stage1 and stage2.
"""

import geopandas as gpd
from shapely.ops import unary_union

_ROAD_WIDTH_DEFAULTS = {
    "motorway":       28.0, "motorway_link":  14.0,
    "trunk":          22.0, "trunk_link":     11.0,
    "primary":        16.0, "primary_link":    8.0,
    "secondary":      12.0, "secondary_link":  6.0,
    "tertiary":       10.0, "tertiary_link":   5.0,
    "residential":     8.0, "living_street":   6.0,
    "service":         6.0, "track":           4.0,
    "path":            3.0, "footway":         3.0,
    "cycleway":        3.0, "unclassified":    7.0,
}
_DEFAULT_ROAD_WIDTH = 7.0


def road_half_width(row) -> float:
    for col in ("width_m", "width", "WIDTH_M", "WIDTH"):
        val = row.get(col, None)
        if val is not None:
            try:
                w = float(val)
                if w > 0:
                    return w / 2.0
            except (TypeError, ValueError):
                pass
    fclass = str(row.get("fclass", row.get("highway", ""))).lower().strip()
    if fclass.startswith("["):
        fclass = fclass.strip("[]'\" ").split(",")[0].strip("'\" ")
    return _ROAD_WIDTH_DEFAULTS.get(fclass, _DEFAULT_ROAD_WIDTH) / 2.0


def build_road_surface(streets_utm):
    """Buffer each road by its half-width and union into a single surface."""
    polys = []
    for _, row in streets_utm.iterrows():
        hw = road_half_width(row)
        try:
            polys.append(row.geometry.buffer(hw, cap_style=2, join_style=2))
        except Exception:
            polys.append(row.geometry.buffer(_DEFAULT_ROAD_WIDTH / 2))
    return unary_union(polys)


def load_roads_for_bbox(streets_shp, wgs84_bbox, utm_epsg):
    """
    Load OSM roads from shapefile within a WGS84 bbox and reproject to UTM.

    Parameters
    ----------
    streets_shp  : str — path to shapefile
    wgs84_bbox   : (west, south, east, north)
    utm_epsg     : int

    Returns (roads_utm GeoDataFrame, road_surface Shapely geometry)
    """
    w, s, e, n = wgs84_bbox
    roads_raw = gpd.read_file(streets_shp, bbox=(w, s, e, n))
    if roads_raw.empty:
        raise ValueError(f"No roads found in bbox {wgs84_bbox}")
    roads_utm    = roads_raw.to_crs(epsg=utm_epsg)
    road_surface = build_road_surface(roads_utm)
    return roads_utm, road_surface