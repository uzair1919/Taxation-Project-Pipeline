"""
esri_tile_fetcher.py — ESRI World Imagery Tile Fetcher
=======================================================
Downloads high-resolution satellite imagery from ESRI's World Imagery service.
This is the same source you used for your correction algorithm.

ESRI tile server:
  https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}

Supports:
  - Fetching a bounding box at a given zoom level
  - Automatic tile stitching
  - Coordinate conversion (lon/lat ↔ tile indices)
"""

import io
import logging
import math
import time
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


class ESRITileFetcher:
    """
    Fetches and stitches ESRI World Imagery tiles for a given bounding box.
    """
    
    TILE_URL_TEMPLATE = (
        "https://services.arcgisonline.com/arcgis/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    
    TILE_SIZE = 256  # pixels per tile
    
    def __init__(self, cache_dir: Path | None = None, user_agent: str | None = None):
        """
        Parameters
        ----------
        cache_dir   : Optional directory to cache downloaded tiles
        user_agent  : Optional custom User-Agent header
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent or (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })
    
    # ── Public API ──────────────────────────────────────────────────────────
    
    def fetch_bbox(self,
                   west: float,
                   south: float,
                   east: float,
                   north: float,
                   zoom: int = 19,
                   output_path: Path | None = None) -> Image.Image:
        """
        Fetch and stitch tiles covering a bounding box.
        
        Parameters
        ----------
        west, south, east, north : Geographic bounds in decimal degrees
        zoom                     : Zoom level (0-23). 19 = ~0.3m/px in Pakistan
        output_path              : Optional path to save the stitched image
        
        Returns
        -------
        PIL.Image (RGB)
        """
        logger.debug(
            f"Fetching ESRI bbox: ({west:.5f}, {south:.5f}) to "
            f"({east:.5f}, {north:.5f}) at zoom {zoom}"
        )
        
        # ── Convert bbox to tile indices ────────────────────────────────────
        nw_tile_x, nw_tile_y = self._lonlat_to_tile(west, north, zoom)
        se_tile_x, se_tile_y = self._lonlat_to_tile(east, south, zoom)
        
        min_tile_x = min(nw_tile_x, se_tile_x)
        max_tile_x = max(nw_tile_x, se_tile_x)
        min_tile_y = min(nw_tile_y, se_tile_y)
        max_tile_y = max(nw_tile_y, se_tile_y)
        
        n_tiles_x = max_tile_x - min_tile_x + 1
        n_tiles_y = max_tile_y - min_tile_y + 1
        
        logger.debug(
            f"  Tile range: x=[{min_tile_x}, {max_tile_x}], "
            f"y=[{min_tile_y}, {max_tile_y}]  ({n_tiles_x}×{n_tiles_y} tiles)"
        )
        
        if n_tiles_x * n_tiles_y > 400:
            logger.warning(
                f"Requesting {n_tiles_x * n_tiles_y} tiles — this may be slow"
            )
        
        # ── Download tiles ──────────────────────────────────────────────────
        tile_array = [[None for _ in range(n_tiles_x)] for _ in range(n_tiles_y)]
        
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                tile_x = min_tile_x + tx
                tile_y = min_tile_y + ty
                
                tile_img = self._fetch_tile(tile_x, tile_y, zoom)
                tile_array[ty][tx] = tile_img
                
                # Rate limiting
                time.sleep(0.05)
        
        # ── Stitch tiles ────────────────────────────────────────────────────
        stitched = self._stitch_tiles(tile_array)
        
        # ── Crop to exact bbox ──────────────────────────────────────────────
        # The tiles cover a slightly larger area than the requested bbox
        # We need to crop to the exact geographic bounds
        
        # Top-left corner of the tile grid in lon/lat
        grid_west, grid_north = self._tile_to_lonlat(min_tile_x, min_tile_y, zoom)
        # Bottom-right corner
        grid_east, grid_south = self._tile_to_lonlat(
            max_tile_x + 1, max_tile_y + 1, zoom
        )
        
        # Pixel coordinates of the requested bbox within the stitched image
        img_w, img_h = stitched.size
        
        crop_left   = int((west  - grid_west)  / (grid_east - grid_west)  * img_w)
        crop_right  = int((east  - grid_west)  / (grid_east - grid_west)  * img_w)
        crop_top    = int((grid_north - north) / (grid_north - grid_south) * img_h)
        crop_bottom = int((grid_north - south) / (grid_north - grid_south) * img_h)
        
        # Clamp to image bounds
        crop_left   = max(0, min(img_w, crop_left))
        crop_right  = max(0, min(img_w, crop_right))
        crop_top    = max(0, min(img_h, crop_top))
        crop_bottom = max(0, min(img_h, crop_bottom))
        
        cropped = stitched.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        logger.debug(
            f"  Stitched {n_tiles_x}×{n_tiles_y} tiles → {stitched.size} px, "
            f"cropped to {cropped.size} px"
        )
        
        # ── Save if requested ───────────────────────────────────────────────
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(output_path, quality=95)
            logger.debug(f"  Saved → {output_path}")
        
        return cropped
    
    # ── Tile download ───────────────────────────────────────────────────────
    
    def _fetch_tile(self, x: int, y: int, z: int) -> Image.Image:
        """
        Fetch a single tile from ESRI, with caching if enabled.
        
        Returns
        -------
        PIL.Image (RGB, 256×256)
        """
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"esri_z{z}_x{x}_y{y}.png"
            if cache_path.exists():
                return Image.open(cache_path).convert("RGB")
        
        # Download from ESRI
        url = self.TILE_URL_TEMPLATE.format(z=z, y=y, x=x)
        
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            
            # Save to cache
            if self.cache_dir:
                cache_path = self.cache_dir / f"esri_z{z}_x{x}_y{y}.png"
                img.save(cache_path)
            
            return img
        
        except Exception as exc:
            logger.warning(f"Failed to fetch tile z={z} x={x} y={y}: {exc}")
            # Return a blank tile as fallback
            return Image.new("RGB", (self.TILE_SIZE, self.TILE_SIZE), (200, 200, 200))
    
    # ── Tile stitching ──────────────────────────────────────────────────────
    
    def _stitch_tiles(self, tile_array: list[list[Image.Image]]) -> Image.Image:
        """
        Stitch a 2D array of tiles into a single image.
        
        Parameters
        ----------
        tile_array : list of lists of PIL.Image (all same size)
        
        Returns
        -------
        PIL.Image (RGB)
        """
        n_rows = len(tile_array)
        n_cols = len(tile_array[0]) if n_rows > 0 else 0
        
        if n_rows == 0 or n_cols == 0:
            return Image.new("RGB", (self.TILE_SIZE, self.TILE_SIZE), (0, 0, 0))
        
        tile_w, tile_h = self.TILE_SIZE, self.TILE_SIZE
        canvas_w = n_cols * tile_w
        canvas_h = n_rows * tile_h
        
        canvas = Image.new("RGB", (canvas_w, canvas_h))
        
        for row_idx, row in enumerate(tile_array):
            for col_idx, tile in enumerate(row):
                if tile is None:
                    tile = Image.new("RGB", (tile_w, tile_h), (200, 200, 200))
                x = col_idx * tile_w
                y = row_idx * tile_h
                canvas.paste(tile, (x, y))
        
        return canvas
    
    # ── Coordinate conversions (Web Mercator) ───────────────────────────────
    
    def _lonlat_to_tile(self, lon: float, lat: float, zoom: int) -> tuple[int, int]:
        """
        Convert (lon, lat) to tile indices (x, y) at a given zoom level.
        Uses Web Mercator projection (EPSG:3857).
        
        Returns
        -------
        (tile_x, tile_y)
        """
        lat_rad = math.radians(lat)
        n = 2 ** zoom
        
        tile_x = int((lon + 180.0) / 360.0 * n)
        tile_y = int(
            (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
            / 2.0 * n
        )
        
        return tile_x, tile_y
    
    def _tile_to_lonlat(self, tile_x: int, tile_y: int, zoom: int) -> tuple[float, float]:
        """
        Convert tile indices to the northwest corner (lon, lat).
        
        Returns
        -------
        (lon, lat) of the tile's top-left corner
        """
        n = 2 ** zoom
        lon = tile_x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
        lat = math.degrees(lat_rad)
        return lon, lat