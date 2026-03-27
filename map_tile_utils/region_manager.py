"""
REGION MANAGER

Main module for region database management and coordinate lookup.
All operations are function-based for easy pipeline integration.
"""

import json
import csv
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ==================== CONFIGURATION ====================

# Use absolute paths based on the module's location so the database is always
# found regardless of the working directory or OS.
_MODULE_DIR = Path(__file__).resolve().parent.parent  # FINAL_PIPELINE root

DATA_ROOT = str(_MODULE_DIR / "regional_data")
DB_FILE   = str(_MODULE_DIR / "regional_data" / "regions.json")


def _normalize_path(path_str: str) -> str:
    """
    Normalise a stored path to use forward slashes so it works on both
    Windows and Linux.  Stored paths may contain Windows backslashes if the
    database was originally created on Windows.
    """
    return path_str.replace("\\", "/")


# ==================== DATABASE OPERATIONS ====================

def _load_database(db_path: str = DB_FILE) -> Dict:
    """Load regions database from file."""
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            return json.load(f)
    return {'regions': {}}


def _save_database(db: Dict, db_path: str = DB_FILE):
    """Save regions database to file."""
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    with open(db_path, 'w') as f:
        json.dump(db, f, indent=2)


def _read_tile_csv(csv_path: str) -> List[Dict]:
    """Read tile coordinates from CSV file."""
    tiles = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile = {
                'filename': row['Tile_Name'],
                'corners': {
                    'bl': {'lat': float(row['BL_Lat']), 'lon': float(row['BL_Lon'])},
                    'br': {'lat': float(row['BR_Lat']), 'lon': float(row['BR_Lon'])},
                    'tr': {'lat': float(row['TR_Lat']), 'lon': float(row['TR_Lon'])},
                    'tl': {'lat': float(row['TL_Lat']), 'lon': float(row['TL_Lon'])}
                }
            }
            
            # Parse row and column from filename (tile_rX_cY.png)
            parts = tile['filename'].replace('.png', '').split('_')
            if len(parts) >= 3:
                tile['row'] = int(parts[1][1:])
                tile['col'] = int(parts[2][1:])
            
            tiles.append(tile)
    
    return tiles


def _calculate_bounds(tiles: List[Dict]) -> Dict:
    """Calculate bounding box for tiles."""
    all_lats = []
    all_lons = []
    
    for tile in tiles:
        c = tile['corners']
        all_lats.extend([c['bl']['lat'], c['br']['lat'], c['tr']['lat'], c['tl']['lat']])
        all_lons.extend([c['bl']['lon'], c['br']['lon'], c['tr']['lon'], c['tl']['lon']])
    
    return {
        'north': max(all_lats),
        'south': min(all_lats),
        'east': max(all_lons),
        'west': min(all_lons),
        'center_lat': sum(all_lats) / len(all_lats),
        'center_lon': sum(all_lons) / len(all_lons)
    }


# ==================== PUBLIC API - REGION MANAGEMENT ====================

def add_region(region_name: str, data_root: str = DATA_ROOT, db_path: str = DB_FILE) -> Dict:
    """
    Add or update a region in the database using standard folder structure.
    
    Expected structure:
        {data_root}/{region_name}/map_tiles/tile_r0_c0.png
        {data_root}/{region_name}/tile_coordinates/{region_name}.csv
    
    Args:
        region_name: Name of the region (e.g., "sukh_chain")
        data_root: Root folder containing regional data (default: "regional_data")
        db_path: Database file path (default: "regions.json")
    
    Returns:
        Region data dictionary
        
    Example:
        >>> add_region("sukh_chain")
        >>> add_region("dha_phase5", data_root="data")
    """
    # Construct paths using forward slashes so they are stored portably in JSON
    region_folder = (Path(data_root) / region_name).as_posix()
    tile_folder   = (Path(data_root) / region_name / "map_tiles").as_posix()
    csv_path      = (Path(data_root) / region_name / "tile_coordinates" / f"{region_name}.csv").as_posix()
    
    # Validate paths
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(tile_folder):
        raise FileNotFoundError(f"Tile folder not found: {tile_folder}")
    
    # Read tiles from CSV
    tiles = _read_tile_csv(csv_path)
    
    # Calculate region bounds
    bounds = _calculate_bounds(tiles)
    
    # Determine grid size
    max_row = max(t['row'] for t in tiles)
    max_col = max(t['col'] for t in tiles)
    
    # Create region entry
    region_data = {
        'name': region_name,
        'tile_folder': tile_folder,
        'csv_path': csv_path,  # Added csv_path to the database entry
        'grid_rows': max_row + 1,
        'grid_cols': max_col + 1,
        'bounds': bounds,
        'tiles': tiles
    }
    
    # Update database
    db = _load_database(db_path)
    db['regions'][region_name] = region_data
    _save_database(db, db_path)
    
    return region_data


def remove_region(region_name: str, db_path: str = DB_FILE) -> bool:
    """
    Remove a region from the database.
    
    Args:
        region_name: Name of the region to remove
        db_path: Database file path
    
    Returns:
        True if removed, False if not found
        
    Example:
        >>> remove_region("old_region")
    """
    db = _load_database(db_path)
    
    if region_name in db['regions']:
        del db['regions'][region_name]
        _save_database(db, db_path)
        return True
    
    return False


def list_regions(db_path: str = DB_FILE) -> List[str]:
    """
    Get list of all region names in database.
    
    Args:
        db_path: Database file path
    
    Returns:
        List of region names
        
    Example:
        >>> regions = list_regions()
        >>> print(regions)
        ['sukh_chain', 'dha_phase5']
    """
    db = _load_database(db_path)
    return list(db['regions'].keys())


def get_region_info(region_name: str, db_path: str = DB_FILE) -> Optional[Dict]:
    """
    Get detailed information about a region.
    
    Args:
        region_name: Name of the region
        db_path: Database file path
    
    Returns:
        Region data dictionary or None if not found
        
    Example:
        >>> info = get_region_info("sukh_chain")
        >>> print(info['grid_rows'], info['grid_cols'])
    """
    db = _load_database(db_path)
    return db['regions'].get(region_name)


# ==================== PUBLIC API - COORDINATE LOOKUP ====================

def _point_in_bounds(lat: float, lon: float, bounds: Dict) -> bool:
    """Check if point is within bounding box."""
    return (bounds['south'] <= lat <= bounds['north'] and
            bounds['west'] <= lon <= bounds['east'])


def _point_in_quadrilateral(lat: float, lon: float, corners: Dict) -> bool:
    """Check if point is inside quadrilateral using cross product method."""
    pts = [
        (corners['bl']['lat'], corners['bl']['lon']),
        (corners['br']['lat'], corners['br']['lon']),
        (corners['tr']['lat'], corners['tr']['lon']),
        (corners['tl']['lat'], corners['tl']['lon'])
    ]
    
    point = (lat, lon)
    
    def cross_product_sign(p1, p2, p):
        return ((p2[0] - p1[0]) * (p[1] - p1[1]) - 
                (p2[1] - p1[1]) * (p[0] - p1[0]))
    
    signs = [cross_product_sign(pts[i], pts[(i + 1) % 4], point) for i in range(4)]
    
    positive = sum(1 for s in signs if s > -1e-10)
    negative = sum(1 for s in signs if s < 1e-10)
    
    return positive == 4 or negative == 4


def lookup_coordinate(lat: float, lon: float, buffer: int = 0, db_path: str = DB_FILE) -> Dict:
    """
    Find which region and tiles contain a coordinate.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        buffer: Number of surrounding tiles to include (0 = only center tile)
        db_path: Database file path
    
    Returns:
        Dictionary with keys:
        - found (bool): Whether coordinate was found in any region
        - region (dict): Region information if found
        - tiles (list): List of covering tiles if found
        - tile_count (int): Number of tiles if found
        
    Example:
        >>> result = lookup_coordinate(31.36148, 74.22393)
        >>> if result['found']:
        ...     print(f"Found in {result['region']['name']}")
        ...     print(f"Tiles: {[t['filename'] for t in result['tiles']]}")
        ...     print(f"CSV Path: {result['region']['csv_path']}")
        
        >>> # With buffer for surrounding tiles
        >>> result = lookup_coordinate(31.36148, 74.22393, buffer=1)
        >>> print(f"Got {result['tile_count']} tiles (3x3 grid)")
    """
    # Load database
    db = _load_database(db_path)
    
    if not db['regions']:
        return {
            'found': False,
            'error': 'No regions in database'
        }
    
    # Check each region
    for region_name, region in db['regions'].items():
        # Quick bounds check
        if not _point_in_bounds(lat, lon, region['bounds']):
            continue
        
        # Find center tile that contains the point
        center_tile = None
        for tile in region['tiles']:
            if _point_in_quadrilateral(lat, lon, tile['corners']):
                center_tile = tile
                break
        
        if not center_tile:
            continue
        
        # Collect tiles
        tiles = [center_tile]
        
        # Add buffer tiles if requested
        if buffer > 0:
            tile_lookup = {(t['row'], t['col']): t for t in region['tiles']}
            cr, cc = center_tile['row'], center_tile['col']
            
            for r in range(cr - buffer, cr + buffer + 1):
                for c in range(cc - buffer, cc + buffer + 1):
                    if (r, c) != (cr, cc) and (r, c) in tile_lookup:
                        tiles.append(tile_lookup[(r, c)])
        
        # Mark center tile
        for tile in tiles:
            tile['is_center'] = (tile == center_tile)
        
        return {
            'found': True,
            'region': {
                'name': region['name'],
                'tile_folder': _normalize_path(region['tile_folder']),
                'csv_path': _normalize_path(region.get('csv_path', '')),
                'grid_rows': region['grid_rows'],
                'grid_cols': region['grid_cols'],
                'bounds': region['bounds']
            },
            'tiles': tiles,
            'tile_count': len(tiles)
        }
    
    # Not found
    return {
        'found': False,
        'message': 'Coordinate not found in any region',
        'available_regions': list(db['regions'].keys())
    }

def get_tile_paths(result: Dict) -> List[str]:
    """
    Extract full tile file paths from lookup result.
    
    Args:
        result: Result dictionary from lookup_coordinate()
    
    Returns:
        List of full tile paths, empty list if not found
        
    Example:
        >>> result = lookup_coordinate(31.36148, 74.22393, buffer=1)
        >>> paths = get_tile_paths(result)
        >>> for path in paths:
        ...     process_tile(path)
    """
    if not result.get('found'):
        return []
    
    folder = result['region']['tile_folder']
    return [os.path.join(folder, tile['filename']) for tile in result['tiles']]


def get_center_tile_path(result: Dict) -> Optional[str]:
    """
    Get path to the center tile (the one containing the coordinate).
    
    Args:
        result: Result dictionary from lookup_coordinate()
    
    Returns:
        Path to center tile, or None if not found
        
    Example:
        >>> result = lookup_coordinate(31.36148, 74.22393)
        >>> center = get_center_tile_path(result)
        >>> if center:
        ...     process_main_tile(center)
    """
    if not result.get('found'):
        return None
    
    folder = result['region']['tile_folder']
    center_tile = next((t for t in result['tiles'] if t.get('is_center')), None)
    
    if center_tile:
        return os.path.join(folder, center_tile['filename'])
    
    return None


# ==================== BATCH OPERATIONS ====================

def lookup_batch(coordinates: List[Tuple[float, float]], buffer: int = 0, 
                 db_path: str = DB_FILE) -> List[Dict]:
    """
    Look up multiple coordinates at once.
    
    Args:
        coordinates: List of (lat, lon) tuples
        buffer: Buffer size for all lookups
        db_path: Database file path
    
    Returns:
        List of result dictionaries
        
    Example:
        >>> coords = [(31.36148, 74.22393), (31.36200, 74.22300)]
        >>> results = lookup_batch(coords)
        >>> for i, result in enumerate(results):
        ...     if result['found']:
        ...         print(f"Coord {i}: {result['region']['name']}")
    """
    return [lookup_coordinate(lat, lon, buffer, db_path) for lat, lon in coordinates]


def add_multiple_regions(region_names: List[str], data_root: str = DATA_ROOT, 
                         db_path: str = DB_FILE) -> Dict[str, Dict]:
    """
    Add multiple regions at once.
    
    Args:
        region_names: List of region names to add
        data_root: Root folder for regional data
        db_path: Database file path
    
    Returns:
        Dictionary mapping region names to their data (or error info)
        
    Example:
        >>> results = add_multiple_regions(['sukh_chain', 'dha_phase5'])
        >>> for name, info in results.items():
        ...     if 'error' not in info:
        ...         print(f"Added {name}: {info['grid_rows']}x{info['grid_cols']}")
    """
    results = {}
    
    for region_name in region_names:
        try:
            region_data = add_region(region_name, data_root, db_path)
            results[region_name] = region_data
        except Exception as e:
            results[region_name] = {'error': str(e)}
    
    return results


# ==================== UTILITY FUNCTIONS ====================

def region_exists(region_name: str, db_path: str = DB_FILE) -> bool:
    """
    Check if a region exists in the database.
    
    Args:
        region_name: Name of the region
        db_path: Database file path
    
    Returns:
        True if region exists
        
    Example:
        >>> if not region_exists("sukh_chain"):
        ...     add_region("sukh_chain")
    """
    db = _load_database(db_path)
    return region_name in db['regions']


def ensure_region(region_name: str, data_root: str = DATA_ROOT, db_path: str = DB_FILE) -> Dict:
    """
    Ensure a region exists in database, adding it if necessary.
    
    Args:
        region_name: Name of the region
        data_root: Root folder for regional data
        db_path: Database file path
    
    Returns:
        Region data dictionary
        
    Example:
        >>> # Always have the region ready
        >>> region = ensure_region("sukh_chain")
        >>> result = lookup_coordinate(lat, lon)
    """
    if not region_exists(region_name, db_path):
        return add_region(region_name, data_root, db_path)
    else:
        return get_region_info(region_name, db_path)


if __name__ == "__main__":
    # Example usage
    print("Region Manager - Example Usage\n")
    
    # Check existing regions
    regions = list_regions()
    print(f"Existing regions: {regions if regions else 'None'}\n")
    
    # Example: Add a region (uncomment to use)
    # region_data = add_region("sukh_chain")
    # print(f"Added region: {region_data['name']}")
    # print(f"Grid size: {region_data['grid_rows']}x{region_data['grid_cols']}\n")
    
    # Example: Lookup coordinate
    if regions:
        result = lookup_coordinate(31.36148, 74.22393, buffer=1)
        
        if result['found']:
            print(f"✅ Coordinate found in: {result['region']['name']}")
            print(f"   Tiles: {result['tile_count']}")
            
            paths = get_tile_paths(result)
            print(f"   Paths: {paths[:2]}...")
        else:
            print("❌ Coordinate not found")
    else:
        print("No regions in database. Add one first:")
        print('  add_region("sukh_chain")')