import csv
from PIL import Image
import os
import json

def load_all_bounds(csv_path):
    all_tiles = {}
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Tile_Name']
            # Standardizing coordinates to match region_manager naming
            # We use TL (Top Left) for North/West and BR (Bottom Right) for South/East
            try:
                parts = name.replace(".png", "").split("_")
                r = int(parts[1][1:])
                c = int(parts[2][1:])
                all_tiles[(r, c)] = {
                    "name": name,
                    "north": float(row['TL_Lat']),
                    "south": float(row['BR_Lat']),
                    "east": float(row['BR_Lon']),
                    "west": float(row['TL_Lon'])
                }
            except KeyError as e:
                print(f"❌ Error: Missing column {e} in CSV. Check if CSV headers match region_manager format.")
                raise
    return all_tiles

def get_closest_four_tiles(lat, lon, tile_data):
    home_r, home_c, home_bounds = None, None, None
    for (r, c), bounds in tile_data.items():
        if bounds['south'] <= lat <= bounds['north'] and bounds['west'] <= lon <= bounds['east']:
            home_r, home_c, home_bounds = r, c, bounds
            break
            
    if home_r is None: return None

    # Determine quadrant center point to find which 4 tiles are closest
    mid_lat = (home_bounds['north'] + home_bounds['south']) / 2
    mid_lon = (home_bounds['east'] + home_bounds['west']) / 2
    
    # Select North/South and East/West neighbors
    row_mod = -1 if lat > mid_lat else 1
    col_mod = -1 if lon < mid_lon else 1
    
    rows = sorted([home_r, home_r + row_mod])
    cols = sorted([home_c, home_c + col_mod])
    
    selected_indices = []
    for r in rows:
        for c in cols:
            if (r, c) in tile_data:
                selected_indices.append((r, c))
            else:
                return None # Return None if we are at the edge of the map
    return selected_indices

def get_stitched_bounds_dict(selected_indices, tile_data):
    # Sort indices to ensure TL is first and BR is last
    # selected_indices are usually [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
    tl_idx = selected_indices[0] 
    br_idx = selected_indices[3] 
    
    return {
        "north": tile_data[tl_idx]['north'],
        "south": tile_data[br_idx]['south'],
        "east": tile_data[br_idx]['east'],
        "west": tile_data[tl_idx]['west']
    }

def stitch_four_with_overlap(indices, tile_data, folder, output_name, OVERLAP_PERCENT):
    tile_names = [tile_data[idx]['name'] for idx in indices]
    imgs = [Image.open(os.path.join(folder, name)) for name in tile_names]
    
    tile_w, tile_h = imgs[0].size
    overlap_x = int(tile_w * OVERLAP_PERCENT)
    overlap_y = int(tile_h * OVERLAP_PERCENT)

    final_w = (tile_w * 2) - overlap_x
    final_h = (tile_h * 2) - overlap_y
    
    final_image = Image.new('RGB', (final_w, final_h))
    
    # Positions for: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    positions = [
        (0, 0),
        (tile_w - overlap_x, 0),
        (0, tile_h - overlap_y),
        (tile_w - overlap_x, tile_h - overlap_y)
    ]

    for i in range(4):
        final_image.paste(imgs[i], positions[i])
    
    final_image.save(output_name, quality=95)
    return output_name

def exec(target_lat, target_lon, CSV_PATH, TILE_FOLDER, OVERLAP_PERCENT, OUTPUT_NAME):
    data = load_all_bounds(CSV_PATH)
    indices = get_closest_four_tiles(target_lat, target_lon, data)

    if indices:
        image_name = stitch_four_with_overlap(indices, data, TILE_FOLDER, OUTPUT_NAME, OVERLAP_PERCENT)
        geo_bounds = get_stitched_bounds_dict(indices, data)
        
        geo_bounds_data = {
            "north": float(f"{geo_bounds['north']:.8f}"),
            "south": float(f"{geo_bounds['south']:.8f}"),
            "east": float(f"{geo_bounds['east']:.8f}"),
            "west": float(f"{geo_bounds['west']:.8f}")
        }

        # print(f"   ✅ Context image saved as {image_name}")
        return geo_bounds_data
    else:
        print(f"   ❌ Could not find 4 surrounding tiles for {target_lat}, {target_lon}. Point might be on the map edge.")
        return None