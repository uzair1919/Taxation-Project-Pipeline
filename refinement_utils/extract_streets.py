import geopandas as gpd
import matplotlib.pyplot as plt

def extract_urban_streets_local(north, south, east, west, shp_path, output_filename="street_mask.png"):
    # print(f"Slicing local shapefile for box: N:{north}, S:{south}, E:{east}, W:{west}...")

    # 1. Load ONLY the bounding box area from the 330MB file
    # Order for GeoPandas bbox is (minx, miny, maxx, maxy) -> (west, south, east, north)
    bbox = (west, south, east, north)
    
    try:
        # This reads only the necessary rows from the disk
        edges = gpd.read_file(shp_path, bbox=bbox)
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return

    if edges.empty:
        print("No streets found in this bounding box. Check your coordinates or file path.")
        return

    # # 2. Save Vector Data (Optional)
    # edges.to_file("streets_vector.geojson", driver='GeoJSON')
    # print(f"Found {len(edges)} street segments. Vector saved.")

    # 3. Generate the Binary Mask
    # Increase DPI or figsize if your satellite image resolution is very high
    fig, ax = plt.subplots(figsize=(15, 15), facecolor='black')
    
    # Adjust linewidth based on how thick you want the 'road' to be in the mask
    edges.plot(ax=ax, color='white', linewidth=2.0, alpha=1)

    # Clean up the plot to get a pure mask
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    # Set the plot limits to match your bounding box exactly
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    
    # # Save as PNG
    # plt.savefig(output_filename, facecolor='black', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # print(f"Binary mask saved as {output_filename}")



if __name__ == "__main__":
    path_to_roads = "pakistan-260222-free.shp/gis_osm_roads_free_1.shp" 

    # Your coordinates
    north, south, east, west = 31.39296886, 31.39087383, 74.16044475, 74.15540894
    extract_urban_streets_local(north, south, east, west, path_to_roads)