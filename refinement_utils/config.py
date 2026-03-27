"""
config.py
=========
Central configuration for the refinement pipeline.
Edit the values here to change pipeline behaviour — everything else reads from this file.
"""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STREETS_SHP = r"pakistan-260222-free.shp/gis_osm_roads_free_1.shp"

# ESRI satellite tile URL (used in all visualisations)
ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Default output root when --output-dir is not given on the CLI
DEFAULT_OUTPUT_DIR = "refinement_output"

# ---------------------------------------------------------------------------
# Coordinate system
# ---------------------------------------------------------------------------
UTM_EPSG = 32643      # UTM Zone 43N — Pakistan
WGS84_EPSG = 4326
WEB_MERCATOR_EPSG = 3857

# ---------------------------------------------------------------------------
# Stage 1 — global alignment
# ---------------------------------------------------------------------------
STAGE1 = dict(
    # Greedy translation step search
    initial_step_m      = 10.0,    # starting step size (metres)
    min_step_m          = 0.1,     # stop translating when step drops below this

    # Loss function
    proximity_weight    = 0.5,     # weight on mean-distance-to-road term
                                   # (higher → tries harder to close gaps)

    # Scale search (runs only if overlaps remain after translation)
    initial_scale_step  = 0.1,     # starting scale increment (fraction)
    min_scale_step      = 0.001,   # stop scaling when increment drops below this
    max_scale_factor    = 1.5,     # hard cap on how much to expand
    min_scale_factor    = 0.5,     # hard cap on how much to shrink
)

# ---------------------------------------------------------------------------
# Stage 2 — per-cluster refinement
# ---------------------------------------------------------------------------
STAGE2 = dict(
    # BFS cluster detection
    gap_threshold_m     = 4.0,     # two plots are in the same cluster if their
                                   # buffered outlines (gap/2 each) overlap

    # Road loading buffer around each cluster
    search_buffer_m     = 80.0,    # metres to expand bbox when loading OSM roads

    # Tile expansion for extra plots around each cluster
    expansion_rings     = 1,       # how many tiles outward to expand from the
                                   # cluster centroid tile

    # Pre-processing intersection clearing
    clear_initial_step  = 0.5,     # starting step for clearing road overlaps (m)
    clear_min_step      = 0.01,    # minimum step before giving up
    clear_max_move      = 25.0,    # maximum total movement allowed (m)

    # Post-clear fine translation
    refine_step_m       = 2.0,     # starting step for fine translation (m)
    min_refine_step     = 0.05,    # minimum step for fine translation (m)

    # Directional stretch
    refine_scale_step   = 0.02,    # starting scale increment per direction
    min_refine_scale    = 0.001,   # stop stretching when increment drops below this
    min_scale_factor    = 0.5,     # hard cap — cannot shrink a cluster below 50%
    max_scale_factor    = 2.0,     # hard cap — cannot expand a cluster above 200%

    # Deduplication tolerance (tile-fetched plots vs pipeline plots)
    dedup_distance_m    = 3.0,     # centroids closer than this are considered duplicates

    # Tile-expansion connectivity filter
    # A plot fetched from an adjacent tile is only kept if its centroid is
    # within this distance of an existing stage-1 plot.  Set to a small
    # multiple of gap_threshold_m so only genuine boundary-crossing plots
    # are admitted; anything further away is an unrelated plot and discarded.
    tile_connect_gap_m  = 8.0,     # metres (default = 2x gap_threshold_m)
)

# ---------------------------------------------------------------------------
# Tile stitching
# ---------------------------------------------------------------------------
TILE_OVERLAP_PERCENT = 0.01       # must match the pipeline capture parameter

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
VIZ = dict(
    satellite_zoom      = 19,      # ESRI tile zoom level (falls back to 18, 17)
    plot_dpi            = 150,     # DPI for saved PNG plots
    stage1_figsize      = (20, 10),
    stage2_figsize      = (22, 11),
    bbox_pad_fraction   = 0.15,    # extra padding around cluster bbox in plots
)