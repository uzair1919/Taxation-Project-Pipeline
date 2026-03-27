"""
config.py — Unified Pipeline Configuration
===========================================
Single source of truth for the combined refinement + SAM2 pipeline.
Edit values here before running orchestrator.py.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  REPOSITORY / PATH LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Directories containing refinement_utils/, map_tile_utils/, esri_tile_fetcher etc.
# Set to BASE_DIR if everything shares the same root; override if they are siblings.
REFINEMENT_ROOT = BASE_DIR
SAM2_ROOT       = BASE_DIR

# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = BASE_DIR / "pipeline_output"
FINAL_EXCEL_NAME   = "final_dataset.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
#  WHAT TO SAVE TO DISK (size controls for large datasets)
# ─────────────────────────────────────────────────────────────────────────────

# Refinement diagnostic plots (stage1_utm.png, stage1_satellite.png, etc.)
SAVE_REFINEMENT_PLOTS = False

# GeoJSON files for each point (stage1_plots.geojson, stage2_plots.geojson)
SAVE_GEOJSON = False

# SAM .npy binary mask files (one per plot, referenced by mask_path in Excel)
SAVE_SAM_MASKS = False

# SAM translucent overlay PNGs (visualisation only)
SAVE_SAM_OVERLAYS = False

# ESRI context images fetched for SAM inference
SAVE_SAM_CONTEXT_IMAGES = False

# ─────────────────────────────────────────────────────────────────────────────
#  VERBOSITY
# ─────────────────────────────────────────────────────────────────────────────

# Main orchestrator always logs at INFO to the log file.
# These flags control whether sub-pipeline verbose output appears on stdout.
REFINEMENT_VERBOSE = False   # stage1/stage2 iteration-level logs
SAM_VERBOSE        = False   # SAM2 batch-processing logs

# ─────────────────────────────────────────────────────────────────────────────
#  REFINEMENT — COORDINATE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

UTM_EPSG          = 32642   # UTM zone 42N (Pakistan). Change for other regions.
WGS84_EPSG        = 4326
WEB_MERCATOR_EPSG = 3857

# ─────────────────────────────────────────────────────────────────────────────
#  REFINEMENT — INPUT DATA
# ─────────────────────────────────────────────────────────────────────────────

STREETS_SHP          = BASE_DIR / "OSM_data" / "gis_osm_roads_free_1.shp"
TILE_OVERLAP_PERCENT = 0.01

# ─────────────────────────────────────────────────────────────────────────────
#  REFINEMENT — STAGE 1 PARAMS
# ─────────────────────────────────────────────────────────────────────────────

STAGE1 = {
    "initial_step_m":     10.0,     # starting step size (metres)
    "min_step_m":         0.05,     # stop translating when step drops below this
    "initial_scale_step": 0.02,     # starting scale increment (fraction)
    "min_scale_step":     0.001,    # stop scaling when increment drops below this
    "min_scale_factor":   0.5,      # hard cap on how much to shrink
    "max_scale_factor":   1.5,      # hard cap on how much to expand
    "proximity_weight":   0.5,     # weight on mean-distance-to-road term (higher → tries harder to close gaps)
}

# ─────────────────────────────────────────────────────────────────────────────
#  REFINEMENT — STAGE 2 PARAMS
# ─────────────────────────────────────────────────────────────────────────────

STAGE2 = {
    # BFS cluster detection
    "gap_threshold_m":               4.0,     # Max gap between plots to be considered within the same cluster.
    "search_buffer_m":               80.0,    # Distance to expand bounding box when loading OSM road data.
    "expansion_rings":               1,       # Number of tiles to search outward from the cluster centroid tile.
    
    # Deduplication & Connectivity
    "dedup_distance_m":              3.0,     # Centroids closer than this distance are treated as duplicate plots.
    "tile_overlap_dedup_threshold":  0.60,    # Threshold for removing overlapping plots during tile fetching.
    "tile_connect_gap_m":            8.0,     # Max distance from cluster allowed to keep plots fetched from adjacent tiles.
    
    # Pre-processing intersection clearing
    "clear_initial_step":            0.5,     # Starting movement increment for clearing road overlaps.
    "clear_min_step":                0.01,    # Minimum step size allowed before the clearing process terminates.
    "clear_max_move":                25.0,    # Total cumulative distance a plot is allowed to shift from its origin.
    
    # Directional stretch
    "stretch_initial_step":          0.02,    # Starting scale increment applied per direction during stretching.
    "stretch_min_step":              0.001,   # Minimum scale increment threshold to stop the stretching process.
    "min_scale_factor":              0.5,     # Hard lower limit preventing clusters from shrinking below 50% size.
    "max_scale_factor":              2.0,     # Hard upper limit preventing clusters from expanding above 200% size.
    
    # Fine translation refinement
    "refine_initial_step":           2.0,     # Starting step size for fine-tuning plot positions after clearing.
    "refine_min_step":               0.05,    # Minimum step size allowed for final translation refinement.
}


# ─────────────────────────────────────────────────────────────────────────────
#  REFINEMENT — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

VIZ = {
    "satellite_zoom":    18,
    "plot_dpi":         150,
    "stage1_figsize":   (18, 9),
    "stage2_figsize":   (18, 9),
    "bbox_pad_fraction": 0.12,
}

# ─────────────────────────────────────────────────────────────────────────────
#  SAM2 MODEL
# ─────────────────────────────────────────────────────────────────────────────

SAM2_CONFIG     = "sam2_hiera_l.yaml"       # short name, resolved by Hydra from pkg://sam2
SAM2_CHECKPOINT = "models/sam2/sam2_hiera_large.pt"     # filename or absolute path to your .pt file

# ─────────────────────────────────────────────────────────────────────────────
#  SAM2 INFERENCE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

SAM_TILE_ZOOM             = 19     # ESRI zoom level (19 ≈ 0.3 m/px over Pakistan)
SAM_CONTEXT_PAD_FRACTION  = 0.05   # fraction to expand bbox around plot union
SAM_TILE_CACHE_DIR        = DEFAULT_OUTPUT_DIR / "esri_tile_cache"

SAM_ROTATION_THRESHOLD_DEG       = 5.0
SAM_ANGLE_GROUPING_TOLERANCE_DEG = 2.0

# SAM_STAGE2_MAX_BBOX_RATIO controls when to fall back to per-cluster:
#   combined_bbox_area / sum(cluster_bbox_areas)
# If this ratio exceeds the threshold the clusters are too spread out and
# the combined image would be mostly empty satellite pixels.  Set to 0
# to always use per-cluster mode; set very high (e.g. 999) to always combine.
SAM_STAGE2_MAX_BBOX_RATIO = 4.0

# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE STAGE SWITCHES
# ─────────────────────────────────────────────────────────────────────────────

RUN_SAM_STAGE1 = True   # run SAM on stage-1 (global alignment) polygons
RUN_SAM_STAGE2 = True   # run SAM on stage-2 (per-cluster refined) polygons

# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_EVERY_N = 5   # write intermediate Excel every N points

# Add debug_info sheet to Excel with per-point/cluster intermediate stats.
DEBUG_EXCEL = False

# ─────────────────────────────────────────────────────────────────────────────
#  HEIGHT ESTIMATION — T-SwinUNet
# ─────────────────────────────────────────────────────────────────────────────
 
# Set to True to enable the height estimation pipeline.
# Requires CDSE credentials and T-SwinUNet model files.
RUN_HEIGHT_ESTIMATION = True
 
# Years to estimate building height for.  Each year produces three columns
# per plot in the Excel: height_m_{year}, height_class_{year}, height_src_{year}.
# Example: HEIGHT_YEARS = [2022, 2023, 2024]
HEIGHT_YEARS: list = [2024]
 
# Geographic buffer around each GPS point in metres.
# This defines the area from which S1/S2 imagery is downloaded and the
# area the T-SwinUNet model infers over (128×128 px at ~10m/px ≈ 1280m).
# 640m matches the training setup buffer.
HEIGHT_BUFFER_M = 640.0
 
# ── T-SwinUNet model paths ────────────────────────────────────────────────────
 
# Root directory of the T-SwinUNet repository (must contain 'libs' package).
TSWIN_ROOT = BASE_DIR / "tswin_unet"
 
# Path to trained model weights.
TSWIN_MODEL_PATH = BASE_DIR / "models" / "tswinunet" / "model.pth"
 
# Path to model config YAML (exp3.yaml or equivalent).
TSWIN_CONFIG_PATH = BASE_DIR / "tswin_unet" / "configs" / "exp3.yaml"
 

# # ── Sentinel Hub OAuth credentials (used for satellite data download) ────────
# # The satellite downloader uses the Sentinel Hub Process API on CDSE.
# # This API is far more efficient than downloading raw product ZIPs — it
# # returns exactly the pixels you need in a single HTTP request.
# #
# # HOW TO GET THESE (free, 2 minutes):
# #   1. Register at https://dataspace.copernicus.eu  (free)
# #   2. Go to https://shapps.dataspace.copernicus.eu/dashboard/#/
# #   3. User Settings → OAuth Clients → + Add
# #   4. Copy the client_id and client_secret here.
# #
# # Free tier: 30,000 processing units/month (~1,000 point-years of imagery).
# CDSE_SH_CLIENT_ID     = "sh-2ce817e3-5823-485e-94b3-06828c1c8eb5"   # e.g. "sh-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
# CDSE_SH_CLIENT_SECRET = "ma2GYffopvLSpfqp124ZI3gqNoVQhhA0"   # the secret shown once at client creation
 

# ── Google Earth Engine credentials ──────────────────────────────────────────
# GEE is used to download S1 and S2 imagery for height estimation.
# It is mature, reliable, and has no hard monthly processing quota for
# non-commercial research use.
#
# ONE-TIME SETUP (takes 2 minutes):
#   pip install earthengine-api
#   import ee; ee.Authenticate()   # opens browser, paste the code once
#   # credentials cached at ~/.config/earthengine/credentials
#
# If you have a GEE cloud project, set it here (optional):
GEE_PROJECT = "ee-mukskhan9999"   # e.g. "my-gee-project-id" or leave as None
 
# ── Storage controls ──────────────────────────────────────────────────────────

# Minimum number of months (per sensor) that must have real satellite data
# before the T-SwinUNet model is run.  If fewer months are available,
# inference is skipped and height columns are set to None with
# source="insufficient_data".  Prevents feeding the model all-zeros input
# which produces meaningless noise output.
MIN_HEIGHT_MONTHS = 3   # out of 12 months; 3 is a reasonable minimum

# Keep S1/S2 satellite TIF files after inference.
# If False, they are written to a temp dir and deleted immediately.
# Keeping them speeds up re-runs (files are skipped if they already exist).
SAVE_HEIGHT_SAT_DATA = True
 
# Keep the T-SwinUNet _pred.tif height rasters.
# These are small (128×128, ~64 KB each) and useful for visual inspection.
SAVE_HEIGHT_PRED_TIFS = True
 
# Suppress per-month satellite download logs.
HEIGHT_VERBOSE = True
 
# ── Height-to-storey classification thresholds ───────────────────────────────
# Format: list of (min_height_m, max_height_m, label) tuples.
# Ranges are half-open: [min, max).  The last entry catches everything above.
# These are based on ~3m floor-to-ceiling height.  Adjust for your region.
HEIGHT_STORY_THRESHOLDS = [
    (0.0,   3.0,  "0-1 storeys"),
    (3.0,   6.0,  "1-2 storeys"),
    (6.0,   9.0,  "2-3 storeys"),
    (9.0,  12.0,  "3-4 storeys"),
    (12.0, 15.0,  "4-5 storeys"),
    (15.0, 999.0, "5+ storeys"),
]