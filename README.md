# Plot Boundary Pipeline — Integrated Orchestrator

End-to-end pipeline that takes a CSV of GPS coordinates, runs two-stage
boundary refinement, runs SAM2 segmentation, and writes a single clean
Excel file with only the data you need.

```
CSV input
  └─ Orchestrator (orchestrator.py)
       ├─ Stage 1  — global alignment (refinement_utils.stage1)
       ├─ Stage 2  — per-cluster refinement (refinement_utils.stage2)
       └─ SAM      — boundary segmentation (multi_plot_sam.MultiPlotSAM2)
            └─ OutputWriter — single clean Excel file
```

---

## Repository layout

```
plot_pipeline/
├── orchestrator.py          ← main entry point (replaces main.py)
├── pipeline/
│   ├── __init__.py
│   ├── refinement_bridge.py ← wraps stage1/stage2, yields PlotRecord objects
│   ├── sam_bridge.py        ← wraps MultiPlotSAM2 + ESRITileFetcher
│   ├── output_writer.py     ← builds the final Excel (and optional debug Excel)
│   └── utils/
│       ├── __init__.py
│       ├── geo.py           ← UTM ↔ WGS84 helpers, WKT builder
│       └── checkpoint.py    ← per-point resume / skip logic
├── config.py                ← single config file (paths, flags, zoom level …)
└── README.md
```

---

## Quick start

```bash
# Full pipeline — refinement + SAM + final Excel
python orchestrator.py points.csv

# Resume interrupted run (skips points with checkpoint marker)
python orchestrator.py points.csv --skip-existing

# Stage-1 only (no SAM)
python orchestrator.py points.csv --stage1-only

# Stage-2 only (reads existing stage1_plots.geojson)
python orchestrator.py points.csv --stage2-only

# Debug mode — saves extra intermediate data to a second Excel sheet
python orchestrator.py points.csv --debug

# Custom output dir
python orchestrator.py points.csv -o my_results/

# Single point
python orchestrator.py --lat 31.45476821 --lon 74.19205844 --id P0001 --name "Test"
```

---

## Input CSV

Tab- or comma-separated with columns: `point_id`, `name`, `latitude`, `longitude`

```
point_id  name                latitude     longitude
P0001     Untitled Placemark  31.45476821  74.19205844
```

---

## Output

```
<output_dir>/
├── final_dataset.xlsx           ← THE file you care about
├── checkpoint.json              ← resume state (safe to delete after run)
├── pipeline.log
└── <point_id>_<name>/
    ├── metadata.json            ← per-point run stats (always written)
    ├── stage1_plots.geojson
    ├── stage2_plots.geojson
    ├── plots/                   ← diagnostic images (stage1_utm.png, etc.)
    ├── masks/<point_id>/        ← SAM binary masks (.npy, one per plot)
    └── overlays/<point_id>/     ← SAM overlay visualisations
```

### final_dataset.xlsx sheets

| Sheet | Rows | Key columns |
|-------|------|-------------|
| `plots_stage1` | one per stage-1 plot | point_id, plot_index, polygon_wkt, sam_mask_rle, sam_mask_h/w, sam_mask_path, sam_bbox_wkt, sam_score, sam_iou, sam_area_m2, sam_status |
| `plots_stage2` | one per stage-2 plot | + cluster_id, plot_index_in_cluster |
| `points` | one per input point | point_id, name, lat, lon, n_stage1_plots, n_clusters, n_stage2_plots, status |

When `--debug` is set a fourth sheet `debug_info` is appended with all
intermediate stats (losses, transform params, cluster-level details).

---

## Adding a third pipeline (e.g. height data)

1. Add your processor class (e.g. `HeightPipeline`) to `pipeline/height_bridge.py`.
2. Call it from `orchestrator.py` after the SAM step — it receives the same
   `PointResult` object and can add fields to it.
3. Register any new output columns in `pipeline/output_writer.py`.

The `PointResult` dataclass in `pipeline/refinement_bridge.py` is the shared
data carrier between all stages.