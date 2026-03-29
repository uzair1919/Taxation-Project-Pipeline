"""
orchestrator.py — Integrated Plot Boundary Pipeline
=====================================================
Entry point for the combined refinement + SAM2 segmentation pipeline.

All sub-pipeline verbosity and file-save behaviour is controlled through
config.py flags and the CLI arguments below.

CLI
---
    python orchestrator.py points.csv
    python orchestrator.py points.csv --skip-existing -o results/
    python orchestrator.py points.csv --stage1-only
    python orchestrator.py points.csv --stage2-only
    python orchestrator.py points.csv --debug
    python orchestrator.py points.csv --quiet
    python orchestrator.py points.csv --no-plots --no-geojson --no-masks
    python orchestrator.py points.csv --start-from P0005
    python orchestrator.py --lat 31.45 --lon 74.19 --id P0001 --name "Test"
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import traceback
from pathlib import Path

import pandas as pd

# ── Bootstrap sys.path ───────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import config as cfg

for _p in (cfg.REFINEMENT_ROOT, cfg.SAM2_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# ── Logging ───────────────────────────────────────────────────────────────────
class _RootInfoFilter(logging.Filter):
    """Block specific root-logger INFO calls that third-party libs emit directly
    via logging.info() instead of a named logger (so setLevel on a named logger
    has no effect on them).
    """
    _BLOCKED = frozenset({
        "For numpy array image, we assume (HxWxC) format",
        "Computing image embeddings for the provided image...",
        "Computing image embeddings for the provided images...",
        "Image embeddings computed.",
    })
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != "root" or record.getMessage() not in self._BLOCKED


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt      = "%(asctime)s [%(levelname)-8s] %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "pipeline.log", mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    # Block root-level INFO noise from third-party libs that bypass named loggers
    _f = _RootInfoFilter()
    for h in logging.root.handlers:
        h.addFilter(_f)
    # Libraries that are too chatty at INFO/WARNING level
    for lib in ("httpx", "urllib3", "PIL", "shapely", "fiona",
                "pyproj", "huggingface_hub"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    # rasterio/GDAL emit repetitive TIFF photometric warnings — suppress to ERROR
    for lib in ("rasterio", "rasterio._env", "rasterio.env"):
        logging.getLogger(lib).setLevel(logging.ERROR)
    # SAM2 named-logger messages (belt-and-suspenders alongside the root filter)
    for lib in ("sam2", "sam2.sam2_image_predictor", "sam2.build_sam",
                "sam2.modeling", "sam2.utils"):
        logging.getLogger(lib).setLevel(logging.WARNING)


logger = logging.getLogger("orchestrator")

# ── Late imports (after sys.path is ready) ────────────────────────────────────
from pipeline.refinement_bridge import run_refinement, PointResult
from pipeline.sam_bridge import SamRunner
from pipeline.output_writer import write_final_excel
from pipeline.utils.checkpoint import CheckpointManager
from pipeline.height_bridge import HeightRunner


# =============================================================================
# Helpers
# =============================================================================

def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", str(name))[:40].strip("_") or "unnamed"


def _point_dir(output_dir: Path, point_id: str, name: str) -> Path:
    return output_dir / f"{point_id}_{_safe_name(name)}"


def _load_points(csv_path: str) -> pd.DataFrame:
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(csv_path, sep=sep, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            if {"point_id", "latitude", "longitude"}.issubset(df.columns):
                df["latitude"]  = df["latitude"].astype(float)
                df["longitude"] = df["longitude"].astype(float)
                df["name"]      = df.get("name", df["point_id"]).fillna(df["point_id"])
                return df.reset_index(drop=True)
        except Exception:
            continue
    raise ValueError(
        f"Cannot parse {csv_path!r}. "
        "Expected columns: point_id, name, latitude, longitude"
    )


# =============================================================================
# Single-point processor
# =============================================================================

def process_point(
    lat:           float,
    lon:           float,
    point_id:      str,
    name:          str,
    out_dir:       Path,
    sam_runner:    SamRunner,
    height_runner: HeightRunner | None = None,
    run_s1:        bool = True,
    run_s2:        bool = True,
    debug_mode:    bool = False,
    verbose:       bool = True,
) -> PointResult:
    """
    Run refinement + SAM (+ optional height estimation) for one point.
    Never raises.
    """
    import time

    run_sam_s1 = getattr(cfg, "RUN_SAM_STAGE1", True)
    run_sam_s2 = getattr(cfg, "RUN_SAM_STAGE2", True)

    stages_desc = (
        (["Stage-1"] if run_s1 else [])
        + (["Stage-2"] if run_s2 else [])
        + (["SAM"] if (run_sam_s1 or run_sam_s2) else [])
    )
    logger.info(
        f"[{point_id}] ({lat:.6f}, {lon:.6f})  pipeline: "
        + " → ".join(stages_desc)
    )

    # ── Stage 1: tile fetch + edge detection + global alignment ──────────
    if run_s1:
        logger.info(f"[{point_id}] Stage 1 — fetching tiles, detecting edges, global alignment …")

    t_ref = time.time()
    point_result = run_refinement(
        lat=lat, lon=lon,
        point_id=point_id, name=name,
        out_dir=str(out_dir),
        run_s1=run_s1, run_s2=run_s2,
        cfg=cfg,
        verbose=verbose,
        debug_mode=debug_mode,
    )
    ref_elapsed = time.time() - t_ref

    if point_result.status == "error":
        logger.error(
            f"[{point_id}] Refinement failed after {ref_elapsed:.0f}s "
            f"— {point_result.error}"
        )
        return point_result

    if run_s1:
        logger.info(
            f"[{point_id}] Stage 1 done — "
            f"{point_result.n_stage1_plots} plots"
        )
    if run_s2:
        logger.info(
            f"[{point_id}] Stage 2 done — "
            f"{point_result.n_stage2_plots} plots "
            f"in {point_result.n_clusters} cluster(s)"
        )
    logger.info(f"[{point_id}] Refinement complete ({ref_elapsed:.0f}s)")

    # ── SAM segmentation ──────────────────────────────────────────────────
    sam_stage_parts = []
    if run_sam_s1 and point_result.stage1_plots:
        sam_stage_parts.append(f"stage-1 ({point_result.n_stage1_plots} plots)")
    if run_sam_s2 and point_result.stage2_plots:
        sam_stage_parts.append(
            f"stage-2 ({point_result.n_stage2_plots} plots, "
            f"{point_result.n_clusters} cluster(s))"
        )

    if sam_stage_parts:
        mode = "combined" if getattr(cfg, "SAM_STAGE2_MAX_BBOX_RATIO", 4.0) > 0 else "per-cluster"
        logger.info(
            f"[{point_id}] SAM — segmenting {' + '.join(sam_stage_parts)} "
            f"(stage2 mode: {mode}) …"
        )

    t_sam = time.time()
    try:
        sam_runner.run_on_point(point_result, out_dir)
        sam_elapsed = time.time() - t_sam

        def _sam_counts(records):
            ok  = sum(1 for r in records
                      if getattr(getattr(r, "sam_result", None), "status", "") == "success")
            return ok, len(records)

        s1_ok, s1_tot = _sam_counts(point_result.stage1_plots)
        s2_ok, s2_tot = _sam_counts(point_result.stage2_plots)
        logger.info(
            f"[{point_id}] SAM done ({sam_elapsed:.0f}s) — "
            f"stage-1: {s1_ok}/{s1_tot} masks OK  |  "
            f"stage-2: {s2_ok}/{s2_tot} masks OK"
        )
    except Exception as exc:
        logger.error(f"[{point_id}] SAM failed: {exc}")
        logger.error(traceback.format_exc())
        from pipeline.sam_bridge import _mark_skipped
        _mark_skipped(point_result.stage1_plots, f"SAM crash: {exc}")
        _mark_skipped(point_result.stage2_plots, f"SAM crash: {exc}")

    # ── Height estimation ────────────────────────────────────────────────
    if height_runner is not None and getattr(cfg, "RUN_HEIGHT_ESTIMATION", False):
        years = list(getattr(cfg, "HEIGHT_YEARS", []))
        if years:
            logger.info(
                f"[{point_id}] Height estimation — "
                f"{len(years)} year(s): {years} …"
            )
            t_height = time.time()
            try:
                height_runner.run_on_point(
                    point_result = point_result,
                    lat          = lat,
                    lon          = lon,
                    out_dir      = out_dir,
                )
                h_elapsed = time.time() - t_height
                # Count plots with valid height for at least the first year
                first_year = years[0]
                n_h_ok = sum(
                    1 for r in (point_result.stage1_plots + point_result.stage2_plots)
                    if getattr(r, "height_results", {}).get(first_year) and
                       getattr(r, "height_results", {}).get(first_year).height_m is not None
                )
                n_total_plots = len(point_result.stage1_plots) + len(point_result.stage2_plots)
                logger.info(
                    f"[{point_id}] Height done ({h_elapsed:.0f}s) — "
                    f"{n_h_ok}/{n_total_plots} plots with valid height"
                )
            except Exception as exc:
                logger.error(f"[{point_id}] Height estimation failed: {exc}")
                logger.error(traceback.format_exc())

    total_elapsed = time.time() - t_sam + ref_elapsed
    logger.info(f"[{point_id}] ✓ Complete ({total_elapsed:.0f}s total)")
    return point_result


# =============================================================================
# Excel helpers
# =============================================================================

def _write_excel_safe(
    results: list, path: Path, debug_mode: bool, height_years: list | None = None
) -> bool:
    """Write Excel. Returns True on success, False on failure. Never raises."""
    if not results:
        return True
    try:
        write_final_excel(results, path, debug_mode=debug_mode, height_years=height_years)
        return True
    except Exception as exc:
        logger.error(f"Excel write failed: {exc}")
        logger.error(traceback.format_exc())
        # Attempt to write to a fallback path so partial results are never lost
        try:
            fallback = path.with_name(path.stem + "_recovery.xlsx")
            write_final_excel(results, fallback, debug_mode=False, height_years=height_years)
            logger.warning(f"Recovery Excel written to: {fallback}")
        except Exception:
            pass
        return False


# =============================================================================
# Batch runner
# =============================================================================

def run_batch(
    points_csv:       str,
    output_dir:       Path,
    skip_existing:    bool = False,
    run_s1:           bool = True,
    run_s2:           bool = True,
    start_from:       str | None = None,
    debug_mode:       bool = False,
    verbose:          bool = True,
    reset_checkpoint: bool = False,
) -> None:

    _setup_logging(output_dir)

    # ── Load input CSV ────────────────────────────────────────────────────
    try:
        df = _load_points(points_csv)
    except Exception as exc:
        logger.error(f"Cannot load points CSV '{points_csv}': {exc}")
        sys.exit(1)
    n_total = len(df)

    checkpoint = CheckpointManager(output_dir)
    if reset_checkpoint:
        checkpoint.reset()

    excel_path     = output_dir / getattr(cfg, "FINAL_EXCEL_NAME", "final_dataset.xlsx")
    height_years_l = list(getattr(cfg, "HEIGHT_YEARS", []))

    logger.info("=" * 70)
    logger.info("  Integrated Plot Boundary Pipeline")
    logger.info(f"  Points CSV    : {points_csv}")
    logger.info(f"  Output dir    : {output_dir.resolve()}")
    logger.info(f"  Total points  : {n_total}")
    logger.info(f"  Stage 1       : {run_s1}")
    logger.info(f"  Stage 2       : {run_s2}")
    logger.info(f"  SAM stage 1   : {getattr(cfg, 'RUN_SAM_STAGE1', True)}")
    logger.info(f"  SAM stage 2   : {getattr(cfg, 'RUN_SAM_STAGE2', True)}")
    logger.info(f"  Save plots    : {getattr(cfg, 'SAVE_REFINEMENT_PLOTS', True)}")
    logger.info(f"  Save GeoJSON  : {getattr(cfg, 'SAVE_GEOJSON', True)}")
    logger.info(f"  Save masks    : {getattr(cfg, 'SAVE_SAM_MASKS', True)}")
    logger.info(f"  Save overlays : {getattr(cfg, 'SAVE_SAM_OVERLAYS', True)}")
    logger.info(f"  Verbose       : {verbose}")
    logger.info(f"  Debug mode    : {debug_mode}")
    logger.info(f"  Skip existing : {skip_existing}")
    if start_from:
        logger.info(f"  Start from    : {start_from}")
    logger.info("=" * 70)

    if start_from:
        mask = df["point_id"] == start_from
        if mask.any():
            df = df[df.index >= df[mask].index[0]].reset_index(drop=True)
        else:
            logger.warning(f"--start-from '{start_from}' not found — processing all")

    all_results: list[PointResult] = []
    n_ok = n_err = n_skip = 0

    # Keep runners as None until initialised so the finally block is safe
    # even if initialisation crashes before the try block is fully entered.
    sam_runner    = None
    height_runner = None

    try:
        # ── Initialise heavy components inside try so finally always saves ──
        try:
            sam_runner = SamRunner(cfg)
        except Exception as exc:
            logger.error(f"SAM runner failed to initialise: {exc}")
            logger.error(traceback.format_exc())
            logger.error("Cannot continue without SAM — aborting.")
            return   # finally block still runs; all_results is empty so no write

        if getattr(cfg, "RUN_HEIGHT_ESTIMATION", False) and height_years_l:
            logger.info(f"Height estimation enabled for years: {height_years_l}")
            try:
                height_runner = HeightRunner(cfg)
            except Exception as exc:
                logger.error(f"Height runner failed to initialise: {exc}")
                logger.error(traceback.format_exc())
                logger.warning("Continuing WITHOUT height estimation for this run.")
                height_runner = None

        # ── Register SIGTERM handler so Linux kill/OOM still saves Excel ────
        import signal as _signal
        def _sigterm_handler(signum, frame):
            logger.warning("SIGTERM received — saving Excel and exiting …")
            if all_results:
                _write_excel_safe(all_results, excel_path, debug_mode, height_years_l)
            sys.exit(0)
        try:
            _signal.signal(_signal.SIGTERM, _sigterm_handler)
        except Exception:
            pass   # not available on all platforms; non-fatal

        import time as _time
        for i, row in df.iterrows():
            # ── Parse row — skip malformed rows rather than crashing ─────
            try:
                point_id = str(row["point_id"]).strip()
                name     = str(row.get("name", row["point_id"])).strip()
                lat      = float(row["latitude"])
                lon      = float(row["longitude"])
            except Exception as exc:
                logger.error(f"Row {i}: cannot parse — {exc} — skipping")
                n_err += 1
                continue

            out_dir_point = _point_dir(output_dir, point_id, name)
            remaining     = n_total - (i + 1)

            logger.info("")
            logger.info(f"─── Point {i+1}/{n_total}  {point_id} — {name} ───")

            if skip_existing and checkpoint.is_done(point_id):
                logger.info(f"[{point_id}] Already completed — skipping  ({remaining} remaining)")
                n_skip += 1
                continue

            t_start = _time.time()
            point_result = process_point(
                lat=lat, lon=lon,
                point_id=point_id, name=name,
                out_dir=out_dir_point,
                sam_runner=sam_runner,
                height_runner=height_runner,
                run_s1=run_s1, run_s2=run_s2,
                debug_mode=debug_mode,
                verbose=verbose,
            )
            elapsed = _time.time() - t_start

            all_results.append(point_result)

            if point_result.status == "ok":
                n_ok += 1
                checkpoint.mark_done(point_id, {
                    "status":         "ok",
                    "n_stage1_plots": point_result.n_stage1_plots,
                    "n_stage2_plots": point_result.n_stage2_plots,
                    "n_clusters":     point_result.n_clusters,
                })
                logger.info(
                    f"[{point_id}] ✓ OK  ({elapsed:.0f}s)  —  "
                    f"batch progress: {n_ok} ok, {n_err} failed, {n_skip} skipped"
                    f"  |  {remaining} remaining"
                )
            else:
                n_err += 1
                checkpoint.mark_failed(point_id, point_result.error or "unknown")
                logger.warning(
                    f"[{point_id}] ✗ FAILED  ({elapsed:.0f}s)  —  "
                    f"batch progress: {n_ok} ok, {n_err} failed, {n_skip} skipped"
                    f"  |  {remaining} remaining"
                )

            # Save Excel after every single point so a crash never loses completed work
            _write_excel_safe(all_results, excel_path, debug_mode, height_years_l)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C) — saving progress and exiting …")
    except Exception as exc:
        logger.error(f"Unexpected batch-level error: {exc}")
        logger.error(traceback.format_exc())
    finally:
        if sam_runner is not None:
            try:
                sam_runner.close()
            except Exception:
                pass
        if height_runner is not None:
            try:
                height_runner.close()
            except Exception:
                pass
        # Final write guarantees the last point is saved even if the loop
        # was interrupted between the per-point write and the next iteration.
        if all_results:
            _write_excel_safe(all_results, excel_path, debug_mode, height_years_l)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  Done.  OK={n_ok}  Errors={n_err}  Skipped={n_skip}")
    logger.info(f"  Excel → {excel_path.resolve()}")
    logger.info("=" * 70)


# =============================================================================
# Single-point convenience wrapper
# =============================================================================

def run_single(
    lat:        float,
    lon:        float,
    point_id:   str,
    name:       str,
    output_dir: Path,
    run_s1:     bool = True,
    run_s2:     bool = True,
    debug_mode: bool = False,
    verbose:    bool = True,
) -> None:
    _setup_logging(output_dir)
    sam_runner     = SamRunner(cfg)
    height_years_l = list(getattr(cfg, "HEIGHT_YEARS", []))
    height_runner  = None
    if getattr(cfg, "RUN_HEIGHT_ESTIMATION", False) and height_years_l:
        height_runner = HeightRunner(cfg)

    out_dir_point = _point_dir(output_dir, point_id, name)
    excel_path    = output_dir / getattr(cfg, "FINAL_EXCEL_NAME", "final_dataset.xlsx")
    result        = None
    try:
        result = process_point(
            lat=lat, lon=lon,
            point_id=point_id, name=name,
            out_dir=out_dir_point,
            sam_runner=sam_runner,
            height_runner=height_runner,
            run_s1=run_s1, run_s2=run_s2,
            debug_mode=debug_mode,
            verbose=verbose,
        )
    finally:
        try:
            sam_runner.close()
        except Exception:
            pass
        if height_runner is not None:
            try:
                height_runner.close()
            except Exception:
                pass
        if result is not None:
            _write_excel_safe([result], excel_path, debug_mode, height_years_l)
            logger.info(f"Done. Excel → {excel_path.resolve()}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        prog="orchestrator.py",
        description="Integrated Plot Boundary Pipeline (refinement + SAM2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py data/coordinates_data1_points.csv
  python orchestrator.py points.csv --skip-existing -o results/
  python orchestrator.py points.csv --stage1-only
  python orchestrator.py points.csv --stage2-only
  python orchestrator.py points.csv --debug
  python orchestrator.py points.csv --quiet
  python orchestrator.py points.csv --no-plots --no-geojson --no-masks
  python orchestrator.py points.csv --start-from P0005
  python orchestrator.py --lat 31.39577858 --lon 74.15519505 --id P0090 --name "Test"
        """,	
    )

    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("points_csv", nargs="?",
                     help="CSV/TSV: point_id, name, latitude, longitude")
    inp.add_argument("--lat", type=float, help="Latitude (single-point mode)")

    p.add_argument("--lon",  type=float)
    p.add_argument("--id",   type=str, default="P0000")
    p.add_argument("--name", type=str, default="point")

    p.add_argument("--output-dir", "-o",
                   default=str(cfg.DEFAULT_OUTPUT_DIR), metavar="DIR")
    p.add_argument("--skip-existing", "-s", action="store_true")
    p.add_argument("--start-from", metavar="POINT_ID", default=None)
    p.add_argument("--reset-checkpoint", action="store_true")

    stg = p.add_mutually_exclusive_group()
    stg.add_argument("--stage1-only", action="store_true",
                     help="Run only stage-1 refinement (no stage-2, no SAM)")
    stg.add_argument("--stage2-only", action="store_true",
                     help="Run only stage-2 + SAM (reads existing stage1_plots.geojson)")

    p.add_argument("--debug",   action="store_true",
                   help="Add debug_info sheet to Excel")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress sub-pipeline verbose output")

    # Output size controls (override config.py flags for this run)
    p.add_argument("--no-plots",    action="store_true",
                   help="Do not save refinement diagnostic plots")
    p.add_argument("--no-geojson",  action="store_true",
                   help="Do not save per-point GeoJSON files")
    p.add_argument("--no-masks",    action="store_true",
                   help="Do not save SAM .npy mask files")
    p.add_argument("--no-overlays", action="store_true",
                   help="Do not save SAM overlay PNGs")
    p.add_argument("--no-context",  action="store_true",
                   help="Do not save SAM context images")

    # Height estimation overrides
    p.add_argument("--no-height",   action="store_true",
                   help="Disable height estimation even if RUN_HEIGHT_ESTIMATION=True in config")
    p.add_argument("--years", nargs="+", type=int, default=None,
                   metavar="YEAR",
                   help="Override HEIGHT_YEARS for this run, e.g. --years 2022 2023 2024")

    args = p.parse_args()

    # Apply CLI overrides to config flags
    if args.no_height:   cfg.RUN_HEIGHT_ESTIMATION   = False
    if args.years:       cfg.HEIGHT_YEARS             = args.years
    if args.no_plots:    cfg.SAVE_REFINEMENT_PLOTS   = False
    if args.no_geojson:  cfg.SAVE_GEOJSON            = False
    if args.no_masks:    cfg.SAVE_SAM_MASKS          = False
    if args.no_overlays: cfg.SAVE_SAM_OVERLAYS       = False
    if args.no_context:  cfg.SAVE_SAM_CONTEXT_IMAGES = False
    if args.quiet:
        cfg.REFINEMENT_VERBOSE = False
        cfg.SAM_VERBOSE        = False

    run_s1  = not args.stage2_only
    run_s2  = not args.stage1_only
    verbose = not args.quiet
    out_dir = Path(args.output_dir)

    if args.lat is not None:
        if args.lon is None:
            p.error("--lat requires --lon")
        run_single(
            lat=args.lat, lon=args.lon,
            point_id=args.id, name=args.name,
            output_dir=out_dir,
            run_s1=run_s1, run_s2=run_s2,
            debug_mode=args.debug,
            verbose=verbose,
        )
    else:
        if not os.path.exists(args.points_csv):
            print(f"Error: file not found: {args.points_csv!r}", file=sys.stderr)
            sys.exit(1)
        run_batch(
            points_csv       = args.points_csv,
            output_dir       = out_dir,
            skip_existing    = args.skip_existing,
            run_s1           = run_s1,
            run_s2           = run_s2,
            start_from       = args.start_from,
            debug_mode       = args.debug,
            verbose          = verbose,
            reset_checkpoint = args.reset_checkpoint,
        )


if __name__ == "__main__":
    main()