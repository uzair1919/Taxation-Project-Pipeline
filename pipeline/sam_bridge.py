"""
pipeline/sam_bridge.py
=======================
Runs SAM2 segmentation on PlotRecord objects and attaches results back.

Changes in this version
-----------------------
* Uses the new multi_plot_sam.MultiPlotSAM2 which loads via from_pretrained.
* Forwards verbose, save_masks, save_overlays from config.
* Context images are saved only when SAVE_SAM_CONTEXT_IMAGES=True.
* No config.SAM2_CHECKPOINT / SAM2_CONFIG references anywhere.
"""

from __future__ import annotations

import logging
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM result dataclass  — attached to each PlotRecord after segmentation
# ---------------------------------------------------------------------------

@dataclass
class SamResult:
    """SAM2 output for one plot polygon."""

    status:            str    # "success" | "failed" | "skipped"
    error:             str    # empty string on success

    mask_geo_wkt:      str    # WGS84 POLYGON WKT of the actual mask boundary
    mask_path:         str    # absolute path to .npy; empty if save_masks=False

    sam_bbox_wkt:      str    # WGS84 POLYGON WKT of SAM's aligned bbox
    sam_score:         float
    sam_iou:           float
    sam_area_m2:       float
    rotation_angle_deg: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wkt_to_bbox_union(wkt_list: list):
    """
    Return the tight (west, south, east, north) union bbox of a list of WKT
    polygon strings.  Returns None if no valid polygons are found.
    Used only for the combined-vs-per-cluster area ratio check.
    """
    from pipeline.utils.geo import parse_wkt_vertices
    all_lons, all_lats = [], []
    for wkt in wkt_list:
        verts = parse_wkt_vertices(wkt)
        if verts:
            all_lons.extend(v[0] for v in verts)
            all_lats.extend(v[1] for v in verts)
    if not all_lons:
        return None
    return (min(all_lons), min(all_lats), max(all_lons), max(all_lats))


def _build_prompts(
    records: list,
    ctx_west: float, ctx_south: float,
    ctx_east: float, ctx_north: float,
    img_w: int, img_h: int,
    stage: str,
) -> list:
    """Convert PlotRecord WKT polygons to SAM prompt dicts with pixel coords."""
    from pipeline.utils.geo import parse_wkt_vertices, geo_to_pixel

    prompts = []
    for rec in records:
        vertices_geo = parse_wkt_vertices(rec.polygon_wkt)
        if not vertices_geo:
            logger.warning(
                f"[{rec.point_id}/{stage}] plot_index={rec.plot_index} "
                f"cluster={rec.cluster_id}: invalid WKT — skipping"
            )
            continue

        vertices_px = [
            list(geo_to_pixel(lon, lat,
                              ctx_west, ctx_south, ctx_east, ctx_north,
                              img_w, img_h))
            for lon, lat in vertices_geo
        ]

        if stage.startswith("stage2"):
            plot_id_str = (
                f"{rec.point_id}_{stage}_c{rec.cluster_id}"
                f"_plot{rec.plot_index:03d}"
            )
        else:
            plot_id_str = f"{rec.point_id}_{stage}_plot{rec.plot_index:03d}"

        prompts.append({
            "plot_id":      plot_id_str,
            "plot_index":   rec.plot_index,
            "cluster_id":   rec.cluster_id,
            "point_id":     rec.point_id,
            "type":         "polygon",
            "polygon_px":   vertices_px,
            "polygon_geo":  vertices_geo,
            "_record_ref":  rec,       # back-reference, not passed to SAM
        })
    return prompts


def _attach_results(prompts: list, sam_results: list) -> None:
    """Write SamResult back onto each PlotRecord referenced in prompts."""
    from pipeline.utils.geo import bbox_corners_to_wkt

    if len(prompts) != len(sam_results):
        logger.warning(
            f"prompts/results length mismatch "
            f"({len(prompts)} vs {len(sam_results)}) — "
            "some records will be missing SAM results"
        )

    for prompt, res in zip(prompts, sam_results):
        rec = prompt["_record_ref"]

        if res.get("status") != "success":
            rec.sam_result = SamResult(
                status="failed",
                error=str(res.get("error", "unknown SAM error")),
                mask_geo_wkt="", mask_path="",
                sam_bbox_wkt="", sam_score=0.0, sam_iou=0.0,
                sam_area_m2=0.0, rotation_angle_deg=0.0,
            )
            continue

        aligned_bbox = res.get("aligned_bbox_geo")
        rec.sam_result = SamResult(
            status             = "success",
            error              = "",
            mask_geo_wkt       = res.get("mask_geo_wkt", ""),
            mask_path          = str(res.get("mask_path", "")),
            sam_bbox_wkt       = bbox_corners_to_wkt(aligned_bbox) if aligned_bbox else "",
            sam_score          = float(res.get("score", 0.0)),
            sam_iou            = float(res.get("iou", 0.0)),
            sam_area_m2        = float(res.get("area_m2", 0.0)),
            rotation_angle_deg = float(res.get("rotation_angle_deg", 0.0)),
        )


def _mark_skipped(records: list, reason: str = "context image unavailable") -> None:
    for rec in records:
        rec.sam_result = SamResult(
            status="skipped", error=reason,
            mask_geo_wkt="", mask_path="",
            sam_bbox_wkt="", sam_score=0.0, sam_iou=0.0,
            sam_area_m2=0.0, rotation_angle_deg=0.0,
        )


# ---------------------------------------------------------------------------
# Public SamRunner class
# ---------------------------------------------------------------------------

class SamRunner:
    """
    Loads MultiPlotSAM2 once and runs it per point.

    Parameters
    ----------
    cfg : unified config module
    """

    def __init__(self, cfg) -> None:
        self.cfg   = cfg
        self._sam  = None   # lazy-loaded on first call to run_on_point
        self._esri = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_on_point(self, point_result, out_dir: Path) -> None:
        """
        Attach SamResult to every PlotRecord in point_result.
        Results written in-place; nothing returned.
        """
        out_dir = Path(out_dir)
        self._ensure_loaded()

        pid = point_result.point_id

        if getattr(self.cfg, "RUN_SAM_STAGE1", True) and point_result.stage1_plots:
            logger.info(
                f"[{pid}] SAM stage1 — {len(point_result.stage1_plots)} plots"
            )
            self._run_flat(point_result.stage1_plots, "stage1", out_dir, pid)

        if getattr(self.cfg, "RUN_SAM_STAGE2", True) and point_result.stage2_plots:
            logger.info(
                f"[{pid}] SAM stage2 — {len(point_result.stage2_plots)} plots "
                f"({point_result.n_clusters} cluster(s))"
            )
            self._run_stage2(point_result.stage2_plots, out_dir, pid)

    def close(self) -> None:
        """Release GPU resources."""
        try:
            if self._sam is not None:
                del self._sam
                self._sam = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._sam is not None:
            return

        # Import from the new multi_plot_sam.py (same directory as this package,
        # or already on sys.path via SAM2_ROOT in config).
        from sam_utils.multi_plot_sam import MultiPlotSAM2
        from sam_utils.esri_tile_fetcher import ESRITileFetcher

        sam2_config     = str(getattr(self.cfg, "SAM2_CONFIG",     "sam2_hiera_l.yaml"))
        sam2_checkpoint = str(getattr(self.cfg, "SAM2_CHECKPOINT", "sam2_hiera_large.pt"))
        verbose       = getattr(self.cfg, "SAM_VERBOSE",      True)
        save_masks    = getattr(self.cfg, "SAVE_SAM_MASKS",    True)
        save_overlays = getattr(self.cfg, "SAVE_SAM_OVERLAYS", True)

        logger.info(f"Loading SAM2 (config={sam2_config}) …")
        self._sam = MultiPlotSAM2(
            sam2_config     = sam2_config,
            sam2_checkpoint = sam2_checkpoint,
            verbose         = verbose,
            save_masks      = save_masks,
            save_overlays   = save_overlays,
        )

        cache = getattr(self.cfg, "SAM_TILE_CACHE_DIR", None)
        self._esri = ESRITileFetcher(
            cache_dir = Path(cache) if cache else None
        )
        logger.info("ESRI tile fetcher ready.")

    def _fetch_context(
        self,
        records: list,
        stage: str,
        point_id: str,
        out_dir: Path,
    ):
        """
        Compute union bbox of all plot WKTs, fetch satellite image.
        Returns (PIL.Image, (west, south, east, north)) or (None, None).
        """
        from pipeline.utils.geo import compute_context_bbox

        pad  = float(getattr(self.cfg, "SAM_CONTEXT_PAD_FRACTION", 0.05))
        zoom = int(getattr(self.cfg, "SAM_TILE_ZOOM", 19))
        save_ctx = getattr(self.cfg, "SAVE_SAM_CONTEXT_IMAGES", True)

        bbox = compute_context_bbox([r.polygon_wkt for r in records], pad_fraction=pad)
        if bbox is None:
            logger.warning(f"[{point_id}/{stage}] No valid WKT polygons — skipping")
            return None, None

        ctx_west, ctx_south, ctx_east, ctx_north = bbox

        # Only create file on disk when requested
        if save_ctx:
            ctx_img_path = out_dir / "context_images" / f"{point_id}_{stage}_context.png"
            ctx_img_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            ctx_img_path = None

        try:
            ctx_img = self._esri.fetch_bbox(
                west=ctx_west, south=ctx_south,
                east=ctx_east, north=ctx_north,
                zoom=zoom,
                output_path=ctx_img_path,    # None → don't save
            )
            logger.info(
                f"[{point_id}/{stage}] Context image: "
                f"{ctx_img.width}×{ctx_img.height} px  zoom={zoom}"
            )
            return ctx_img, (ctx_west, ctx_south, ctx_east, ctx_north)
        except Exception as exc:
            logger.error(f"[{point_id}/{stage}] ESRI fetch failed: {exc}")
            logger.error(traceback.format_exc())
            return None, None

    def _run_flat(
        self,
        records: list,
        stage: str,
        out_dir: Path,
        point_id: str,
    ) -> None:
        """Run SAM on a flat list of records (stage1 or one cluster of stage2)."""
        ctx_img, ctx_tuple = self._fetch_context(records, stage, point_id, out_dir)
        if ctx_img is None:
            _mark_skipped(records)
            return

        ctx_west, ctx_south, ctx_east, ctx_north = ctx_tuple
        prompts = _build_prompts(
            records, ctx_west, ctx_south, ctx_east, ctx_north,
            ctx_img.width, ctx_img.height, stage,
        )
        if not prompts:
            _mark_skipped(records, "no valid WKT polygons")
            return

        try:
            sam_results = self._sam.segment_multiple_plots(
                point_id      = point_id,
                stage         = stage,
                context_image = ctx_img,
                context_bbox  = (ctx_west, ctx_south, ctx_east, ctx_north),
                prompts       = prompts,
                output_dir    = out_dir,
            )
            _attach_results(prompts, sam_results)
        except Exception as exc:
            logger.error(f"[{point_id}/{stage}] SAM inference failed: {exc}")
            logger.error(traceback.format_exc())
            _mark_skipped(records, f"SAM inference error: {exc}")

    def _run_stage2(
        self,
        records: list,
        out_dir: Path,
        point_id: str,
    ) -> None:
        """
        Run SAM on all stage-2 plots using a single combined context image.

        All cluster records are gathered into one bbox, a single satellite
        image is fetched and set_image() is called once per rotation group
        across ALL clusters — instead of once per cluster.  This eliminates
        N-1 tile fetches and N-1 SAM encoder passes for N clusters.

        Cluster identity is preserved: each prompt carries cluster_id and
        results are matched back to PlotRecords by position.

        Falls back to per-cluster processing if the combined bbox is too
        large relative to the sum of individual cluster bboxes (controlled
        by SAM_STAGE2_MAX_BBOX_RATIO in config, default 4.0).  A ratio
        above the threshold means clusters are far apart and the combined
        image would waste most pixels on empty space.
        """
        from collections import defaultdict
        from pipeline.utils.geo import compute_context_bbox, wkt_to_bbox

        if not records:
            return

        n_clusters = len({r.cluster_id for r in records})

        # ── Decide: combined vs per-cluster ───────────────────────────────
        # Compare area of the combined bbox vs sum of per-cluster bboxes.
        # If combined area / sum(cluster areas) > threshold, clusters are
        # too spread out and per-cluster is more efficient.
        max_ratio = float(getattr(self.cfg, "SAM_STAGE2_MAX_BBOX_RATIO", 4.0))
        use_combined = True

        if n_clusters > 1 and max_ratio > 0:
            clusters_map: dict = defaultdict(list)
            for rec in records:
                clusters_map[rec.cluster_id].append(rec.polygon_wkt)

            combined_bbox = wkt_to_bbox_union([r.polygon_wkt for r in records])
            cluster_area_sum = 0.0
            for wkts in clusters_map.values():
                bb = wkt_to_bbox_union(wkts)
                if bb:
                    cluster_area_sum += (bb[2] - bb[0]) * (bb[3] - bb[1])

            if combined_bbox and cluster_area_sum > 0:
                cb = combined_bbox
                combined_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
                ratio = combined_area / cluster_area_sum
                if ratio > max_ratio:
                    logger.info(
                        f"[{point_id}] SAM stage2 — clusters spread out "
                        f"(bbox ratio {ratio:.1f} > {max_ratio}) → per-cluster mode"
                    )
                    use_combined = False
                else:
                    logger.info(
                        f"[{point_id}] SAM stage2 — combined mode "
                        f"({n_clusters} clusters, bbox ratio {ratio:.1f})"
                    )

        if use_combined:
            # All clusters → one context image → one set_image per rotation group
            self._run_flat(records, "stage2", out_dir, point_id)
        else:
            # Fallback: per-cluster (original behaviour)
            clusters_by_id: dict = defaultdict(list)
            for rec in records:
                clusters_by_id[rec.cluster_id].append(rec)
            for cluster_id, cluster_records in sorted(clusters_by_id.items()):
                logger.info(
                    f"[{point_id}] SAM cluster {cluster_id} — "
                    f"{len(cluster_records)} plots"
                )
                self._run_flat(cluster_records, f"stage2_c{cluster_id}", out_dir, point_id)