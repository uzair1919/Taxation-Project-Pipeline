"""
multi_plot_sam.py
=================
SAM2-based plot segmentation.

Loads SAM2 exactly as the original pipeline did — via build_sam2() with
a short config filename and a checkpoint path, both resolved by the sam2
library installed through pip.

Added vs the original
---------------------
* verbose=False  — silences INFO-level logs; errors always surface.
* save_masks=False  — skips writing .npy files; mask_path becomes "".
* save_overlays=False  — skips writing overlay PNGs.
* Everything else (angle grouping, rotation, RLE, IoU) is unchanged.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from shapely.geometry import Polygon
from shapely.affinity import rotate as shapely_rotate

logger = logging.getLogger(__name__)


class MultiPlotSAM2:
    """Optimized SAM2 with intelligent batching.

    Parameters
    ----------
    sam2_config     : Short config filename as expected by build_sam2, e.g.
                      "sam2_hiera_l.yaml".  Resolved from the installed sam2
                      package's config directory by Hydra — do NOT pass an
                      absolute path.
    sam2_checkpoint : Path to the model weights .pt file.  Can be a bare
                      filename (resolved by the sam2 package) or an absolute
                      path to a file you have downloaded.
    verbose         : If False, INFO-level logs are suppressed.
    save_masks      : If False, .npy mask files are not written to disk.
    save_overlays   : If False, overlay PNGs are not saved.
    """

    def __init__(
        self,
        sam2_config:     str,
        sam2_checkpoint: str,
        verbose:         bool = True,
        save_masks:      bool = True,
        save_overlays:   bool = True,
    ) -> None:
        self.verbose       = verbose
        self.save_masks    = save_masks
        self.save_overlays = save_overlays

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
        self._log(f"Loading SAM2 on {self.device} …")
        self._log(f"  config     : {sam2_config}")
        self._log(f"  checkpoint : {sam2_checkpoint}")

        start = datetime.now()
        sam2  = build_sam2(
            config_file = sam2_config,
            ckpt_path   = sam2_checkpoint,
            device      = self.device,
        )
        self.predictor = SAM2ImagePredictor(sam2)
        elapsed = (datetime.now() - start).total_seconds()
        self._log(f"  SAM2 loaded in {elapsed:.1f}s")

        self.rotation_threshold       = 5.0
        self.angle_grouping_tolerance = 2.0

    # ------------------------------------------------------------------
    # Public API  (identical signature to original)
    # ------------------------------------------------------------------

    def segment_multiple_plots(
        self,
        point_id:      str,
        stage:         str,
        context_image: Image.Image,
        context_bbox:  tuple,
        prompts:       list,
        output_dir:    Path,
    ) -> list:
        results = []

        try:
            img_np = np.array(context_image)
            if img_np.ndim == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

            h, w = img_np.shape[:2]

            plot_angles = [
                (p, self._calculate_rotation_angle(p.get("polygon_px", [])))
                for p in prompts
            ]
            angle_groups = self._group_by_angle(plot_angles)
            self._log(
                f"  [{point_id}/{stage}] {len(prompts)} plots → "
                f"{len(angle_groups)} rotation group(s)"
            )

            for angle, group_prompts in angle_groups.items():
                try:
                    results.extend(self._process_angle_group(
                        angle, group_prompts, img_np, context_bbox,
                        output_dir, point_id, stage,
                    ))
                except Exception as exc:
                    logger.error(f"[{point_id}/{stage}] Group {angle:.1f}° failed: {exc}")
                    logger.debug(traceback.format_exc())
                    for prompt, _ in group_prompts:
                        results.append(self._failed_result(prompt, f"group error: {exc}"))
                finally:
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

        except Exception as exc:
            logger.error(f"[{point_id}/{stage}] SAM2 batch error: {exc}")
            logger.debug(traceback.format_exc())

        if self.save_overlays:
            try:
                self._save_translucent_overlay(
                    context_image, prompts, results, output_dir, point_id, stage
                )
            except Exception as exc:
                logger.warning(f"[{point_id}/{stage}] Overlay save failed: {exc}")

        return results

    # ------------------------------------------------------------------
    # Rotation helpers
    # ------------------------------------------------------------------

    def _calculate_rotation_angle(self, polygon_px: list) -> float:
        if not polygon_px or len(polygon_px) < 3:
            return 0.0
        try:
            poly  = Polygon(polygon_px)
            mrr   = poly.minimum_rotated_rectangle
            pts   = np.array(mrr.exterior.coords[:-1])
            sides = [
                (np.linalg.norm(pts[(i+1) % len(pts)] - pts[i]), pts[i], pts[(i+1) % len(pts)])
                for i in range(len(pts))
            ]
            sides.sort(key=lambda x: x[0], reverse=True)
            _, p1, p2 = sides[0]
            angle_deg = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            while angle_deg >  45: angle_deg -= 90
            while angle_deg < -45: angle_deg += 90
            return angle_deg
        except Exception:
            return 0.0

    def _group_by_angle(self, plot_angles: list) -> dict:
        groups: dict = defaultdict(list)
        for prompt, angle in plot_angles:
            key = (
                0.0 if abs(angle) < self.rotation_threshold
                else round(angle / self.angle_grouping_tolerance) * self.angle_grouping_tolerance
            )
            groups[key].append((prompt, angle))
        return dict(groups)

    # ------------------------------------------------------------------
    # Group processing
    # ------------------------------------------------------------------

    def _process_angle_group(
        self,
        group_angle:   float,
        group_prompts: list,
        img_np:        np.ndarray,
        context_bbox:  tuple,
        output_dir:    Path,
        point_id:      str,
        stage:         str,
    ) -> list:
        h, w = img_np.shape[:2]

        if abs(group_angle) < self.rotation_threshold:
            rotated_img  = img_np
            new_w, new_h = w, h
            M = M_inv    = None
            new_center   = (w / 2, h / 2)
        else:
            center    = (w / 2, h / 2)
            angle_rad = np.radians(group_angle)
            cos_a, sin_a = abs(np.cos(angle_rad)), abs(np.sin(angle_rad))
            new_w = int(h * sin_a + w * cos_a)
            new_h = int(h * cos_a + w * sin_a)

            M = cv2.getRotationMatrix2D(center, group_angle, 1.0)
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            rotated_img = cv2.warpAffine(
                img_np, M, (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            new_center = (new_w / 2, new_h / 2)
            M_inv = cv2.getRotationMatrix2D(new_center, -group_angle, 1.0)
            M_inv[0, 2] -= (new_w - w) / 2
            M_inv[1, 2] -= (new_h - h) / 2

        # Set image ONCE for the entire group (embeddings computed once)
        self.predictor.set_image(rotated_img)

        results = []
        for prompt, actual_angle in group_prompts:
            try:
                results.append(self._process_single_plot(
                    prompt, actual_angle, img_np, rotated_img,
                    M, M_inv, new_center, w, h, new_w, new_h,
                    context_bbox, output_dir, point_id, stage,
                ))
            except Exception as exc:
                logger.error(f"[{point_id}/{stage}] Plot {prompt['plot_id']} error: {exc}")
                logger.debug(traceback.format_exc())
                results.append(self._failed_result(prompt, str(exc)))

        return results

    # ------------------------------------------------------------------
    # Single-plot processing
    # ------------------------------------------------------------------

    def _process_single_plot(
        self,
        prompt, angle, img_np, rotated_img,
        M, M_inv, new_center, w, h, new_w, new_h,
        context_bbox, output_dir, point_id, stage,
    ) -> dict:
        plot_id    = prompt["plot_id"]
        plot_index = prompt["plot_index"]
        polygon_px = prompt.get("polygon_px", [])

        if M is not None:
            poly_shifted = Polygon([
                (x + (new_w - w) / 2, y + (new_h - h) / 2)
                for x, y in polygon_px
            ])
            rotated_poly = shapely_rotate(poly_shifted, -angle, origin=new_center)
        else:
            rotated_poly = Polygon(polygon_px)

        min_x, min_y, max_x, max_y = rotated_poly.bounds
        bbox_rotated = np.array([int(min_x), int(min_y), int(max_x), int(max_y)])

        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox_rotated,
            mask_input=None,
            multimask_output=False,
        )

        mask_rotated = masks[0]
        score        = float(scores[0])

        if M_inv is not None:
            mask_back = cv2.warpAffine(
                mask_rotated.astype(np.uint8), M_inv, (w, h),
                flags=cv2.INTER_NEAREST,
            )
        else:
            mask_back = mask_rotated.astype(np.uint8)

        mask_binary = (mask_back > 0).astype(np.uint8)

        if M_inv is not None:
            corners_rot = np.array([
                [min_x, min_y], [max_x, min_y],
                [max_x, max_y], [min_x, max_y],
            ], dtype=np.float32)
            corners_orig = cv2.transform(corners_rot.reshape(1, -1, 2), M_inv)[0]
        else:
            corners_orig = np.array([
                [min_x, min_y], [max_x, min_y],
                [max_x, max_y], [min_x, max_y],
            ])

        ctx_west, ctx_south, ctx_east, ctx_north = context_bbox
        aligned_bbox_geo = [
            [ctx_west + (px_x / w) * (ctx_east  - ctx_west),
             ctx_north - (px_y / h) * (ctx_north - ctx_south)]
            for px_x, px_y in corners_orig
        ]

        deg_per_px_lon = (ctx_east  - ctx_west)  / w
        deg_per_px_lat = (ctx_north - ctx_south) / h
        area_m2 = (
            float(np.sum(mask_binary))
            * (deg_per_px_lon * 96_000)
            * (deg_per_px_lat * 111_000)
        )

        gt_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(polygon_px, dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(gt_mask, [pts], 1)
        intersection = np.logical_and(mask_binary, gt_mask).sum()
        union        = np.logical_or(mask_binary,  gt_mask).sum()
        iou          = float(intersection / union) if union > 0 else 0.0

        mask_path_str = ""
        if self.save_masks:
            mask_dir = Path(output_dir) / "masks" / point_id / stage
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{plot_id}.npy"
            np.save(mask_path, mask_binary)
            mask_path_str = str(mask_path)

        ctx_west, ctx_south, ctx_east, ctx_north = context_bbox
        mask_geo_wkt = self._mask_to_geo_polygon(
            mask_binary, ctx_west, ctx_south, ctx_east, ctx_north
        )

        return {
            "plot_id":            plot_id,
            "plot_index":         plot_index,
            "cluster_id":         prompt.get("cluster_id"),
            "status":             "success",
            "error":              "",
            "mask":               mask_binary,
            "mask_geo_wkt":       mask_geo_wkt,
            "area_m2":            area_m2,
            "score":              score,
            "iou":                iou,
            "mask_path":          mask_path_str,
            "aligned_bbox_geo":   aligned_bbox_geo,
            "rotation_angle_deg": angle,
        }

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    def _save_translucent_overlay(
        self, context_img, prompts, results, output_dir, point_id, stage
    ) -> None:
        overlay_dir = Path(output_dir) / "overlays" / point_id
        overlay_dir.mkdir(parents=True, exist_ok=True)

        img     = np.array(context_img.copy())
        overlay = img.copy()

        # Match results by plot_id — results are in angle-group order, not prompt order
        results_by_id = {r.get("plot_id"): r for r in results if r.get("plot_id")}
        for i, prompt in enumerate(prompts):
            color      = self._get_color(i)
            polygon_px = prompt.get("polygon_px", [])
            res        = results_by_id.get(prompt.get("plot_id"), {})
            if len(polygon_px) >= 3:
                pts = np.array(polygon_px, dtype=np.int32)
                cv2.polylines(overlay, [pts], isClosed=True,
                              color=(0, 255, 255), thickness=1)
            if res.get("mask") is not None:
                colored_mask = np.zeros_like(img)
                colored_mask[res["mask"] > 0] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.3, 0)

        Image.fromarray(overlay.astype(np.uint8)).save(
            overlay_dir / f"{stage}_overlay.png"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)

    @staticmethod
    def _failed_result(prompt: dict, error: str) -> dict:
        return {
            "plot_id":            prompt["plot_id"],
            "plot_index":         prompt["plot_index"],
            "cluster_id":         prompt.get("cluster_id"),
            "status":             "failed",
            "error":              error,
            "mask":               None,
            "mask_geo_wkt":       "",
            "area_m2":            0.0,
            "score":              0.0,
            "iou":                0.0,
            "mask_path":          "",
            "aligned_bbox_geo":   None,
            "rotation_angle_deg": 0.0,
        }

    @staticmethod
    def _get_color(idx: int) -> tuple:
        colors = [
            (255, 0, 0),   (0, 255, 0),   (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0),   (0, 128, 0),   (0, 0, 128),
            (255, 128, 0), (128, 255, 0), (0, 255, 128),
        ]
        return colors[idx % len(colors)]

    @staticmethod
    def _mask_to_geo_polygon(
        mask: np.ndarray,
        ctx_west: float,
        ctx_south: float,
        ctx_east: float,
        ctx_north: float,
    ) -> str:
        """
        Convert a binary pixel mask to a georeferenced WGS84 WKT polygon.

        Finds the largest contour in the mask, simplifies it, and maps
        each pixel coordinate to (lon, lat) using the context image bbox.
        Returns a WKT POLYGON string, or "" if no valid contour is found.
        """
        h, w = mask.shape
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return ""
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 1:
            return ""
        # Simplify contour to keep WKT compact while preserving shape
        epsilon = 0.005 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) < 3:
            return ""
        lon_range = ctx_east  - ctx_west
        lat_range = ctx_north - ctx_south
        geo_pts = []
        for pt in simplified:
            px_x = float(pt[0][0])
            px_y = float(pt[0][1])
            lon = ctx_west  + (px_x / w) * lon_range
            lat = ctx_north - (px_y / h) * lat_range
            geo_pts.append((lon, lat))
        # Close the ring
        geo_pts.append(geo_pts[0])
        coords_str = ", ".join(f"{lon:.8f} {lat:.8f}" for lon, lat in geo_pts)
        return f"POLYGON (({coords_str}))"