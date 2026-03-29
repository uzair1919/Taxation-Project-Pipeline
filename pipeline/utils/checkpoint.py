"""
pipeline/utils/checkpoint.py
=============================
Atomic, JSON-backed checkpoint manager.

Each point is recorded as completed or failed.  On restart the orchestrator
can skip completed points entirely, avoiding redundant SAM2 inference and
ESRI tile fetches for large datasets.

Design
------
* Writes to a .tmp file then renames — atomic on POSIX, safe-ish on Windows.
* Stores minimal serialisable state only (no numpy arrays / PIL images).
* Thread-safe in the sense that all mutations happen in the main thread
  (the pipeline is single-threaded per point by design).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Tracks pipeline progress for a batch run.

    State file format (checkpoint.json)
    ------------------------------------
    {
      "version": 1,
      "started": "<iso timestamp>",
      "completed": {
        "<point_id>": { ...minimal result dict... }
      },
      "failed": {
        "<point_id>": { "error": "...", "timestamp": "..." }
      }
    }
    """

    VERSION = 1

    def __init__(self, output_dir: Path) -> None:
        self.checkpoint_path = Path(output_dir) / "checkpoint.json"
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_done(self, point_id: str) -> bool:
        """Return True if this point has been successfully completed."""
        return point_id in self.data["completed"]

    def was_failed(self, point_id: str) -> bool:
        """Return True if this point previously failed."""
        return point_id in self.data["failed"]

    def mark_done(self, point_id: str, result_summary: Dict[str, Any]) -> None:
        """
        Record a successful point result.

        Parameters
        ----------
        point_id       : unique point identifier
        result_summary : serialisable dict (no numpy arrays, no PIL images)
        """
        self.data["completed"][point_id] = {
            **result_summary,
            "_checkpoint_ts": datetime.now().isoformat(timespec="seconds"),
        }
        self.data["failed"].pop(point_id, None)
        self._save()

    def mark_failed(self, point_id: str, error: str) -> None:
        """Record a failed point."""
        self.data["failed"][point_id] = {
            "error": error,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._save()

    def mark_excel_written(self, point_id: str) -> None:
        """Confirm that the Excel was successfully written for this point."""
        if point_id in self.data["completed"]:
            self.data["completed"][point_id]["excel_written"] = True
            self._save()

    def is_complete(
        self,
        point_id:        str,
        cfg:             Any,
        excel_point_ids: Optional[Set[str]],
    ) -> Tuple[bool, str]:
        """
        Return (should_skip, reason).

        should_skip=True only when every step required by the current config
        is confirmed done.  Checks performed:

        1. Point is in the completed checkpoint.
        2. Point data is present in the output Excel
           (verified via excel_point_ids if the file could be read; falls back
           to the stored excel_written flag when the file could not be loaded).
        3. For new-format entries: refinement succeeded, and all years in
           cfg.HEIGHT_YEARS have recorded height results.

        Legacy entries (written before granular tracking) pass once the Excel
        presence check is satisfied, so old runs are never silently discarded.
        """
        if point_id not in self.data["completed"]:
            return False, "not yet processed"

        entry = self.data["completed"][point_id]

        # ── 1. Excel presence / write confirmation ───────────────────────
        if excel_point_ids is not None:
            # We loaded the Excel successfully — check directly
            if point_id not in excel_point_ids:
                return False, "data not found in output Excel"
        else:
            # Could not load the Excel — fall back to the stored flag.
            # For legacy entries the flag is absent; treat as True (optimistic)
            # so we don't force a full re-run every time the Excel is unreadable.
            excel_ok = entry.get("excel_written", "refinement_ok" not in entry)
            if not excel_ok:
                return False, "Excel write not confirmed and Excel file unreadable"

        # ── 2. Legacy entries pass once Excel presence is satisfied ──────
        if "refinement_ok" not in entry:
            return True, "complete (legacy checkpoint entry)"

        # ── 3. Granular checks for new-format entries ────────────────────
        if not entry.get("refinement_ok", False):
            return False, "refinement did not complete successfully"

        if getattr(cfg, "RUN_HEIGHT_ESTIMATION", False):
            required = {int(y) for y in getattr(cfg, "HEIGHT_YEARS", [])}
            done     = {int(y) for y in entry.get("height_years_done", [])}
            missing  = required - done
            if missing:
                return False, f"height estimation missing for year(s): {sorted(missing)}"

        return True, "all steps complete"

    def reset(self) -> None:
        """Wipe all checkpoint state (use with --reset-checkpoint)."""
        self.data = self._empty()
        self._save()
        logger.info("Checkpoint reset.")

    def get_completed(self) -> List[str]:
        """Return list of completed point_ids."""
        return list(self.data["completed"].keys())

    def get_failed(self) -> List[str]:
        """Return list of failed point_ids."""
        return list(self.data["failed"].keys())

    @property
    def n_completed(self) -> int:
        return len(self.data["completed"])

    @property
    def n_failed(self) -> int:
        return len(self.data["failed"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty(self) -> dict:
        return {
            "version": self.VERSION,
            "started": datetime.now().isoformat(timespec="seconds"),
            "completed": {},
            "failed": {},
        }

    def _load(self) -> dict:
        if self.checkpoint_path.exists():
            try:
                data = json.loads(
                    self.checkpoint_path.read_text(encoding="utf-8")
                )
                if data.get("version") == self.VERSION:
                    logger.info(
                        f"Checkpoint loaded: "
                        f"{len(data.get('completed', {}))} completed, "
                        f"{len(data.get('failed', {}))} failed"
                    )
                    return data
                logger.warning(
                    "Checkpoint version mismatch — starting fresh."
                )
            except Exception as exc:
                logger.warning(f"Could not read checkpoint: {exc} — starting fresh.")
        return self._empty()

    def _save(self) -> None:
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.data, indent=2, default=str),
            encoding="utf-8",
        )
        tmp.replace(self.checkpoint_path)