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
from typing import Any, Dict, List, Optional

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