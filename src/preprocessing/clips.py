"""Stage 1 of the Phase 2A preprocessing pipeline: clip-level decisions and
the provenance ledger.

Implements the approved Phase 2A policy (FINAL_PROJECT_BLUEPRINT.md Change
Log, 2026-10-08):
  D1   drop clips flagged ``extreme_short`` (< 0.5 s)
  D4   drop clips flagged ``silent`` (RMS <= -60 dBFS)
  D2   truncate each kept clip to its first 15.0 s
  D6   mono downmix (mean over channels)   - applied in Stage 3
  D5   resample to 16 kHz with soxr_hq     - applied in Stage 3
  OA4  babycry grouping: DonateACry clips group by canonical subject UUID,
       ReCANVo clips group by recording session (YYMMDD_HHMM prefix)

Writes ``data/provenance/phase2a_clip_ledger.csv`` with one row per manifest
clip, kept or excluded, covering the provenance fields of blueprint section
5.4 (source dataset/release, original label, sample id + grouping id,
duration/sample rate/energy, accepted class, and every applied parameter).
The ledger is the single source of truth for which clips Stage 3 may read.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

DONATEACRY_ID_RE = re.compile(
    r"^donateacry:([0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12})"
)
RECANVO_SESSION_RE = re.compile(r"^recanvo:(\d{6}_\d{4})")

LEDGER_COLUMNS = [
    "clip_id",
    "source_dataset",
    "source_file",
    "source_id",
    "group_key",
    "original_label",
    "project_class",
    "class_id",
    "license",
    "source_url",
    "local_path",
    "duration_s",
    "sample_rate",
    "channels",
    "rms_db",
    "peak",
    "sha256",
    "health_flags",
    "status",
    "reason",
    "truncated",
    "processed_duration_s",
    "resample_sr",
    "mono",
    "window_samples",
    "window_step_samples",
    "pad_fraction",
]


def resolve_path(raw: str | Path) -> Path:
    """Resolve a manifest ``local_path`` against the project root."""
    p = Path(raw)
    return p if p.is_absolute() else config.PROJECT_ROOT / p


def group_key_for(source_dataset: str, source_id: str) -> str:
    """Return the split-atomic source group for one manifest row (OA4)."""
    sid = str(source_id)
    if sid.startswith("donateacry:"):
        m = DONATEACRY_ID_RE.match(sid)
        if m is None:
            raise ValueError(f"donateacry source_id without canonical UUID: {sid!r}")
        return f"babycry/donateacry:{m.group(1).lower()}"
    if sid.startswith("recanvo:"):
        m = RECANVO_SESSION_RE.match(sid)
        if m is None:
            raise ValueError(f"recanvo source_id without session prefix: {sid!r}")
        return f"babycry/recanvo:{m.group(1)}"
    if not sid or sid == "nan" or sid == "None":
        raise ValueError(f"empty source_id for dataset {source_dataset!r}")
    return f"{str(source_dataset).lower()}/{sid}"


def _exclusion_reason(extreme: bool, silent: bool) -> str:
    parts = []
    if extreme:
        parts.append(f"extreme_short(<{config.EXCLUDE_SHORTER_THAN_S:g}s)")
    if silent:
        parts.append(f"silent(RMS<={config.EXCLUDE_SILENT_DBFS:g}dBFS)")
    return ";".join(parts)


def build_ledger(
    manifest_path: Path | None = None,
    file_health_path: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Build the Phase 2A clip ledger from the manifest + file health flags."""
    manifest_path = Path(manifest_path or config.MANIFEST_PATH)
    file_health_path = Path(file_health_path or config.FILE_HEALTH_PATH)
    out_path = Path(out_path or (config.PROVENANCE_DIR / "phase2a_clip_ledger.csv"))

    manifest = pd.read_csv(manifest_path)
    health = pd.read_csv(file_health_path)

    if len(manifest) != len(health) or not np.array_equal(
        health["manifest_row"].to_numpy(), np.arange(len(manifest))
    ):
        raise ValueError("file_health.csv rows are not aligned with the master manifest")

    flag_lists = health["flags"].fillna("").str.split(",")
    extreme = flag_lists.apply(lambda xs: "extreme_short" in xs)
    silent = flag_lists.apply(lambda xs: "silent" in xs)
    reason = pd.Series(
        [_exclusion_reason(e, s) for e, s in zip(extreme, silent)],
        index=manifest.index,
        dtype=object,
    )
    status = pd.Series(
        np.where(reason != "", "excluded", "kept"), index=manifest.index, dtype=object
    )
    kept = status == "kept"

    class_id = manifest["project_class"].map(config.TRIGGER_CLASS_TO_ID)
    missing = sorted(manifest.loc[class_id.isna(), "project_class"].unique())
    if missing:
        raise ValueError(f"project_class values not in config.TRIGGER_CLASS_TO_ID: {missing}")

    clip_id = manifest["source_dataset"].astype(str) + "/" + manifest["source_file"].astype(str)
    if clip_id.duplicated().any():
        dupes = clip_id[clip_id.duplicated()].head(5).tolist()
        raise ValueError(f"clip_id is not unique in the manifest: {dupes}")

    group_key = [
        group_key_for(ds, sid)
        for ds, sid in zip(
            manifest["source_dataset"].astype(str), manifest["source_id"].astype(str)
        )
    ]

    duration = health["duration"]
    processed = pd.Series(np.nan, index=manifest.index, dtype="float64")
    processed.loc[kept] = duration.loc[kept].clip(upper=config.TRUNCATE_SECONDS)

    def kept_only(values: dict) -> pd.Series:
        col = pd.Series(values["empty"], index=manifest.index, dtype=values["dtype"])
        col.loc[kept] = values["fill"]
        return col

    resample_sr = kept_only(
        {"empty": np.nan, "dtype": "float64", "fill": float(config.SAMPLE_RATE)}
    )
    mono = kept_only({"empty": pd.NA, "dtype": "boolean", "fill": True})
    window_samples = kept_only(
        {"empty": np.nan, "dtype": "float64", "fill": float(config.WINDOW_SAMPLES)}
    )
    window_step = kept_only(
        {"empty": np.nan, "dtype": "float64", "fill": float(config.WINDOW_STEP)}
    )
    pad_fraction = kept_only(
        {"empty": np.nan, "dtype": "float64", "fill": float(config.MAX_ZERO_PAD_FRACTION)}
    )

    ledger = pd.DataFrame(
        {
            "clip_id": clip_id,
            "source_dataset": manifest["source_dataset"],
            "source_file": manifest["source_file"],
            "source_id": manifest["source_id"],
            "group_key": group_key,
            "original_label": manifest["original_label"],
            "project_class": manifest["project_class"],
            "class_id": class_id,
            "license": manifest["license"],
            "source_url": manifest["source_url"],
            "local_path": manifest["local_path"],
            "duration_s": duration,
            "sample_rate": health["sample_rate"],
            "channels": health["channels"],
            "rms_db": health["rms_db"],
            "peak": health["peak"],
            "sha256": health["sha256"],
            "health_flags": health["flags"],
            "status": status,
            "reason": reason,
            "truncated": (duration > config.TRUNCATE_SECONDS) & kept,
            "processed_duration_s": processed,
            "resample_sr": resample_sr,
            "mono": mono,
            "window_samples": window_samples,
            "window_step_samples": window_step,
            "pad_fraction": pad_fraction,
        }
    )[LEDGER_COLUMNS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(out_path, index=False)
    return ledger
