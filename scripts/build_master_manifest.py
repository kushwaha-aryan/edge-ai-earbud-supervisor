"""Build data/manifests/master_audio_manifest.csv from per-dataset manifest CSVs.

Each per-dataset manifest lives in scripts/results/*_manifest.csv and must have at
least: source_dataset, source_file, source_id, original_label, project_class, local_path.
Optional: license, source_url, duration, sample_rate, channels, fold.

Rows are validated against the audio on disk (exists/readable/length/samplerate/
channels/peak/sha256) the first time they are merged. Validation results are
cached in scripts/results/_validation/<source_dataset>__<source_file>.json so
rebuilds are fast; delete that folder to force re-validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
VALCACHE = RESULTS / "_validation"
OUT = ROOT / "data" / "manifests" / "master_audio_manifest.csv"

REQUIRED = [
    "source_dataset",
    "source_file",
    "source_id",
    "original_label",
    "project_class",
    "local_path",
]

# legacy per-dataset column names -> canonical master names
LEGACY_RENAME = {
    "audio_path": "local_path",
    "slice_file_name": "source_file",
    "fsID": "source_id",
    "class": "original_label",
}

# defaults applied only when the column is absent or empty
DATASET_DEFAULTS = {
    "UrbanSound8K": {
        "license": "CC BY-NC 3.0",
        "source_url": "https://urbansounddataset.weebly.com/urbansound8k.html",
    },
    "ESC-50": {
        "license": "CC BY-NC 3.0 (ESC-50)",
        "source_url": "https://github.com/karolpiczak/ESC-50",
    },
}
OPTIONAL = [
    "license",
    "source_url",
    "duration",
    "sample_rate",
    "channels",
    "fold",
    "start",
    "end",
]
VALIDATION = [
    "validation_status",
    "v_duration",
    "v_sample_rate",
    "v_channels",
    "v_peak",
    "v_sha256",
    "v_notes",
]


def validate_row(row: dict) -> dict:
    key = f"{row['source_dataset']}__{row['source_file']}"
    VALCACHE.mkdir(parents=True, exist_ok=True)
    cpath = VALCACHE / (key.replace("/", "_").replace("\\", "_") + ".json")
    if cpath.exists():
        return json.loads(cpath.read_text(encoding="utf-8"))

    out = {k: "" for k in VALIDATION}
    path = Path(row["local_path"])
    if not path.is_absolute():
        path = ROOT / path
    try:
        if not path.exists():
            raise FileNotFoundError(str(path))
        info = sf.info(str(path))
        peak = 0.0
        with sf.SoundFile(str(path)) as f:
            while True:
                data = f.read(65536, dtype="float32", always_2d=False)
                if data.size == 0:
                    break
                peak = max(peak, float(np.max(np.abs(data))))
        notes = []
        if info.duration <= 0:
            notes.append("zero_duration")
        if peak < 1e-4:
            notes.append("near_silent")
        if info.samplerate <= 0:
            notes.append("bad_samplerate")
        out.update(
            {
                "validation_status": "ok" if not notes else "warning",
                "v_duration": round(info.duration, 4),
                "v_sample_rate": info.samplerate,
                "v_channels": info.channels,
                "v_peak": round(peak, 6),
                "v_notes": ";".join(notes),
            }
        )
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out["v_sha256"] = h.hexdigest()
    except Exception as exc:  # noqa: BLE001 - capture any read failure
        out["validation_status"] = "invalid"
        out["v_notes"] = f"{type(exc).__name__}: {exc}"[:250]

    cpath.write_text(json.dumps(out), encoding="utf-8")
    return out


def main() -> None:
    parts = sorted(RESULTS.glob("*_manifest.csv"))
    if not parts:
        raise SystemExit("no per-dataset manifests found in scripts/results")

    frames = []
    for p in parts:
        df = pd.read_csv(p, dtype={"source_id": str, "source_file": str, "fold": str})
        df = df.rename(columns={k: v for k, v in LEGACY_RENAME.items() if k in df.columns})
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise SystemExit(f"{p.name}: missing required columns {missing}")
        for c in OPTIONAL + VALIDATION:
            if c not in df.columns:
                df[c] = ""

        for dataset, defaults in DATASET_DEFAULTS.items():
            mask = df["source_dataset"].eq(dataset)
            for col, val in defaults.items():
                empty = df[col].isna() | df[col].astype(str).str.strip().eq("")
                df.loc[mask & empty, col] = val

        need_dur = df["duration"].isna() | df["duration"].astype(str).str.strip().eq("")
        has_se = need_dur & df["start"].notna() & df["end"].notna()
        if has_se.any():
            dur = (
                df.loc[has_se, "end"].astype(float) - df.loc[has_se, "start"].astype(float)
            ).round(4)
            df["duration"] = df["duration"].astype(object)
            df.loc[has_se, "duration"] = dur.astype(object)

        frames.append(df[REQUIRED + OPTIONAL + VALIDATION])

    master = pd.concat(frames, ignore_index=True)
    for c in OPTIONAL + VALIDATION:
        if c in master.columns:
            master[c] = master[c].astype(object)

    dup_mask = master.duplicated(subset=["source_dataset", "source_file"], keep="first")
    if dup_mask.any():
        print(f"dropping {int(dup_mask.sum())} duplicate (dataset,file) rows")
        master = master[~dup_mask]

    need = master["validation_status"].astype(str).str.strip().eq("")
    if need.any():
        print(f"validating {int(need.sum())} new rows ...")
        for idx in master.index[need]:
            res = validate_row(master.loc[idx].to_dict())
            for k, v in res.items():
                master.at[idx, k] = v

    OUT.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT, index=False)

    ok = (master["validation_status"] == "ok").sum()
    warn = (master["validation_status"] == "warning").sum()
    bad = (master["validation_status"] == "invalid").sum()
    print(f"\nwrote {OUT}")
    print(f"rows={len(master)} ok={ok} warning={warn} invalid={bad}")
    print("\nper project_class:")
    print(master.groupby(["project_class", "validation_status"]).size().to_string())


if __name__ == "__main__":
    main()
