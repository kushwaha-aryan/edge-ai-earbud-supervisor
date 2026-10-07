"""Fetch ESC-50 clips (per-class, selective) into data/raw/esc50/<ProjectClass>/.

Source: https://github.com/karolpiczak/ESC-50 (CC BY-NC 3.0, ESC-10 subset CC BY 3.0).
Downloads individual wav files from raw.githubusercontent.com, validates each clip,
and writes scripts/results/esc50_manifest.csv (schema consumed by build_master_manifest).
Grouping: ESC-50 column `src_file` identifies the original source recording, so all
5s slices cut from the same recording share a source_id (used later for split grouping).
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
RAW = ROOT / "data" / "raw" / "esc50"
META_CACHE = RESULTS / "esc50.csv"

BASE_RAW = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"
BASE_META = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC50_URL = "https://github.com/karolpiczak/ESC-50"
LICENSE = "CC BY-NC 3.0 (ESC-50); ESC-10 subset CC BY 3.0"

# ESC-50 category -> project_class (only classes needed by the taxonomy)
CATEGORY_MAP = {
    "crying_baby": "Baby_Crying",
    "clock_alarm": "Alarm",
    "glass_breaking": "Glass_Breaking",
    "helicopter": "Aircraft",
    "airplane": "Aircraft",
    "train": "Train",
    "footsteps": "Footsteps",
    "door_wood_knock": "Knocking",
    "car_horn": "Vehicle_Horn",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_bytes(data: bytes) -> dict:
    out: dict[str, object] = {}
    try:
        info = sf.info(io.BytesIO(data))
        audio, _sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        notes = []
        if info.duration <= 0:
            notes.append("zero_duration")
        if peak < 1e-4:
            notes.append("near_silent")
        out = {
            "validation_status": "ok" if not notes else "warning",
            "v_duration": round(info.duration, 4),
            "v_sample_rate": info.samplerate,
            "v_channels": info.channels,
            "v_peak": round(peak, 6),
            "v_notes": ";".join(notes),
            "v_sha256": sha256_bytes(data),
        }
    except Exception as exc:  # noqa: BLE001
        out = {
            "validation_status": "invalid",
            "v_duration": "",
            "v_sample_rate": "",
            "v_channels": "",
            "v_peak": "",
            "v_notes": f"{type(exc).__name__}: {exc}"[:250],
            "v_sha256": sha256_bytes(data),
        }
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not META_CACHE.exists():
        r = requests.get(BASE_META, timeout=60)
        r.raise_for_status()
        META_CACHE.write_bytes(r.content)
    meta = pd.read_csv(META_CACHE)
    print(f"esc50.csv rows: {len(meta)}")

    sel = meta[meta["category"].isin(CATEGORY_MAP)].copy()
    print(f"selected clips: {len(sel)} across {sel['category'].nunique()} categories")

    # leakage guard: skip clips whose source recording already exists via US8K / FSD50K
    blocked: set[str] = set()
    try:
        us8k = pd.read_csv(RESULTS / "urbansound8k_manifest.csv", dtype={"fsID": str})
        blocked |= set(us8k["fsID"].dropna().astype(str))
    except FileNotFoundError:
        pass
    try:
        fsd = pd.read_csv(RESULTS / "fsd50k_manifest.csv", dtype={"source_id": str})
        blocked |= set(fsd["source_id"].dropna().astype(str))
    except FileNotFoundError:
        pass
    if blocked:
        before = len(sel)
        sel = sel[~sel["src_file"].astype(str).isin(blocked)].reset_index(drop=True)
        print(f"leakage guard: dropped {before - len(sel)} ESC-50 clips with shared source IDs")

    rows = []
    session = requests.Session()
    for _, r in sel.iterrows():
        pclass = CATEGORY_MAP[r["category"]]
        dest_dir = RAW / pclass
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / r["filename"]

        if dest.exists() and dest.stat().st_size > 0:
            data = dest.read_bytes()
            already = True
        else:
            resp = session.get(BASE_RAW + r["filename"], timeout=60)
            resp.raise_for_status()
            data = resp.content
            dest.write_bytes(data)
            already = False

        v = validate_bytes(data)
        rows.append(
            {
                "source_dataset": "ESC-50",
                "source_file": r["filename"],
                "source_id": str(r["src_file"]),
                "original_label": r["category"],
                "project_class": pclass,
                "local_path": str(dest.relative_to(ROOT)),
                "license": LICENSE,
                "source_url": f"{ESC50_URL}/blob/master/audio/{r['filename']}",
                "duration": v["v_duration"],
                "sample_rate": v["v_sample_rate"],
                "channels": v["v_channels"],
                "fold": str(r["fold"]),
                "start": 0.0,
                "end": v["v_duration"],
                **{
                    k: v[k]
                    for k in (
                        "validation_status",
                        "v_duration",
                        "v_sample_rate",
                        "v_channels",
                        "v_peak",
                        "v_sha256",
                        "v_notes",
                    )
                },
                "_reused": already,
            }
        )
        flag = "reused" if already else v["validation_status"]
        print(f"  {r['filename']} -> {pclass} [{flag}]")

    df = pd.DataFrame(rows).drop(columns=["_reused"])
    out = RESULTS / "esc50_manifest.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out} rows={len(df)}")
    print(df.groupby(["project_class", "validation_status"]).size().to_string())


if __name__ == "__main__":
    main()
