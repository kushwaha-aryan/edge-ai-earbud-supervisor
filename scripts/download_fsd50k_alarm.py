"""Selective download of FSD50K clips whose label set contains 'Alarm'.

- Reads cached labels (labels/dev.csv, labels/eval.csv); no full-dataset snapshot.
- Filters out Telephone/Ringtone co-labels (ringtone pollution); samples 800 (seed 42).
- Downloads individual wav files via direct resolve URLs into data/raw/fsd50k/Alarm/.
- Skips files already present; validates each (readable/duration/sr/channels/peak/sha256).
- Writes scripts/results/fsd50k_manifest.csv (schema for build_master_manifest.py).
"""

from __future__ import annotations

import hashlib
import io
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
DEST = ROOT / "data" / "raw" / "fsd50k" / "Alarm"

N_TARGET = 800
SEED = 42
EXCLUDE_COLABELS = {"Telephone", "Ringtone"}
BASE = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/"
LICENSE = "CC BY 4.0"
DATASET_URL = "https://huggingface.co/datasets/Fhrozen/FSD50k"

N_WORKERS = 8


def sset(s: str) -> set[str]:
    return {x.strip() for x in str(s).split(",") if x.strip()}


def load_alarm_rows() -> pd.DataFrame:
    dev = pd.read_csv(
        hf_hub_download(repo_id="Fhrozen/FSD50k", filename="labels/dev.csv", repo_type="dataset")
    )
    ev = pd.read_csv(
        hf_hub_download(repo_id="Fhrozen/FSD50k", filename="labels/eval.csv", repo_type="dataset")
    )
    dev["split"] = dev["split"].astype(str)
    ev["split"] = "eval"
    allc = pd.concat([dev, ev], ignore_index=True)

    ev_fnames = set(ev["fname"].astype(str))

    def rel_path(r) -> str:
        if str(r["fname"]) in ev_fnames:
            return f"clips/eval/{r['fname']}.wav"
        return f"clips/dev/{r['fname']}.wav"

    mask = allc["labels"].apply(lambda s: "Alarm" in sset(s))
    print(f"clips with label Alarm: {mask.sum()}")
    mask &= allc["labels"].apply(lambda s: not (sset(s) & EXCLUDE_COLABELS))
    print(f"after excluding {sorted(EXCLUDE_COLABELS)} co-labels: {mask.sum()}")

    sub = allc[mask].copy()
    sub["rel_path"] = sub.apply(rel_path, axis=1)
    if len(sub) > N_TARGET:
        sub = sub.sample(N_TARGET, random_state=SEED).reset_index(drop=True)
    print(f"selected for download: {len(sub)}")
    return sub


def validate(path: Path) -> dict:
    try:
        data = path.read_bytes()
        info = sf.info(io.BytesIO(data))
        audio, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        notes = []
        if info.duration <= 0:
            notes.append("zero_duration")
        if peak < 1e-4:
            notes.append("near_silent")
        return {
            "validation_status": "ok" if not notes else "warning",
            "v_duration": round(info.duration, 4),
            "v_sample_rate": info.samplerate,
            "v_channels": info.channels,
            "v_peak": round(peak, 6),
            "v_notes": ";".join(notes),
            "v_sha256": hashlib.sha256(data).hexdigest(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "validation_status": "invalid",
            "v_duration": "",
            "v_sample_rate": "",
            "v_channels": "",
            "v_peak": "",
            "v_notes": f"{type(exc).__name__}: {exc}"[:250],
            "v_sha256": "",
        }


def fetch_one(row: dict) -> dict:
    dest = DEST / f"{row['fname']}.wav"
    reused = dest.exists() and dest.stat().st_size > 0
    if not reused:
        last = None
        for attempt in range(3):
            try:
                r = requests.get(BASE + row["rel_path"], timeout=90, allow_redirects=True)
                r.raise_for_status()
                dest.write_bytes(r.content)
                last = None
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(2 * (attempt + 1))
        if last is not None:
            return {
                "source_file": f"{row['fname']}.wav",
                "validation_status": "invalid",
                "v_notes": f"download failed: {last}"[:250],
                "reused": False,
                "ok": False,
            }

    v = validate(dest)
    v.update({"reused": reused, "ok": v["validation_status"] != "invalid"})
    return {"source_file": f"{row['fname']}.wav", **v}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    sub = load_alarm_rows()

    # leakage guard: FSD50K fname vs UrbanSound8K fsID (both Freesound-derived)
    try:
        us8k = pd.read_csv(RESULTS / "urbansound8k_manifest.csv", dtype={"fsID": str})
        overlap = set(us8k["fsID"].dropna()) & set(sub["fname"].astype(str))
        if overlap:
            print(f"WARNING: {len(overlap)} clips share source IDs with US8K, removing")
            sub = sub[~sub["fname"].astype(str).isin(overlap)].reset_index(drop=True)
    except FileNotFoundError:
        pass

    rows = sub.to_dict("records")
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(fetch_one, r): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            if i % 50 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)} done ({time.time()-t0:.0f}s)")

    res_by_file = {r["source_file"]: r for r in results}
    manifest_rows = []
    for r in rows:
        v = res_by_file.get(f"{r['fname']}.wav", {})
        dest = DEST / f"{r['fname']}.wav"
        status = v.get("validation_status", "invalid")
        manifest_rows.append(
            {
                "source_dataset": "FSD50K",
                "source_file": f"{r['fname']}.wav",
                "source_id": str(r["fname"]),
                "original_label": r["labels"],
                "project_class": "Alarm",
                "local_path": str(dest.relative_to(ROOT)) if dest.exists() else "",
                "license": LICENSE,
                "source_url": f"{DATASET_URL}/blob/main/{r['rel_path']}",
                "duration": v.get("v_duration", ""),
                "sample_rate": v.get("v_sample_rate", ""),
                "channels": v.get("v_channels", ""),
                "fold": r["split"],
                "start": 0.0,
                "end": v.get("v_duration", ""),
                "validation_status": status,
                "v_duration": v.get("v_duration", ""),
                "v_sample_rate": v.get("v_sample_rate", ""),
                "v_channels": v.get("v_channels", ""),
                "v_peak": v.get("v_peak", ""),
                "v_sha256": v.get("v_sha256", ""),
                "v_notes": v.get("v_notes", ""),
            }
        )

    df = pd.DataFrame(manifest_rows)
    out = RESULTS / "fsd50k_manifest.csv"
    df.to_csv(out, index=False)

    valid = df[df["validation_status"] != "invalid"]
    n = len(df)
    n_ok = int((df["validation_status"] == "ok").sum())
    n_warn = int((df["validation_status"] == "warning").sum())
    n_bad = int((df["validation_status"] == "invalid").sum())
    mb = sum((DEST / f).stat().st_size for f in df["source_file"] if (DEST / f).exists()) / 1e6
    print(f"\nmanifest: {out}")
    print(f"downloaded+valid rows: {len(valid)}/{n}  ok={n_ok} warning={n_warn} invalid={n_bad}")
    print(f"disk usage: {mb:.0f} MB in {DEST}")
    if n_bad:
        print("invalid files:")
        for _, r in df[df['validation_status'] == 'invalid'].iterrows():
            print(f"  {r['source_file']}: {r['v_notes']}")


if __name__ == "__main__":
    main()
