"""Selective FSD50K download for all remaining project classes (batch).

Same workflow proven with the Alarm class:
- filter labels/*metadata* CSVs (never a full-dataset snapshot)
- deterministic sampling (seed 42) with per-class caps
- leakage guard: drop clips whose fname matches US8K fsID or ESC-50 src_file
- cross-class dedupe (a clip serves at most one project class)
- individual resolve-URL downloads, skip-if-exists
- validate (readable/duration/sr/channels/peak/sha256)
- merge rows into scripts/results/fsd50k_manifest.csv
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
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
DEST_ROOT = ROOT / "data" / "raw" / "fsd50k"

SEED = 42
N_WORKERS = 12
BASE = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/"
LICENSE = "CC BY 4.0"
DATASET_URL = "https://huggingface.co/datasets/Fhrozen/FSD50k"


def sset_lower(s: str) -> set[str]:
    return {x.strip().lower() for x in str(s).split(",") if x.strip()}


def make_matcher(spec):
    if spec == "glass_shatter":

        def m(ll: set[str]) -> bool:
            return "shatter" in ll and "glass" in ll

        return m

    def m(ll: set[str]) -> bool:
        return any(t in ll for t in spec)

    return m


# (project_class, label-substrings OR 'glass_shatter', cap or None)
CFG = [
    ("Vehicle_Horn", ["vehicle_horn_and_car_horn_and_honking"], None),
    ("Doorbell", ["doorbell"], None),
    ("Train", ["train"], None),
    ("Aircraft", ["aircraft"], None),  # matches 'Aircraft' and 'Fixed-wing_aircraft...'
    ("Motorcycle", ["motorcycle"], None),
    ("Help_Shouting", ["shout", "screaming"], 500),
    ("Glass_Breaking", "glass_shatter", 500),
    ("Gunshot", ["gunshot_and_gunfire"], None),
    ("Knocking", ["knock"], None),
    ("Footsteps", ["walk_and_footsteps"], None),
]


def load_labels() -> pd.DataFrame:
    dev = pd.read_csv(
        hf_hub_download(repo_id="Fhrozen/FSD50k", filename="labels/dev.csv", repo_type="dataset")
    )
    ev = pd.read_csv(
        hf_hub_download(repo_id="Fhrozen/FSD50k", filename="labels/eval.csv", repo_type="dataset")
    )
    ev = ev.assign(split="eval")
    allc = pd.concat([dev, ev], ignore_index=True)

    info = allc["labels"].str.lower()
    print(
        "info: rows containing 'fire' =", int(info.str.contains("fire").sum()),
        "| 'smoke' =", int(info.str.contains("smoke").sum()),
    )
    return allc


def blocked_ids() -> set[str]:
    blocked: set[str] = set()
    try:
        us8k = pd.read_csv(RESULTS / "urbansound8k_manifest.csv", dtype={"fsID": str})
        blocked |= set(us8k["fsID"].dropna().astype(str))
        print(f"leakage guard: {len(blocked)} US8K source IDs loaded")
    except FileNotFoundError:
        print("leakage guard: urbansound8k_manifest.csv not found")
    try:
        esc = pd.read_csv(RESULTS / "esc50_manifest.csv", dtype={"source_id": str})
        n0 = len(blocked)
        blocked |= set(esc["source_id"].dropna().astype(str))
        print(f"leakage guard: +{len(blocked)-n0} ESC-50 source IDs")
    except FileNotFoundError:
        pass
    try:
        fsd = pd.read_csv(RESULTS / "fsd50k_manifest.csv", dtype={"source_id": str, "project_class": str})
        prev = set(fsd.loc[fsd["project_class"] == "Alarm", "source_id"].dropna().astype(str))
        n0 = len(blocked)
        blocked |= prev
        print(f"leakage guard: +{len(blocked)-n0} existing FSD50K clip IDs (Alarm)")
    except FileNotFoundError:
        pass
    return blocked


def select(allc: pd.DataFrame, blocked: set[str]) -> list[dict]:
    assignments: dict[str, list] = {}
    seen: set[str] = set()
    allc = allc.copy()
    allc["_ll"] = allc["labels"].map(sset_lower)

    for pclass, spec, cap in CFG:
        matcher = make_matcher(spec)
        sub = allc[allc["_ll"].map(matcher)].copy()
        if spec == "glass_shatter" and len(sub) < 400:
            print(f"[{pclass}] glass+shatter only {len(sub)} -> falling back to all 'shatter'")
            sub = allc[allc["_ll"].map(lambda ll: "shatter" in ll)].copy()

        sub = sub[~sub["fname"].astype(str).isin(blocked)]
        sub = sub[~sub["fname"].astype(str).isin(seen)]
        n_all = len(sub)
        if cap and len(sub) > cap:
            sub = sub.sample(cap, random_state=SEED)
        sub = sub.reset_index(drop=True)
        seen |= set(sub["fname"].astype(str))
        assignments[pclass] = sub.to_dict("records")
        print(f"[{pclass}] candidates={n_all} selected={len(sub)} (cap={cap})")
    return assignments


def estimate_size(rows: list[dict]) -> float:
    sample = rows[:6]
    sizes = []
    for r in sample:
        try:
            h = requests.head(BASE + r["rel_path"], allow_redirects=True, timeout=20)
            sizes.append(int(h.headers.get("content-length", 0)))
        except Exception:  # noqa: BLE001
            pass
    if not sizes:
        return 0.0
    return sum(sizes) / len(sizes) * len(rows) / 1e6


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
        if info.samplerate not in (8000, 11025, 16000, 22050, 32000, 44100, 48000):
            notes.append(f"unusual_sr_{info.samplerate}")
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


def fetch_one(job: tuple[dict, str]) -> dict:
    row, pclass = job
    dest = DEST_ROOT / pclass / f"{row['fname']}.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
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
                "fname": str(row["fname"]),
                "pclass": pclass,
                "validation_status": "invalid",
                "v_notes": f"download failed: {last}"[:250],
            }
    v = validate(dest)
    return {"fname": str(row["fname"]), "pclass": pclass, **v}


def main() -> None:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    allc = load_labels()
    blocked = blocked_ids()
    assignments = select(allc, blocked)

    ev_fnames = set(
        pd.read_csv(
            hf_hub_download(repo_id="Fhrozen/FSD50k", filename="labels/eval.csv", repo_type="dataset")
        )["fname"].astype(str)
    )

    jobs: list[tuple[dict, str]] = []
    job_meta: dict[tuple[str, str], dict] = {}
    for pclass, rows in assignments.items():
        for r in rows:
            split = "eval" if str(r["fname"]) in ev_fnames else str(r["split"])
            r = dict(r)
            r["rel_path"] = (
                f"clips/eval/{r['fname']}.wav" if split == "eval" else f"clips/dev/{r['fname']}.wav"
            )
            r["split"] = split
            jobs.append((r, pclass))
            job_meta[(pclass, str(r["fname"]))] = r

    print(f"\ntotal clips to fetch: {len(jobs)}")
    for pclass, rows in assignments.items():
        if rows:
            est = estimate_size(rows)
            print(f"  [{pclass}] {len(rows)} clips ~ {est:.0f} MB (estimated)")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(fetch_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 200 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done ({time.time()-t0:.0f}s)")

    res_by = {(r["pclass"], r["fname"]): r for r in results}
    new_rows = []
    for pclass, fname in job_meta:
        r = job_meta[(pclass, fname)]
        v = res_by.get((pclass, fname), {})
        dest = DEST_ROOT / pclass / f"{fname}.wav"
        new_rows.append(
            {
                "source_dataset": "FSD50K",
                "source_file": f"{fname}.wav",
                "source_id": fname,
                "original_label": r["labels"],
                "project_class": pclass,
                "local_path": str(dest.relative_to(ROOT)) if dest.exists() else "",
                "license": LICENSE,
                "source_url": f"{DATASET_URL}/blob/main/{r['rel_path']}",
                "duration": v.get("v_duration", ""),
                "sample_rate": v.get("v_sample_rate", ""),
                "channels": v.get("v_channels", ""),
                "fold": r["split"],
                "start": 0.0,
                "end": v.get("v_duration", ""),
                "validation_status": v.get("validation_status", "invalid"),
                "v_duration": v.get("v_duration", ""),
                "v_sample_rate": v.get("v_sample_rate", ""),
                "v_channels": v.get("v_channels", ""),
                "v_peak": v.get("v_peak", ""),
                "v_sha256": v.get("v_sha256", ""),
                "v_notes": v.get("v_notes", ""),
            }
        )

    new_df = pd.DataFrame(new_rows)
    out = RESULTS / "fsd50k_manifest.csv"
    if out.exists():
        old = pd.read_csv(out, dtype={"source_id": str, "source_file": str})
        merged = pd.concat([old, new_df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["source_dataset", "source_file", "project_class"], keep="first"
        )
    else:
        merged = new_df
    merged.to_csv(out, index=False)

    print(f"\nmanifest: {out} rows={len(merged)} (added {len(new_df)})")
    for pclass, rows in assignments.items():
        sub = new_df[new_df["project_class"] == pclass]
        ok = (sub["validation_status"] != "invalid").sum()
        mb = (
            sum(
                (DEST_ROOT / pclass / f"{f}").stat().st_size
                for f in sub["source_file"]
                if (DEST_ROOT / pclass / f).exists()
            )
            / 1e6
        )
        print(f"  [{pclass}] downloaded={ok}/{len(rows)} size={mb:.0f} MB")
    bad = new_df[new_df["validation_status"] == "invalid"]
    if len(bad):
        print(f"invalid: {len(bad)}")
        for _, r in bad.head(20).iterrows():
            print(f"  {r['project_class']}/{r['source_file']}: {r['v_notes']}")


if __name__ == "__main__":
    main()
