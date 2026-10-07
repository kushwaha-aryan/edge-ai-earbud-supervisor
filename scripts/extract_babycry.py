import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "babycry"
SHARD = RAW / "train-00000-of-00004.parquet"
OUTDIR = RAW / "Baby_Crying"
MANIFEST = ROOT / "scripts" / "results" / "babycry_manifest.csv"
N_TARGET = 500
SEED = 42

KEEP_DONATEACRY = {"hungry", "belly_pain", "discomfort", "tired"}  # burping excluded
KEEP_RECANVO = {"dysregulated", "dysregulation-sick", "dysregulation-bathroom", "frustrated"}


def main() -> None:
    cols = ["clip_id", "source_dataset", "orig_label", "valence",
            "source_license", "duration_s", "orig_sr", "audio"]
    t = pq.ParquetFile(SHARD).read(columns=cols).to_pandas()

    print("label x valence (recanvo):")
    rv = t[t["source_dataset"] == "recanvo"]
    print(pd.crosstab(rv["orig_label"], rv["valence"]).to_string())

    sel = t[
        (t["source_dataset"] == "donateacry") & (t["orig_label"].isin(KEEP_DONATEACRY))
        | (t["source_dataset"] == "recanvo")
        & (t["valence"] == "distress")
        & (t["orig_label"].isin(KEEP_RECANVO))
    ].copy()
    print(f"\neligible after filter: {len(sel)}")
    if len(sel) < N_TARGET:
        print("WARNING: eligible pool smaller than target")
    sel = sel.sample(n=min(N_TARGET, len(sel)), random_state=SEED).sort_index()
    print(f"selected: {len(sel)}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows, bad = [], 0
    for i, r in enumerate(sel.itertuples(index=False), 1):
        blob = r.audio["bytes"]
        safe = r.clip_id.replace(":", "__").replace("/", "__")
        dest = OUTDIR / f"{safe}.wav"
        if not dest.exists():
            dest.write_bytes(blob)
        try:
            info = sf.info(str(dest))
            if abs(info.duration - float(r.duration_s)) > 0.35:
                raise ValueError(f"duration {info.duration:.2f} != {r.duration_s:.2f}")
            if info.samplerate != int(r.orig_sr):
                raise ValueError(f"sr {info.samplerate} != {r.orig_sr}")
            h = hashlib.sha256(blob).hexdigest()
            status, notes, dur, sr, ch = "ok", "", info.duration, info.samplerate, info.channels
        except Exception as e:  # noqa: BLE001
            bad += 1
            status, notes, dur, sr, ch, h = "invalid", str(e)[:120], None, None, None, None
            print(f"  INVALID {dest.name}: {notes}")
        rows.append(
            dict(
                source_dataset="owlgebra_babycry",
                source_file=f"{safe}.wav",
                source_id=r.clip_id,
                original_label=r.orig_label,
                project_class="Baby_Crying",
                local_path=str(dest.relative_to(ROOT)).replace("\\", "/"),
                license=r.source_license,
                source_url="https://huggingface.co/datasets/owlgebra-ai/babycry",
                duration=dur,
                sample_rate=sr,
                channels=ch,
                fold=-1,
                start=0.0,
                end=dur,
                validation_status=status,
                v_duration=dur,
                v_sample_rate=sr,
                v_channels=ch,
                v_peak=None,
                v_sha256=h,
                v_notes=notes,
            )
        )
        if i % 100 == 0:
            print(f"  {i}/{len(sel)} extracted")

    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST, index=False)
    print(f"\nwrote {MANIFEST} rows={len(df)} bad={bad}")
    print(df.groupby(["original_label", "validation_status"]).size().to_string())
    print(df["license"].value_counts().to_dict())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
