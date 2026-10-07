import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "babycry"
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://huggingface.co/datasets/owlgebra-ai/babycry/resolve/main/data"
SHARD = "train-00000-of-00004.parquet"
OUT = RAW / SHARD


def main() -> None:
    if not OUT.exists():
        import requests

        print(f"downloading {SHARD} ...")
        with requests.get(f"{BASE}/{SHARD}", stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            n = 0
            with open(OUT.with_suffix(".part"), "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
                    if n % (50 << 20) < (1 << 20):
                        print(f"  {n / 1e6:.0f}/{total / 1e6:.0f} MB")
            OUT.with_suffix(".part").replace(OUT)
        print(f"done {OUT.stat().st_size / 1e6:.1f} MB")

    pf = pq.ParquetFile(OUT)
    print("schema:")
    for field in pf.schema_arrow:
        print("  ", field.name, ":", str(field.type)[:80])

    cols = ["clip_id", "source_dataset", "orig_label", "valence", "speaker_id",
            "source_license", "duration_s", "orig_sr"]
    t = pf.read(columns=cols).to_pandas()
    print(f"\nrows: {len(t)}")
    print("\nsource_dataset x orig_label:")
    print(t.groupby(["source_dataset", "orig_label"]).size().to_string())
    print("\nvalence counts per source:")
    print(t.groupby(["source_dataset", "valence"]).size().to_string())
    print("\nduration_s stats:")
    print(t["duration_s"].describe().to_string())
    print("\norig_sr:", t["orig_sr"].value_counts().to_dict())
    print("licenses:", t["source_license"].value_counts().to_dict())
    print("speakers:", t["speaker_id"].nunique())
    print("\nfirst rows:")
    print(t.head(10).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
