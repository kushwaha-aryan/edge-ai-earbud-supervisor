import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "zenodo_3689288"
RESULTS = ROOT / "scripts" / "results"
RECORD = "3689288"
ZIP_PATH = RAW / "osr_domestic.zip"

# zip-internal path keyword -> project class (conservative: exact intent only)
MAP = {
    "fire alarm": "Smoke_Fire_Alarm",
    "smoke alarm": "Smoke_Fire_Alarm",
    "smoke detector": "Smoke_Fire_Alarm",
    "fire_alarm": "Smoke_Fire_Alarm",
    "smoke_alarm": "Smoke_Fire_Alarm",
    "door bell": "Doorbell",
    "doorbell": "Doorbell",
    "door_bell": "Doorbell",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(name: str) -> str | None:
    low = name.lower()
    for k, v in MAP.items():
        if k in low:
            return v
    return None


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    # 1. metadata -> file url
    meta = requests.get(f"https://zenodo.org/api/records/{RECORD}", timeout=60).json()
    fl = meta["files"][0]
    url = fl["links"]["self"]
    size_expected = fl["size"]
    print(f"record file: {fl['key']} size={size_expected / 1e6:.1f} MB")

    # 2. download (skip if complete)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size == size_expected:
        print("zip already downloaded")
    else:
        print("downloading zip ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = ZIP_PATH.with_suffix(".part")
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
                    if n % (50 << 20) < (1 << 20):
                        print(f"  {n / 1e6:.0f}/{size_expected / 1e6:.0f} MB")
            tmp.replace(ZIP_PATH)
        print(f"downloaded {ZIP_PATH.stat().st_size / 1e6:.1f} MB")

    # 3. inspect structure
    zf = zipfile.ZipFile(ZIP_PATH)
    names = [n for n in zf.namelist() if not n.endswith("/")]
    print(f"zip entries: {len(names)}")
    top_dirs = sorted({n.split("/")[0] for n in zf.namelist() if "/" in n})
    print(f"top-level entries: {top_dirs[:30]}")

    # candidate dirs/files by class
    buckets: dict[str, list[str]] = {"Smoke_Fire_Alarm": [], "Doorbell": []}
    for n in names:
        c = classify(n)
        if c:
            buckets[c].append(n)
    for c, lst in buckets.items():
        print(f"{c}: {len(lst)} entries matching")
        for x in lst[:5]:
            print(f"   {x}")
    if not buckets["Smoke_Fire_Alarm"] and not buckets["Doorbell"]:
        # show distinct parent dirs so mapping can be diagnosed
        parents = sorted({"/".join(n.split("/")[:-1]) for n in names})
        print("parent dirs:")
        for p in parents[:80]:
            print(f"   {p}")
        sys.exit(2)

    # 4. extract matching entries
    rows = []
    for cls, lst in buckets.items():
        outdir = RAW / cls
        outdir.mkdir(parents=True, exist_ok=True)
        for n in lst:
            dest = outdir / Path(n).name
            if not dest.exists():
                dest.write_bytes(zf.read(n))
            try:
                info = sf.info(str(dest))
                dur, sr, ch = info.duration, info.samplerate, info.channels
                if dur <= 0 or dur > 60 or sr < 8000 or ch not in (1, 2):
                    raise ValueError(f"bad dur/sr/ch {dur}/{sr}/{ch}")
                status, notes = "ok", ""
            except Exception as e:  # noqa: BLE001
                dur = sr = ch = None
                status, notes = "invalid", str(e)[:120]
                print(f"  INVALID {dest.name}: {notes}")
            rows.append(
                dict(
                    source_dataset=f"zenodo_{RECORD}",
                    source_file=Path(n).name,
                    source_id=Path(n).stem,
                    original_label=Path(n).parts[-2] if len(Path(n).parts) > 1 else Path(n).stem,
                    project_class=cls,
                    local_path=str(dest.relative_to(ROOT)).replace("\\", "/"),
                    license=meta.get("metadata", {}).get("license", {}).get("id", "cc-by-4.0"),
                    source_url=url,
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
                    v_sha256=sha256(dest) if status == "ok" else None,
                    v_notes=notes,
                )
            )
    zf.close()

    # 5. manifest (full rewrite each run; script re-validates everything)
    out = RESULTS / "zenodo_3689288_manifest.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"wrote {out} rows={len(df)}")
    print(df.groupby(["project_class", "validation_status"]).size())


if __name__ == "__main__":
    main()
