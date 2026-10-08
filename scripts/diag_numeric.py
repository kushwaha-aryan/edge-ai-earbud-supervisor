"""Phase 1.5 — numeric audio diagnostics (validation only).

Same clip picks as make_diag_spectrograms.py (shortest/median/longest/
lowest-energy per class). Computes per-clip: dc_offset, silence_ratio
(1024-frame hops below -60 dBFS), zcr, mean spectral centroid, crest
factor. Writes scripts/results/dataset_validation/diagnostics.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEALTH = ROOT / "scripts" / "results" / "dataset_validation" / "file_health.csv"
OUT = ROOT / "scripts" / "results" / "dataset_validation" / "diagnostics.csv"
HOP = 1024
SILENCE_DB = -60.0


def picks_for(g: pd.DataFrame) -> list[pd.Series]:
    g = g.sort_values("duration")
    p = [g.iloc[0], g.iloc[len(g) // 2], g.iloc[-1]]
    low = g.sort_values("rms_db").iloc[0]
    if not any(x["local_path"] == low["local_path"] for x in p):
        p.append(low)
    return p


def metrics(path_str: str):
    data, sr = sf.read(str(ROOT / path_str), dtype="float32",
                       always_2d=False)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    n = mono.shape[0]
    dc = float(np.mean(mono))
    peak = float(np.max(np.abs(mono))) if n else 0.0
    rms = float(np.sqrt(np.mean(mono * mono))) if n else 0.0
    crest = peak / rms if rms > 0 else float("inf")
    # frame energy
    nfr = n // HOP
    if nfr:
        fr = mono[: nfr * HOP].reshape(nfr, HOP)
        frms = np.sqrt((fr * fr).mean(axis=1)) + 1e-12
        frdb = 20 * np.log10(frms)
        sil = float(np.mean(frdb <= SILENCE_DB))
        zcr = float(np.mean(np.abs(np.diff(np.sign(fr), axis=1)).mean(axis=1)) / 2.0)
    else:
        sil, zcr = 1.0, 0.0
    # spectral centroid on a single FFT of the whole clip
    spec = np.abs(np.fft.rfft(mono * np.hanning(n))) if n else np.array([0.0])
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    centroid = float((freqs * spec).sum() / (spec.sum() + 1e-12))
    return dc, sil, zcr, centroid, crest


def main() -> int:
    h = pd.read_csv(HEALTH)
    h["flags"] = h["flags"].fillna("")
    rows = []
    for cls, g in h.groupby("project_class"):
        for role, row in zip(
                ["shortest", "median", "longest", "lowest_energy"],
                picks_for(g)):
            dc, sil, zcr, cen, crest = metrics(row["local_path"])
            rows.append({
                "project_class": cls, "pick": role,
                "source_dataset": row["source_dataset"],
                "local_path": row["local_path"],
                "duration": row["duration"], "rms_db": row["rms_db"],
                "dc_offset": round(dc, 5),
                "silence_ratio": round(sil, 4),
                "zcr": round(zcr, 4),
                "spectral_centroid_hz": round(cen, 1),
                "crest_factor": round(crest, 2) if np.isfinite(crest) else None,
                "flags": row["flags"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
