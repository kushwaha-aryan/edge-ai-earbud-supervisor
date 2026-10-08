"""Phase 1.5 — diagnostic spectrograms (validation only, read-only over audio).

For each project class, picks 4 clips (shortest, median, longest,
lowest-energy) and writes one figure with waveform + spectrogram panels to
scripts/results/dataset_validation/spectrograms/<Class>.png.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import spectrogram  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEALTH = ROOT / "scripts" / "results" / "dataset_validation" / "file_health.csv"
OUTDIR = ROOT / "scripts" / "results" / "dataset_validation" / "spectrograms"


def pick_clips(g: pd.DataFrame) -> list[pd.Series]:
    g = g.sort_values("duration")
    picks = [g.iloc[0], g.iloc[len(g) // 2], g.iloc[-1]]
    low = g.sort_values("rms_db").iloc[0]
    if not any(p["local_path"] == low["local_path"] for p in picks):
        picks.append(low)
    return picks


def panel(ax_wav, ax_spec, row: pd.Series) -> None:
    path = ROOT / row["local_path"]
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    flat = data.mean(axis=1) if data.ndim > 1 else data
    x = np.arange(flat.shape[0]) / sr

    ax_wav.plot(x, flat, lw=0.4, color="#1f77b4")
    ax_wav.set_ylim(-1.05, 1.05)
    peak = float(np.max(np.abs(flat))) if flat.size else 0.0
    if peak >= 0.999:
        n_clip = float(np.mean(np.abs(flat) >= 0.999))
        ax_wav.axhline(0.999, color="r", lw=0.5, ls=":")
        ax_wav.axhline(-0.999, color="r", lw=0.5, ls=":")
    else:
        n_clip = 0.0
    ax_wav.set_ylabel("amp")

    f, t, sxx = spectrogram(flat, fs=sr, nperseg=min(2048, max(256, sr // 20)))
    sdb = 10.0 * np.log10(sxx + 1e-12)
    ax_spec.pcolormesh(t, f, sdb, shading="auto", cmap="magma",
                       vmin=sdb.max() - 80.0, vmax=sdb.max())
    ax_spec.set_ylabel("Hz")
    ax_spec.set_xlabel("s")

    flags = str(row["flags"]) if isinstance(row["flags"], str) else ""
    title = (
        f"{row['source_dataset']} | {row['project_class']} | dur {row['duration']:.2f}s"
        f" | {int(sr)}Hz {int(row['channels'])}ch | RMS {row['rms_db']:.1f}dB"
        f" | peak {peak:.3f} | clip {n_clip*100:.2f}%"
    )
    if flags:
        title += f" | {flags}"
    ax_wav.set_title(title, fontsize=8)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(HEALTH)
    h["flags"] = h["flags"].fillna("")
    made = []
    for cls, g in h.groupby("project_class"):
        picks = pick_clips(g)
        fig, axes = plt.subplots(
            len(picks), 2, figsize=(11, 2.1 * len(picks)),
            squeeze=False, constrained_layout=True)
        for i, row in enumerate(picks):
            panel(axes[i][0], axes[i][1], row)
        fig.suptitle(f"{cls} — diagnostic samples "
                     f"({len(g)} clips: shortest / median / longest / lowest-energy)",
                     fontsize=11)
        out = OUTDIR / f"{cls}.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        made.append(out.name)
        print("wrote", out.name)
    print("total:", len(made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
