from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd

REPO = "Fhrozen/FSD50k"

# Candidate project classes we want to investigate in FSD50K.
CANDIDATES = [
    "Alarm",
    "Aircraft",
    "Motorcycle",
    "Train",
    "Shout",
    "Knock",
    "Glass",
    "Dog",
    "Engine",
    "Drill",
    "Siren",
    "Vehicle_horn",
    "Gunshot",
    "Footsteps",
    "Crying",
    "Jackhammer",
]

# Download metadata/labels only.
vocab_path = hf_hub_download(
    repo_id=REPO,
    filename="labels/vocabulary.csv",
    repo_type="dataset",
)

dev_path = hf_hub_download(
    repo_id=REPO,
    filename="labels/dev.csv",
    repo_type="dataset",
)

eval_path = hf_hub_download(
    repo_id=REPO,
    filename="labels/eval.csv",
    repo_type="dataset",
)

# vocabulary.csv has no header.
vocab = pd.read_csv(
    vocab_path,
    header=None,
    names=["id", "label", "mid"],
)

dev = pd.read_csv(dev_path)
eval_df = pd.read_csv(eval_path)

all_labels = pd.concat(
    [dev[["fname", "labels"]], eval_df[["fname", "labels"]]],
    ignore_index=True,
)

print("\n=== FSD50K candidate vocabulary ===")

available = vocab[vocab["label"].isin(CANDIDATES)].copy()

print(available[["label", "mid"]].to_string(index=False))

print("\n=== Candidate clip counts ===")

counts = {}

for label in CANDIDATES:
    mask = all_labels["labels"].fillna("").apply(
        lambda x: label in [item.strip() for item in x.split(",")]
    )

    counts[label] = int(mask.sum())

result = pd.DataFrame(
    list(counts.items()),
    columns=["label", "clips"],
).sort_values("clips", ascending=False)

print(result.to_string(index=False))

print("\n=== All FSD50K vocabulary classes ===")
print(vocab["label"].to_string(index=False))