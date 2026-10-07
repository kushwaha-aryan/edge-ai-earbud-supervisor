from huggingface_hub import hf_hub_download
import pandas as pd

repo = "Fhrozen/FSD50k"

# Download metadata
dev_path = hf_hub_download(
    repo_id=repo,
    filename="labels/dev.csv",
    repo_type="dataset"
)

eval_path = hf_hub_download(
    repo_id=repo,
    filename="labels/eval.csv",
    repo_type="dataset"
)

vocab_path = hf_hub_download(
    repo_id=repo,
    filename="labels/vocabulary.csv",
    repo_type="dataset"
)

dev = pd.read_csv(dev_path)
eval_df = pd.read_csv(eval_path)

vocab = pd.read_csv(
    vocab_path,
    header=None,
    names=["id", "label", "mid"]
)

print("\n=== FSD50K Vocabulary ===")
print("Classes:", len(vocab))

# Combine train/dev metadata + eval
all_labels = pd.concat([
    dev[["fname", "labels"]],
    eval_df[["fname", "labels"]]
], ignore_index=True)

# Convert each comma-separated label string into individual labels
label_counts = {}

for labels in all_labels["labels"]:
    for label in str(labels).split(","):
        label = label.strip()
        label_counts[label] = label_counts.get(label, 0) + 1

# Candidate classes for our project
targets = [
    "Siren",
    "Vehicle_horn",
    "Alarm",
    "Dog",
    "Glass",
    "Gunshot",
    "Crying",
    "Shout",
    "Footsteps",
    "Knock",
    "Engine",
    "Motorcycle",
    "Train",
    "Aircraft",
    "Drill",
    "Jackhammer",
]

print("\n=== Candidate Class Counts ===")

for target in targets:
    print(f"{target:20} {label_counts.get(target, 0)}")