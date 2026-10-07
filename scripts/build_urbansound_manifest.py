from pathlib import Path
import pandas as pd

DATASET_ROOT = Path(
    r"C:\Users\BITPATNA\.cache\kagglehub\datasets\chrisfilo\urbansound8k\versions\1"
)

METADATA = DATASET_ROOT / "UrbanSound8K.csv"
OUTPUT = Path("results/urbansound8k_manifest.csv")

CLASS_MAP = {
    "car_horn": "Vehicle_Horn",
    "dog_bark": "Dog_Bark",
    "drilling": "Drilling",
    "engine_idling": "Car_Engine",
    "gun_shot": "Gunshot",
    "jackhammer": "Jackhammer",
    "siren": "Siren",
}

df = pd.read_csv(METADATA)

selected = df[df["class"].isin(CLASS_MAP)].copy()

selected["source_dataset"] = "UrbanSound8K"
selected["project_class"] = selected["class"].map(CLASS_MAP)

selected["audio_path"] = selected.apply(
    lambda row: str(DATASET_ROOT / f"fold{row['fold']}" / row["slice_file_name"]),
    axis=1,
)

manifest = selected[
    [
        "source_dataset",
        "audio_path",
        "slice_file_name",
        "fsID",
        "start",
        "end",
        "fold",
        "class",
        "project_class",
    ]
].copy()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(OUTPUT, index=False)

print("UrbanSound8K manifest created.")
print("Rows:", len(manifest))
print("\nProject classes:")
print(manifest["project_class"].value_counts().sort_index())
print("\nSaved to:", OUTPUT.resolve())