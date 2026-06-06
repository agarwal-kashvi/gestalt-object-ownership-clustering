import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

BASE_DIR = "../data/SV/output/ownershipAssignment/with_ground_truth"
OUT_DIR = "../imgs"

os.makedirs(OUT_DIR, exist_ok=True)

files = {
    "Seeded Iterative": "SEEDED_ITERATIVE_PredictedLocations_FT=0.0.csv",
    "KMeans": "KMEANS_PredictedLocations_FT=0.0.csv",
    "DBSCAN": "DBSCAN_PredictedLocations_FT=0.0.csv",
    "BIRCH": "BIRCH_PredictedLocations_N=311_T=0.25_BF=150.csv",
    "Spectral": "SPECTRAL_PredictedLocations_N=311_G=15.0.csv",
}

print("BASE_DIR exists?", os.path.exists(BASE_DIR))
print("OUT_DIR:", os.path.abspath(OUT_DIR))

for method, filename in files.items():
    path = os.path.join(BASE_DIR, filename)
    print("Checking:", path, os.path.exists(path))

for method, filename in files.items():
    path = os.path.join(BASE_DIR, filename)

    df = pd.read_csv(path)

    df = df[df["ground_truth_location"].notna()]
    df = df[df["ground_truth_location"] != "None"]
    df = df[df["predicted_location"].notna()]

    y_true = df["ground_truth_location"].astype(str)
    y_pred = df["predicted_location"].astype(str)

    labels = sorted(y_true.unique())

    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        xticks_rotation=45,
        cmap="Blues",
        ax=ax,
        colorbar=True,
    )

    ax.set_title(f"{method} Confusion Matrix")
    ax.set_xlabel("Predicted Location")
    ax.set_ylabel("True Location")

    plt.tight_layout()

    out_file = os.path.join(
        OUT_DIR,
        f"{method.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved: {out_file}")