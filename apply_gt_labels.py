
import os
import pandas as pd

LABEL_FILE = "../data/DC/output/ground_truth_labeling/dc_objects_to_label.xlsx"
OUT_DIR = "../data/DC/output/ground_truth_labeling"

REVIEWED_OUT = os.path.join(OUT_DIR, "dc_reviewed_objects.csv")
NONE_OUT = os.path.join(OUT_DIR, "dc_none_objects.csv")
LOCATION_LABELS_OUT = os.path.join(OUT_DIR, "dc_labeled_objects_only.csv")

label_col = "manual_ground_truth_location"

# Without keep_default_na=False, pandas converts the text "None" into NaN.
df = pd.read_excel(LABEL_FILE, keep_default_na=False)

print("Total rows:", len(df))

df[label_col] = df[label_col].astype("string").str.strip()

reviewed_mask = df[label_col].notna() & (df[label_col] != "")
reviewed_df = df[reviewed_mask].copy()

none_mask = reviewed_df[label_col].str.lower() == "none"
none_df = reviewed_df[none_mask].copy()

location_labeled_df = reviewed_df[~none_mask].copy()
location_labeled_df["ground_truth_location"] = location_labeled_df[label_col]

reviewed_df.to_csv(REVIEWED_OUT, index=False)
none_df.to_csv(NONE_OUT, index=False)
location_labeled_df.to_csv(LOCATION_LABELS_OUT, index=False)

print("\nTop manual label values:")
print(df[label_col].value_counts(dropna=False).head(30))

print("\nTotal rows in labeling sheet:", len(df))
print("Rows manually reviewed:", len(reviewed_df))
print("Rows marked None:", len(none_df))
print("Rows assigned to actual locations:", len(location_labeled_df))

print("\nSaved reviewed rows to:")
print(REVIEWED_OUT)

print("\nSaved None rows to:")
print(NONE_OUT)

print("\nSaved actual location-labeled rows to:")
print(LOCATION_LABELS_OUT)

print("\nPreview of None rows:")
print(none_df[["object_label_id", "name", label_col]].head(20))

print("\nPreview of actual location labels:")
print(location_labeled_df[["object_label_id", "name", label_col]].head(20))