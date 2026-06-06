# make_gt.py

import os
import argparse
import pandas as pd


BASE = os.path.dirname(os.path.abspath(__file__))

GT_FILE = os.path.join(
    BASE,
    "..",
    "data",
    "DC",
    "output",
    "ground_truth_labeling",
    "dc_objects_to_label.xlsx",
)

DEFAULT_OUT_DIR = os.path.join(
    BASE,
    "..",
    "data",
    "DC",
    "output",
    "ownershipAssignment",
    "with_ground_truth",
)

LABEL_COL = "manual_ground_truth_location"


def clean_string(x):
    if pd.isna(x):
        return ""

    return str(x).strip()


def normalize_for_compare(x):
    return clean_string(x).lower()


def is_none_label(x):
    return normalize_for_compare(x) in ["none", "nan", "null"]


def is_blank_label(x):
    return clean_string(x) == ""


def round_coord(x, digits=7):
    if pd.isna(x):
        return None
    return round(float(x), digits)


def make_match_key(df):
    df = df.copy()

    if "name" not in df.columns:
        df["name"] = ""

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError(
            "Cannot match because latitude/longitude columns are missing."
        )

    df["match_name"] = df["name"].astype(str).str.strip().str.lower()
    df["match_lat"] = df["latitude"].apply(round_coord)
    df["match_lon"] = df["longitude"].apply(round_coord)

    df["match_key"] = (
        df["match_name"]
        + "__"
        + df["match_lat"].astype(str)
        + "__"
        + df["match_lon"].astype(str)
    )

    return df


def infer_algorithm_name(predictions_path, user_algorithm=None):
    if user_algorithm is not None and clean_string(user_algorithm) != "":
        return clean_string(user_algorithm).upper()

    filename = os.path.basename(predictions_path).upper()

    known_algorithms = [
        "SEEDED_ITERATIVE",
        "SEEDED",
        "DBSCAN",
        "KMEANS",
        "BIRCH",
        "SPECTRAL",
    ]

    for algo in known_algorithms:
        if algo in filename:
            return algo

    return "ALGORITHM"


def load_manual_gt(include_none=False):
    gt = pd.read_excel(GT_FILE, keep_default_na=False)

    if LABEL_COL not in gt.columns:
        raise ValueError(f"Missing column in Excel: {LABEL_COL}")

    gt[LABEL_COL] = gt[LABEL_COL].astype(str).str.strip()

    reviewed = gt[~gt[LABEL_COL].apply(is_blank_label)].copy()

    reviewed["manual_is_none"] = reviewed[LABEL_COL].apply(is_none_label)

    if include_none:
        final_gt = reviewed.copy()
    else:
        # For old SV-style ClusteringMetrics, exclude None rows.
        final_gt = reviewed[reviewed["manual_is_none"] == False].copy()

    final_gt["ground_truth_location"] = final_gt[LABEL_COL]

    # keep the label as None only if include_none=True.
    final_gt.loc[final_gt["manual_is_none"], "ground_truth_location"] = "None"

    final_gt = make_match_key(final_gt)

    print("\nLoaded manual DC ground truth")
    print("Excel file:", GT_FILE)
    print("Total rows in Excel:", len(gt))
    print("Reviewed rows:", len(reviewed))
    print("Manual None rows:", reviewed["manual_is_none"].sum())
    print("Manual actual-location rows:", (~reviewed["manual_is_none"]).sum())

    if include_none:
        print("Using reviewed rows INCLUDING None:", len(final_gt))
    else:
        print("Using only actual-location rows:", len(final_gt))

    return final_gt


def load_predictions(pred_file):
    # Loads any algorithm 
    pred = pd.read_csv(pred_file)

    print("\nLoaded predictions")
    print("Prediction file:", pred_file)
    print("Rows:", len(pred))
    print("Columns:", list(pred.columns))

    if "ground_truth_location" in pred.columns:
        pred = pred.drop(columns=["ground_truth_location"])

    pred = make_match_key(pred)

    return pred


def get_gt_merge_columns(gt, merge_key):
    cols = [
        merge_key,
        "ground_truth_location",
        LABEL_COL,
        "manual_is_none",
    ]

    optional_cols = [
        "object_label_id",
        "raw_id",
        "box_id",
        "kind",
    ]

    for col in optional_cols:
        if col in gt.columns and col not in cols:
            cols.append(col)

    # Avoid duplicate merge key if optional cols include it.
    seen = set()
    final_cols = []

    for col in cols:
        if col in gt.columns and col not in seen:
            final_cols.append(col)
            seen.add(col)

    return final_cols


def merge_gt_with_predictions(pred, gt):
    # raw_id
    if "raw_id" in pred.columns and "raw_id" in gt.columns:
        cols = get_gt_merge_columns(gt, "raw_id")

        merged = pred.merge(
            gt[cols],
            on="raw_id",
            how="inner",
        )

        if len(merged) > 0:
            print("\nMerged using raw_id")
            return merged

    # object_label_id
    if "object_label_id" in pred.columns and "object_label_id" in gt.columns:
        cols = get_gt_merge_columns(gt, "object_label_id")

        merged = pred.merge(
            gt[cols],
            on="object_label_id",
            how="inner",
        )

        if len(merged) > 0:
            print("\nMerged using object_label_id")
            return merged

    # fallback key
    cols = get_gt_merge_columns(gt, "match_key")

    merged = pred.merge(
        gt[cols],
        on="match_key",
        how="inner",
    )

    print("\nMerged using fallback key: name + rounded latitude + rounded longitude")

    return merged


def find_col(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            return col

    return None


def ensure_predicted_location_column(df):
    df = df.copy()

    pred_col = find_col(
        df,
        [
            "predicted_location",
            "PredictedLocation",
            "predictedLocation",
            "predicted_locations",
            "assigned_location",
            "location_assignment",
            "owner_location",
            "location",
        ],
    )

    if pred_col is None:
        print("\nWARNING:")
        print("Could not find a predicted location column.")
        print("Your output file may not work with ClusteringMetrics.py.")
        print("Columns available:", list(df.columns))
        return df

    if pred_col != "predicted_location":
        df["predicted_location"] = df[pred_col]

    df["predicted_location"] = df["predicted_location"].apply(clean_string)

    return df


def make_output_filename(predictions_path, algorithm, include_none):
    algorithm = algorithm.upper()

    basename = os.path.basename(predictions_path)
    if "PredictedLocations" in basename:
        name = basename.replace(".csv", "")
        out_name = f"{algorithm}_DC_PredictedLocations"
        if "_PredictedLocations_" in name:
            suffix = name.split("_PredictedLocations_", 1)[1]
            out_name = f"{algorithm}_DC_PredictedLocations_{suffix}.csv"
        else:
            out_name = f"{algorithm}_DC_PredictedLocations.csv"

    else:
        out_name = f"{algorithm}_DC_PredictedLocations.csv"

    if include_none:
        out_name = out_name.replace(".csv", "_WITH_NONE.csv")

    return out_name


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--predictions",
        required=True,
        help="Path to any algorithm prediction CSV.",
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        default=None,
        help="Algorithm name. Example: SEEDED_ITERATIVE, DBSCAN, KMEANS, BIRCH, SPECTRAL. If omitted, inferred from filename.",
    )

    parser.add_argument(
        "--include_none",
        action="store_true",
        help="Include manual None rows too. Do not use this for normal old ClusteringMetrics location-name evaluation.",
    )

    parser.add_argument(
        "-o",
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for merged with-ground-truth CSV.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    algorithm = infer_algorithm_name(args.predictions, args.algorithm)

    print("\nAlgorithm:", algorithm)

    gt = load_manual_gt(include_none=args.include_none)
    pred = load_predictions(args.predictions)

    merged = merge_gt_with_predictions(pred, gt)
    merged = ensure_predicted_location_column(merged)

    print("\nMerge result")
    print("Rows with manual GT matched to predictions:", len(merged))

    if len(merged) == 0:
        print("\nERROR: No rows matched.")
        print("Check whether the prediction CSV has raw_id, object_label_id, or name/latitude/longitude.")
        return

    print("\nGround truth label counts:")
    print(merged["ground_truth_location"].value_counts().head(50))

    print("\nPredicted location counts in matched rows:")
    if "predicted_location" in merged.columns:
        print(merged["predicted_location"].value_counts(dropna=False).head(50))
    else:
        print("No predicted_location column found.")

    out_name = make_output_filename(
        predictions_path=args.predictions,
        algorithm=algorithm,
        include_none=args.include_none,
    )

    out_path = os.path.join(args.out_dir, out_name)

    merged.to_csv(out_path, index=False)

    print("\nSaved:")
    print(out_path)

    print("\nNow run:")
    print(f"python ClusteringMetrics.py -c {out_path}")


if __name__ == "__main__":
    main()