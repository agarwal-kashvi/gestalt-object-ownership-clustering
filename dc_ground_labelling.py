import os
import json
import pandas as pd
import matplotlib.pyplot as plt

import os

BASE = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE, "..", "data", "DC", "output", "dataCollection")
OUT_DIR = os.path.join(BASE, "..", "data", "DC", "output", "ground_truth_labeling")
os.makedirs(OUT_DIR, exist_ok=True)
print("SCRIPT BASE:", BASE)
print("INPUT_DIR:", INPUT_DIR)
print("OUT_DIR:", OUT_DIR)
# bounding box
BBOX = {
    "min_lon": -77.0500,
    "max_lon": -77.0400,
    "min_lat": 38.9000,
    "max_lat": 38.9100,
}

def load_jsons(prefix):
    data = {}
    for file in os.listdir(INPUT_DIR):
        if file.startswith(prefix) and file.endswith(".json"):
            path = os.path.join(INPUT_DIR, file)
            print("Loading:", path)
            with open(path, "r", encoding="utf-8") as f:
                data.update(json.load(f))
    return data

def in_bbox(row):
    return (
        BBOX["min_lon"] <= row["longitude"] <= BBOX["max_lon"]
        and BBOX["min_lat"] <= row["latitude"] <= BBOX["max_lat"]
    )

def dict_to_df(data, kind):
    rows = []
    for key, val in data.items():
        if "latitude" in val and "longitude" in val:
            rows.append({
                "id": key,
                "name": val.get("name", key),
                "latitude": val["latitude"],
                "longitude": val["longitude"],
                "source": val.get("source", ""),
                "ground_truth_location": val.get("ground_truth_location", None),
                "kind": kind,
            })
    return pd.DataFrame(rows)

objects = load_jsons("objects")
locations = load_jsons("locations")

objects_df = dict_to_df(objects, "object")
locations_df = dict_to_df(locations, "location")

objects_small = objects_df[objects_df.apply(in_bbox, axis=1)].copy()
locations_small = locations_df[locations_df.apply(in_bbox, axis=1)].copy()

objects_small = objects_small.reset_index(drop=True)
locations_small = locations_small.reset_index(drop=True)

objects_small["label_id"] = ["O" + str(i) for i in range(len(objects_small))]
locations_small["label_id"] = ["L" + str(i) for i in range(len(locations_small))]

print("Objects in bbox:", len(objects_small))
print("Locations in bbox:", len(locations_small))

objects_small.to_csv(os.path.join(OUT_DIR, "dc_objects_to_label.csv"), index=False)
locations_small.to_csv(os.path.join(OUT_DIR, "dc_locations_reference.csv"), index=False)

print("Saved:")
print(os.path.join(OUT_DIR, "dc_objects_to_label.csv"))
print(os.path.join(OUT_DIR, "dc_locations_reference.csv"))
