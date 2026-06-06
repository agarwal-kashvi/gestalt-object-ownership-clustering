import os
import json
import math
import pandas as pd
import plotly.graph_objects as go
print("SCRIPT STARTED")

BASE = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE, "..", "data", "DC", "output", "dataCollection")
OUT_DIR = os.path.join(BASE, "..", "data", "DC", "output", "ground_truth_labeling")

os.makedirs(OUT_DIR, exist_ok=True)

REGION = {
    "min_lon": -77.0500,
    "max_lon": -77.0300,
    "min_lat": 38.8950,
    "max_lat": 38.9150,
}

N_ROWS = 5
N_COLS = 5

def load_jsons(prefix):
    data = {}

    for file in os.listdir(INPUT_DIR):
        if file.startswith(prefix) and file.endswith(".json"):
            path = os.path.join(INPUT_DIR, file)
            print("Loading:", path)

            with open(path, "r", encoding="utf-8") as f:
                data.update(json.load(f))

    return data


def dict_to_df(data, kind):
    rows = []

    for key, val in data.items():
        if "latitude" not in val or "longitude" not in val:
            continue

        rows.append({
            "raw_id": key,
            "name": val.get("name", key),
            "latitude": val["latitude"],
            "longitude": val["longitude"],
            "source": val.get("source", ""),
            "ground_truth_location": val.get("ground_truth_location", None),
            "kind": kind,
        })

    return pd.DataFrame(rows)


def filter_bbox(df, bbox):
    return df[
        (df["longitude"] >= bbox["min_lon"]) &
        (df["longitude"] <= bbox["max_lon"]) &
        (df["latitude"] >= bbox["min_lat"]) &
        (df["latitude"] <= bbox["max_lat"])
    ].copy()


def make_subboxes(region, n_rows, n_cols):
    boxes = []

    lon_step = (region["max_lon"] - region["min_lon"]) / n_cols
    lat_step = (region["max_lat"] - region["min_lat"]) / n_rows

    box_id = 0

    for r in range(n_rows):
        for c in range(n_cols):
            min_lon = region["min_lon"] + c * lon_step
            max_lon = min_lon + lon_step

            min_lat = region["min_lat"] + r * lat_step
            max_lat = min_lat + lat_step

            boxes.append({
                "box_id": f"B{box_id}",
                "min_lon": min_lon,
                "max_lon": max_lon,
                "min_lat": min_lat,
                "max_lat": max_lat,
            })

            box_id += 1

    return boxes


objects = load_jsons("objects")
locations = load_jsons("locations")

objects_df = dict_to_df(objects, "object")
locations_df = dict_to_df(locations, "location")

subboxes = make_subboxes(REGION, N_ROWS, N_COLS)

all_objects_to_label = []
all_locations_reference = []

for bbox in subboxes:
    box_id = bbox["box_id"]

    obj_sub = filter_bbox(objects_df, bbox).reset_index(drop=True)
    loc_sub = filter_bbox(locations_df, bbox).reset_index(drop=True)

    obj_sub["box_id"] = box_id
    loc_sub["box_id"] = box_id

    obj_sub["object_label_id"] = [f"{box_id}_O{i}" for i in range(len(obj_sub))]
    loc_sub["location_label_id"] = [f"{box_id}_L{i}" for i in range(len(loc_sub))]

    obj_sub["manual_ground_truth_location"] = ""

    all_objects_to_label.append(obj_sub)
    all_locations_reference.append(loc_sub)

    if len(obj_sub) == 0 and len(loc_sub) == 0:
        print(f"{box_id}: empty, skipping map.")
        continue

    center_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
    center_lon = (bbox["min_lon"] + bbox["max_lon"]) / 2

    fig = go.Figure()

    fig.add_trace(
        go.Scattermapbox(
            lat=obj_sub["latitude"],
            lon=obj_sub["longitude"],
            mode="markers+text",
            marker=dict(size=8, color="blue", opacity=0.75),
            text=obj_sub["object_label_id"],
            textposition="top right",
            customdata=obj_sub[["name", "source", "ground_truth_location"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Object: %{customdata[0]}<br>"
                "Existing GT: %{customdata[2]}<br>"
                "Source: %{customdata[1]}<br>"
                "Lat: %{lat}<br>"
                "Lon: %{lon}<extra></extra>"
            ),
            name="Objects to Label"
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=loc_sub["latitude"],
            lon=loc_sub["longitude"],
            mode="markers+text",
            marker=dict(size=13, color="red", opacity=0.95),
            text=loc_sub["location_label_id"] + ": " + loc_sub["name"].astype(str),
            textposition="top right",
            customdata=loc_sub[["name", "raw_id"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Location: %{customdata[0]}<br>"
                "Raw ID: %{customdata[1]}<br>"
                "Lat: %{lat}<br>"
                "Lon: %{lon}<extra></extra>"
            ),
            name="Candidate Locations"
        )
    )

    fig.update_layout(
        title=f"DC Ground Truth Labeling Map - {box_id}",
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=16
        ),
        height=800,
        margin=dict(r=0, t=50, l=0, b=0),
        legend_title_text="Layer"
    )

    html_path = os.path.join(OUT_DIR, f"dc_labeling_{box_id}.html")
    fig.write_html(html_path)

    print(
        f"{box_id}: objects={len(obj_sub)}, locations={len(loc_sub)}, saved={html_path}"
    )


objects_out = pd.concat(all_objects_to_label, ignore_index=True)
locations_out = pd.concat(all_locations_reference, ignore_index=True)

objects_csv = os.path.join(OUT_DIR, "dc_objects_to_label.csv")
locations_csv = os.path.join(OUT_DIR, "dc_locations_reference.csv")
objects_xlsx = os.path.join(OUT_DIR, "dc_objects_to_label.xlsx")

objects_out.to_csv(objects_csv, index=False)
locations_out.to_csv(locations_csv, index=False)

try:
    objects_out.to_excel(objects_xlsx, index=False)
except Exception as e:
    print("Could not write Excel file. CSV was still saved.")
    print(e)

print("\nSaved labeling files:")
print(objects_csv)
print(locations_csv)
print(objects_xlsx)