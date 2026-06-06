import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
#BASE_DIR = "../data/DC/output/ownershipAssignment/with_ground_truth"
BASE_DIR = "../data/sv/output/ownershipAssignment/with_ground_truth"
OUT_DIR = "../imgs"
os.makedirs(OUT_DIR, exist_ok=True)

files = {
    "Seeded Iterative": "SEEDED_ITERATIVE_PredictedLocations_FT=0.0.csv",
    "KMeans": "KMEANS_DC_PredictedLocations_FT=0.0.csv",
    "DBSCAN": "DBSCAN_DC_PredictedLocations_FT=0.0.csv",
}
target_locations = [
    "Baruch Bench of Inspiration",
    "Society of the Cincinnati",
    "White House Deli",
    "EMeRG (Emergency Medical Response Group)",
    "Red Cross",
    "Alexander Robey Shepherd",
    "JoS. A. Bank",
    "Commodore John Barry Statue",
    "Jamya Williams",
    "Ammathar Thai Cuisine",
    "Foundry United Methodist Church Day Care Center",
    "White House Visitor Center",
    "White House Peace Vigil",
    "Beefsteak",
    "Peet's Coffee",
    "Black Lives Matter Plaza Northwest",
    "Plan of the White House",
    "Embassy of Indonesia",
    "Silver Cycles Dc",
    "Nelson Mandela Memorial",
    "Laborers International Union of North America",
    "Embassy of the Dominican Republic",
    "Embassy of the Republic of Congo",
    "Embassy of Hungary",
    "Floor Frame",
    "Cafe Aria",
    "Mikko",
    "Malbec",
    "Peabody Conservatory of the John Hopkins University",
    "7-Eleven",
    "Miss Pixie's",
    "Avenue Title Group",
    "Pearl Dive Oyster Palace",
    "FedEx Office",
    "Great Wall Szechuan House",
    "Church of Scientology",
    "Dupont Circle",
    "Mount Olivet Lutheran Church",
    "Citibank",
]
'''    "Ali's Vineyard",
    "Little River Winery and Café",
    "Faber Vineyard",
    "Ugly Duckling Wines",
    "Oakover Grounds",
    "Lancaster Wines",'''

'''location_colors = {
    "Ali's Vineyard": "#4363d8",                 
    "Little River Winery and Café": "#911eb4",  
    "Faber Vineyard": "#42d4f4",                
    "Ugly Duckling Wines": "#f58231",           
    "Oakover Grounds": "#3cb44b",               
    "Lancaster Wines": "#e6194B",               
}'''
location_colors = {
    "Baruch Bench of Inspiration": "#1f77b4",
    "Society of the Cincinnati": "#ff7f0e",
    "White House Deli": "#2ca02c",
    "EMeRG (Emergency Medical Response Group)": "#d62728",
    "Red Cross": "#9467bd",
    "Alexander Robey Shepherd": "#8c564b",
    "JoS. A. Bank": "#e377c2",
    "Commodore John Barry Statue": "#7f7f7f",
    "Jamya Williams": "#bcbd22",
    "Ammathar Thai Cuisine": "#17becf",
    "Foundry United Methodist Church Day Care Center": "#393b79",
    "White House Visitor Center": "#637939",
    "White House Peace Vigil": "#8c6d31",
    "Beefsteak": "#843c39",
    "Peet's Coffee": "#7b4173",
    "Black Lives Matter Plaza Northwest": "#3182bd",
    "Plan of the White House": "#e6550d",
    "Embassy of Indonesia": "#31a354",
    "Silver Cycles Dc": "#756bb1",
    "Nelson Mandela Memorial": "#636363",
    "Laborers International Union of North America": "#6baed6",
    "Embassy of the Dominican Republic": "#fd8d3c",
    "Embassy of the Republic of Congo": "#74c476",
    "Embassy of Hungary": "#9e9ac8",
    "Floor Frame": "#969696",
    "Cafe Aria": "#9ecae1",
    "Mikko": "#fdae6b",
    "Malbec": "#a1d99b",
    "Peabody Conservatory of the John Hopkins University": "#bcbddc",
    "7-Eleven": "#bdbdbd",
    "Miss Pixie's": "#08519c",
    "Avenue Title Group": "#a63603",
    "Pearl Dive Oyster Palace": "#006d2c",
    "FedEx Office": "#54278f",
    "Great Wall Szechuan House": "#252525",
    "Church of Scientology": "#6b6ecf",
    "Dupont Circle": "#b5cf6b",
    "Mount Olivet Lutheran Church": "#e7ba52",
    "Citibank": "#ce6dbd",
}

for method, filename in files.items():
    df = pd.read_csv(os.path.join(BASE_DIR, filename))

    df["plot_group"] = df["predicted_location"].where(
        df["predicted_location"].isin(target_locations),
        "None / Other"
    )

    none_df = df[df["plot_group"] == "None / Other"].copy()
    colored_df = df[df["plot_group"] != "None / Other"].copy()

    fig = px.scatter_mapbox(
        none_df,
        lat="latitude",
        lon="longitude",
        hover_name="name",
        hover_data=["predicted_location", "ground_truth_location"],
        color_discrete_sequence=["gray"],
        zoom=11,
        height=750,
        title=f"{method} Cluster Assignment Map with Background"
    )

    colored_fig = px.scatter_mapbox(
        colored_df,
        lat="latitude",
        lon="longitude",
        color="plot_group",
        hover_name="name",
        color_discrete_map=location_colors,
        hover_data=["predicted_location", "ground_truth_location"],
        zoom=11,
        height=750,
    )

    for trace in colored_fig.data:
        fig.add_trace(trace)

    fig.update_traces(marker=dict(size=7, opacity=0.45), selector=dict(mode="markers"))

    for trace in fig.data:
        if trace.name != "":
            trace.marker.size = 9
            trace.marker.opacity = 0.85

    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 45, "l": 0, "b": 0},
        legend_title_text="Predicted Location"
    )

    out_file = os.path.join(
        OUT_DIR,
        f"{method.lower().replace(' ', '_')}_dc_gt_interactive_map.html"
    )

    fig.write_html(out_file)
    print("Saved:", out_file)