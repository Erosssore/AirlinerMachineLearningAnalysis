import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# File paths using the script directory as a reference
airports_file = os.path.join(script_dir, "data", "airports.dat")
routes_file = os.path.join(script_dir, "data", "routes.dat")

# If files are still in a parent directory, adjust the path accordingly
if not os.path.exists(airports_file):
    airports_file = os.path.join(script_dir, "..", "data", "airports.dat")
    routes_file = os.path.join(script_dir, "..", "data", "routes.dat")

# Check if files exist
if not os.path.exists(airports_file):
    raise FileNotFoundError(f"Cannot find airports data file. Tried: {airports_file}")
if not os.path.exists(routes_file):
    raise FileNotFoundError(f"Cannot find routes data file. Tried: {routes_file}")

# Load the data with the appropriate headers for airports
airports_cols = [
    "AirportID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude",
    "Altitude", "Timezone", "DST", "TzDatabaseTimeZone", "Type", "Source"
]
airports_df = pd.read_csv(airports_file, header=None, names=airports_cols)

# Load routes data
routes_cols = [
    "Airline", "AirlineID", "SourceAirport", "SourceAirportID",
    "DestinationAirport", "DestinationAirportID", "Codeshare",
    "Stops", "Equipment"
]
routes_df = pd.read_csv(routes_file, header=None, names=routes_cols)

# Clean airport data
airports_df_clean = airports_df.dropna(subset=["Latitude", "Longitude", "IATA"])

# Create figure with a specific projection
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.Robinson())

# Add map features
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS, linewidth=0.5)

# Add gridlines
gridlines = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gridlines.top_labels = False
gridlines.right_labels = False

# Plot the airports
ax.scatter(
    airports_df_clean["Longitude"],
    airports_df_clean["Latitude"],
    s=3,  # Small dots
    c="red",  # Red is visible against the map
    alpha=0.7,
    marker="o",
    transform=ccrs.PlateCarree()  # Specify the coordinate system of the data
)

# Add a title
plt.title("Global Airport Locations", fontsize=16)

# Optional: Add a legend
plt.legend(['Airports'], loc='lower right')

# Save the figure (optional)
plt.savefig("airport_locations_cartopy.png", dpi=300, bbox_inches="tight")

# Display the plot
plt.show()

# Let's create a second visualization: airports by country (top 15)
country_counts = airports_df["Country"].value_counts().head(15)

plt.figure(figsize=(14, 8))
country_counts.plot(kind="bar", color="skyblue")
plt.title("Top 15 Countries by Number of Airports", fontsize=15)
plt.xlabel("Country")
plt.ylabel("Number of Airports")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Display the plot
plt.show()

# Let's create a third visualization: busiest airports (most connections)
source_counts = routes_df["SourceAirport"].value_counts()
dest_counts = routes_df["DestinationAirport"].value_counts()

# Sum up connections (both incoming and outgoing)
total_connections = source_counts.add(dest_counts, fill_value=0)
top_airports = total_connections.sort_values(ascending=False).head(20)

# Get airport names for the top connected airports
top_airports_data = []
for airport_code in top_airports.index:
    airport_info = airports_df[airports_df["IATA"] == airport_code]
    if not airport_info.empty:
        name = airport_info.iloc[0]["Name"]
        city = airport_info.iloc[0]["City"]
        country = airport_info.iloc[0]["Country"]
        connections = top_airports[airport_code]
        top_airports_data.append({
            "IATA": airport_code,
            "Name": name,
            "City": city,
            "Country": country,
            "Connections": connections
        })

top_airports_df = pd.DataFrame(top_airports_data)

plt.figure(figsize=(15, 10))
bars = plt.barh(top_airports_df["IATA"], top_airports_df["Connections"], color="coral")
plt.title("Top 20 Airports by Total Route Connections", fontsize=15)
plt.xlabel("Number of Connections (Routes)")
plt.ylabel("Airport Code")
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Add airport names as annotations
for i, bar in enumerate(bars):
    plt.text(
        bar.get_width() + 5,
        bar.get_y() + bar.get_height() / 2,
        f"{top_airports_df.iloc[i]['City']}, {top_airports_df.iloc[i]['Country']}",
        va="center"
    )

plt.tight_layout()

# Display the plot
plt.show()
