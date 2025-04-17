import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Load the data
airports_file = "../../data/airports.dat"
airports_cols = [
    "AirportID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude",
    "Altitude", "Timezone", "DST", "TzDatabaseTimeZone", "Type", "Source"
]
airports_df = pd.read_csv(airports_file, header=None, names=airports_cols)

# Clean data
airports_df_clean = airports_df.dropna(subset=["Latitude", "Longitude", "IATA"])

# Extract features for clustering
X = airports_df_clean[["Latitude", "Longitude"]].values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=8, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
airports_df_clean['Cluster'] = clusters

# Create a map to visualize the clusters
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

# Plot airports with cluster colors
for cluster_id in range(kmeans.n_clusters):
    cluster_points = airports_df_clean[airports_df_clean['Cluster'] == cluster_id]
    ax.scatter(
        cluster_points["Longitude"],
        cluster_points["Latitude"],
        s=10,
        label=f'Cluster {cluster_id}',
        transform=ccrs.PlateCarree()
    )

# Add gridlines
gridlines = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gridlines.top_labels = False
gridlines.right_labels = False

plt.title("K-means Clustering of Airports by Geographic Location", fontsize=16)
plt.legend(loc='lower left')
plt.show()