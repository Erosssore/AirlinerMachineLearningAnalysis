import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import tqdm


def load_and_preprocess_data():
    """Load and preprocess airport and route data"""
    # Load data
    airports_file = "../../data/airports.dat"
    routes_file = "../../data/routes.dat"

    # Load the data with the appropriate headers
    airports_cols = [
        "AirportID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude",
        "Altitude", "Timezone", "DST", "TzDatabaseTimeZone", "Type", "Source"
    ]
    airports_df = pd.read_csv(airports_file, header=None, names=airports_cols)

    routes_cols = [
        "Airline", "AirlineID", "SourceAirport", "SourceAirportID",
        "DestinationAirport", "DestinationAirportID", "Codeshare",
        "Stops", "Equipment"
    ]
    routes_df = pd.read_csv(routes_file, header=None, names=routes_cols)

    # Convert ID columns to numeric, handling any non-numeric values
    routes_df['SourceAirportID'] = pd.to_numeric(routes_df['SourceAirportID'], errors='coerce')
    routes_df['DestinationAirportID'] = pd.to_numeric(routes_df['DestinationAirportID'], errors='coerce')

    # Drop rows with missing airport IDs
    routes_df = routes_df.dropna(subset=['SourceAirportID', 'DestinationAirportID'])

    # Ensure consistent integer types
    routes_df['SourceAirportID'] = routes_df['SourceAirportID'].astype(int)
    routes_df['DestinationAirportID'] = routes_df['DestinationAirportID'].astype(int)

    # Merge airport details with routes
    routes_with_src = routes_df.merge(
        airports_df[["AirportID", "Latitude", "Longitude", "Country", "IATA"]],
        left_on="SourceAirportID",
        right_on="AirportID",
        suffixes=("", "_src")
    )

    routes_with_both = routes_with_src.merge(
        airports_df[["AirportID", "Latitude", "Longitude", "Country", "IATA"]],
        left_on="DestinationAirportID",
        right_on="AirportID",
        suffixes=("_src", "_dest")
    )

    # Calculate distance between airports (haversine formula)
    def haversine_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r

    routes_with_both['Distance'] = routes_with_both.apply(
        lambda row: haversine_distance(
            row['Latitude_src'], row['Longitude_src'],
            row['Latitude_dest'], row['Longitude_dest']
        ),
        axis=1
    )

    # Add more features
    routes_with_both['SameCountry'] = (routes_with_both['Country_src'] == routes_with_both['Country_dest']).astype(int)
    routes_with_both['LatDiff'] = abs(routes_with_both['Latitude_src'] - routes_with_both['Latitude_dest'])
    routes_with_both['LonDiff'] = abs(routes_with_both['Longitude_src'] - routes_with_both['Longitude_dest'])

    # Add continent information
    continents = {
        'United States': 'North America',
        'Canada': 'North America',
        'Mexico': 'North America',
        'Germany': 'Europe',
        'France': 'Europe',
        'United Kingdom': 'Europe',
        'Italy': 'Europe',
        'Spain': 'Europe',
        'China': 'Asia',
        'Japan': 'Asia',
        'India': 'Asia',
        'Singapore': 'Asia',
        'Australia': 'Oceania',
        'Brazil': 'South America',
        'Argentina': 'South America',
        'South Africa': 'Africa',
        'Egypt': 'Africa'
    }

    # Apply a default continent for countries not in the dictionary
    routes_with_both['SourceContinent'] = routes_with_both['Country_src'].map(
        lambda c: continents.get(c, 'Other'))
    routes_with_both['DestContinent'] = routes_with_both['Country_dest'].map(
        lambda c: continents.get(c, 'Other'))

    # Prepare feature matrix for anomaly detection
    X = routes_with_both[['Distance', 'LatDiff', 'LonDiff']].values

    return routes_with_both, X


def generate_visualization_images(routes_with_both, X, output_dir='anomaly_comparisons'):
    """
    Generate multiple visualizations with different parameters and save as images to be sent to /anomaly_comparisons folder.
    Note: This method is inefficient, it re-runs the Isolation Forest for each visualization.
          This is a necessary sacrifice for the accuracy of regional visualizations and long/short distance visualizations.
          Ultimately, you'll see inconsistency in the results but this is a good method to see the effect of different parameters.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Different contamination levels
    contamination_levels = [0.005, 0.01, 0.02, 0.05]
    for contamination in contamination_levels:
        # Run Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=42)
        clf.fit(X)

        # Update predictions
        routes_df = routes_with_both.copy()
        routes_df['IsAnomaly'] = clf.predict(X) == -1

        # Create visualization (using your existing plotting function, adapted to save files)
        plt.figure(figsize=(16, 10))

        # Create a GeoAxes object instead of a standard Axes
        ax = plt.subplot(111, projection=ccrs.Robinson())

        # Plot normal routes (sample to avoid overcrowding)
        normal_routes = routes_df[~routes_df['IsAnomaly']].sample(min(300, (~routes_df['IsAnomaly']).sum()))
        for _, route in normal_routes.iterrows():
            plt.plot([route['Longitude_src'], route['Longitude_dest']],
                     [route['Latitude_src'], route['Latitude_dest']],
                     'b-', linewidth=0.5, alpha=0.3, transform=ccrs.Geodetic())

        # Plot anomalous routes
        anomalous_routes = routes_df[routes_df['IsAnomaly']]
        for _, route in anomalous_routes.iterrows():
            plt.plot([route['Longitude_src'], route['Longitude_dest']],
                     [route['Latitude_src'], route['Latitude_dest']],
                     'r-', linewidth=0.7, alpha=0.7, transform=ccrs.Geodetic())

        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

        # Add title and legend
        anomaly_count = routes_df['IsAnomaly'].sum()
        plt.title(f'Flight Routes: {anomaly_count} Anomalies Detected ({contamination * 100:.1f}% of all routes)')

        # Add custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=0.5, alpha=0.3, label='Normal Routes (Sample)'),
            Line2D([0], [0], color='red', linewidth=0.7, alpha=0.7, label=f'Anomalous Routes ({anomaly_count})')
        ]
        plt.legend(handles=legend_elements, loc='upper left')

        # Add gridlines
        gridlines = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gridlines.top_labels = False
        gridlines.right_labels = False

        # Save the figure
        plt.savefig(f'{output_dir}/anomalies_{int(contamination * 1000)}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Generate some regional/filtered visualizations as well
    regions = {
        'North_America': lambda r: (r['SourceContinent'] == 'North America') | (r['DestContinent'] == 'North America'),
        'Europe': lambda r: (r['SourceContinent'] == 'Europe') | (r['DestContinent'] == 'Europe'),
        'Asia': lambda r: (r['SourceContinent'] == 'Asia') | (r['DestContinent'] == 'Asia'),
        'Long_Distance': lambda r: r['Distance'] > 5000,
        'Short_Distance': lambda r: r['Distance'] < 1000
    }

    for region_name, filter_func in regions.items():
        # Filter routes
        filtered_routes = routes_with_both[filter_func(routes_with_both)]
        filtered_X = X[filter_func(routes_with_both).values]

        if len(filtered_routes) > 0:
            # Run Isolation Forest
            contamination = 0.01  # Use a fixed contamination for regional views (1% contamination rate)
            clf = IsolationForest(contamination=contamination, random_state=42)
            clf.fit(filtered_X)

            # Update predictions
            filtered_routes = filtered_routes.copy()  # Create a copy to avoid SettingWithCopyWarning
            filtered_routes['IsAnomaly'] = clf.predict(filtered_X) == -1

            # Create and save visualization (similar to above)
            plt.figure(figsize=(16, 10))

            # Use Cartopy projection here too
            ax = plt.subplot(111, projection=ccrs.Robinson())

            # Plot normal routes
            normal_routes = filtered_routes[~filtered_routes['IsAnomaly']]
            if len(normal_routes) > 0:
                normal_routes = normal_routes.sample(min(300, len(normal_routes)))
                for _, route in normal_routes.iterrows():
                    plt.plot([route['Longitude_src'], route['Longitude_dest']],
                             [route['Latitude_src'], route['Latitude_dest']],
                             'b-', linewidth=0.5, alpha=0.3, transform=ccrs.Geodetic())

            # Plot anomalous routes
            anomalous_routes = filtered_routes[filtered_routes['IsAnomaly']]
            for _, route in anomalous_routes.iterrows():
                plt.plot([route['Longitude_src'], route['Longitude_dest']],
                         [route['Latitude_src'], route['Latitude_dest']],
                         'r-', linewidth=0.7, alpha=0.7, transform=ccrs.Geodetic())

            # Add map features
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

            # Add title and legend
            anomaly_count = filtered_routes['IsAnomaly'].sum()
            plt.title(f'{region_name} Routes: {anomaly_count} Anomalies Detected')

            # Add custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=0.5, alpha=0.3, label='Normal Routes (Sample)'),
                Line2D([0], [0], color='red', linewidth=0.7, alpha=0.7, label=f'Anomalous Routes ({anomaly_count})')
            ]
            plt.legend(handles=legend_elements, loc='upper left')

            # Add gridlines
            gridlines = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gridlines.top_labels = False
            gridlines.right_labels = False

            plt.savefig(f'{output_dir}/region_{region_name}.png', dpi=300, bbox_inches='tight')
            plt.close()


def main():
    # Load and preprocess the data
    routes_with_both, X = load_and_preprocess_data()

    # Focus only on 1% contamination as specified
    contamination = 0.01

    # Run Isolation Forest with 1% contamination
    print("\033[94m-\033[0m" * 50)
    print(f"\033[94m{f'Running Anomaly Detection with {contamination * 100:.1f}% Contamination':^50}\033[0m")
    print("\033[94m-\033[0m" * 50 + "\n")

    # Print some basic stats about the dataset
    print(f"Loaded {len(routes_with_both)} routes with complete data")
    print(f"Feature matrix shape: {X.shape}")

    # Create directory for images if it doesn't exist
    os.makedirs('anomaly_comparisons', exist_ok=True)

    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(X)

    # Label the anomalies
    routes_with_anomalies = routes_with_both.copy()
    routes_with_anomalies['IsAnomaly'] = clf.predict(X) == -1

    # Count anomalies
    anomaly_count = routes_with_anomalies['IsAnomaly'].sum()
    normal_count = len(routes_with_anomalies) - anomaly_count

    print(
        f"Found {anomaly_count} anomalous routes out of {len(routes_with_anomalies)} ({anomaly_count / len(routes_with_anomalies) * 100:.2f}%)")

    # Display statistics
    print("\nAverage metrics for normal vs anomalous routes:")
    metrics_by_group = routes_with_anomalies.groupby('IsAnomaly')[['Distance', 'LatDiff', 'LonDiff']].mean()
    print(metrics_by_group)

    # Calculate standard deviation for each group
    metrics_std_by_group = routes_with_anomalies.groupby('IsAnomaly')[['Distance', 'LatDiff', 'LonDiff']].std()
    print("\nStandard deviation for normal vs anomalous routes:")
    print(metrics_std_by_group)

    # Print min and max values for each group
    print("\nMinimum values for normal vs anomalous routes:")
    print(routes_with_anomalies.groupby('IsAnomaly')[['Distance', 'LatDiff', 'LonDiff']].min())

    print("\nMaximum values for normal vs anomalous routes:")
    print(routes_with_anomalies.groupby('IsAnomaly')[['Distance', 'LatDiff', 'LonDiff']].max())

    # Visualize the distribution of metrics for normal vs anomalous routes
    print("\n" + "\033[91m-\033[0m" * 50)
    print(f"\033[91m{'Generation Visualization Images':^50}\033[0m")
    print("\033[91m-\033[0m" * 50)

    # Calculate total visualizations to track in the progress bar
    contamination_levels = [0.005, 0.01, 0.02, 0.05]
    regions = ['North_America', 'Europe', 'Asia', 'Long_Distance', 'Short_Distance']
    total_visualizations = len(contamination_levels) + len(regions)

    # Use tqdm to show a progress bar for visualization generation
    with tqdm.tqdm(total=total_visualizations, desc="Generating visualizations", unit="image",
                   colour="RED") as pbar:
        # Call generate_visualization_images with modified version to update progress bar
        # We'll track the progress manually outside the function

        # Create output directory if it doesn't exist
        output_dir = 'anomaly_comparisons'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Different contamination levels
        for contamination in contamination_levels:
            # Run Isolation Forest
            clf = IsolationForest(contamination=contamination, random_state=42)
            clf.fit(X)

            # Update predictions
            routes_df = routes_with_both.copy()
            routes_df['IsAnomaly'] = clf.predict(X) == -1

            # Create visualization (using your existing plotting function, adapted to save files)
            plt.figure(figsize=(16, 10))
            ax = plt.subplot(111, projection=ccrs.Robinson())

            # Plot normal routes (sample to avoid overcrowding)
            normal_routes = routes_df[~routes_df['IsAnomaly']].sample(min(300, (~routes_df['IsAnomaly']).sum()))
            for _, route in normal_routes.iterrows():
                plt.plot([route['Longitude_src'], route['Longitude_dest']],
                         [route['Latitude_src'], route['Latitude_dest']],
                         'b-', linewidth=0.5, alpha=0.3, transform=ccrs.Geodetic())

            # Plot anomalous routes
            anomalous_routes = routes_df[routes_df['IsAnomaly']]
            for _, route in anomalous_routes.iterrows():
                plt.plot([route['Longitude_src'], route['Longitude_dest']],
                         [route['Latitude_src'], route['Latitude_dest']],
                         'r-', linewidth=0.7, alpha=0.7, transform=ccrs.Geodetic())

            # Add map features
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

            # Add title and legend
            anomaly_count = routes_df['IsAnomaly'].sum()
            plt.title(f'Flight Routes: {anomaly_count} Anomalies Detected ({contamination * 100:.1f}% of all routes)')

            # Add custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=0.5, alpha=0.3, label='Normal Routes (Sample)'),
                Line2D([0], [0], color='red', linewidth=0.7, alpha=0.7, label=f'Anomalous Routes ({anomaly_count})')
            ]
            plt.legend(handles=legend_elements, loc='upper left')

            # Add gridlines
            gridlines = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gridlines.top_labels = False
            gridlines.right_labels = False

            # Save the figure
            plt.savefig(f'{output_dir}/anomalies_{int(contamination * 1000)}.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Update progress bar
            pbar.update(1)

        # Generate some regional/filtered visualizations as well
        region_filters = {
            'North_America': lambda r: (r['SourceContinent'] == 'North America') | (
                        r['DestContinent'] == 'North America'),
            'Europe': lambda r: (r['SourceContinent'] == 'Europe') | (r['DestContinent'] == 'Europe'),
            'Asia': lambda r: (r['SourceContinent'] == 'Asia') | (r['DestContinent'] == 'Asia'),
            'Long_Distance': lambda r: r['Distance'] > 5000,
            'Short_Distance': lambda r: r['Distance'] < 1000
        }

        for region_name, filter_func in region_filters.items():
            # Filter routes
            filtered_routes = routes_with_both[filter_func(routes_with_both)]
            filtered_X = X[filter_func(routes_with_both).values]

            if len(filtered_routes) > 0:
                # Run Isolation Forest
                contamination = 0.01  # Use a fixed contamination for regional views (1% contamination rate)
                clf = IsolationForest(contamination=contamination, random_state=42)
                clf.fit(filtered_X)

                # Update predictions
                filtered_routes = filtered_routes.copy()  # Create a copy to avoid SettingWithCopyWarning
                filtered_routes['IsAnomaly'] = clf.predict(filtered_X) == -1

                # Create and save visualization
                plt.figure(figsize=(16, 10))
                ax = plt.subplot(111, projection=ccrs.Robinson())

                # Plot normal routes
                normal_routes = filtered_routes[~filtered_routes['IsAnomaly']]
                if len(normal_routes) > 0:
                    normal_routes = normal_routes.sample(min(300, len(normal_routes)))
                    for _, route in normal_routes.iterrows():
                        plt.plot([route['Longitude_src'], route['Longitude_dest']],
                                 [route['Latitude_src'], route['Latitude_dest']],
                                 'b-', linewidth=0.5, alpha=0.3, transform=ccrs.Geodetic())

                # Plot anomalous routes
                anomalous_routes = filtered_routes[filtered_routes['IsAnomaly']]
                for _, route in anomalous_routes.iterrows():
                    plt.plot([route['Longitude_src'], route['Longitude_dest']],
                             [route['Latitude_src'], route['Latitude_dest']],
                             'r-', linewidth=0.7, alpha=0.7, transform=ccrs.Geodetic())

                # Add map features
                ax.add_feature(cfeature.LAND, facecolor='lightgray')
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

                # Add title and legend
                anomaly_count = filtered_routes['IsAnomaly'].sum()
                plt.title(f'{region_name} Routes: {anomaly_count} Anomalies Detected')

                # Add custom legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='blue', linewidth=0.5, alpha=0.3, label='Normal Routes (Sample)'),
                    Line2D([0], [0], color='red', linewidth=0.7, alpha=0.7, label=f'Anomalous Routes ({anomaly_count})')
                ]
                plt.legend(handles=legend_elements, loc='upper left')

                # Add gridlines
                gridlines = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gridlines.top_labels = False
                gridlines.right_labels = False

                plt.savefig(f'{output_dir}/region_{region_name}.png', dpi=300, bbox_inches='tight')
                plt.close()

            # Update progress bar
            pbar.update(1)

    # Create distribution plots using matplotlib instead of seaborn
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    # Distance distribution
    normal_dist = routes_with_anomalies[~routes_with_anomalies['IsAnomaly']]['Distance']
    anomaly_dist = routes_with_anomalies[routes_with_anomalies['IsAnomaly']]['Distance']

    axes[0].hist(normal_dist, bins=30, alpha=0.5, density=True, label='Normal', color='blue')
    axes[0].hist(anomaly_dist, bins=30, alpha=0.5, density=True, label='Anomalous', color='red')
    axes[0].set_title('Distance Distribution: Normal vs Anomalous Routes')
    axes[0].set_xlabel('Distance (km)')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # LatDiff distribution
    normal_lat = routes_with_anomalies[~routes_with_anomalies['IsAnomaly']]['LatDiff']
    anomaly_lat = routes_with_anomalies[routes_with_anomalies['IsAnomaly']]['LatDiff']

    axes[1].hist(normal_lat, bins=30, alpha=0.5, density=True, label='Normal', color='blue')
    axes[1].hist(anomaly_lat, bins=30, alpha=0.5, density=True, label='Anomalous', color='red')
    axes[1].set_title('Latitude Difference Distribution: Normal vs Anomalous Routes')
    axes[1].set_xlabel('Latitude Difference')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    # LonDiff distribution
    normal_lon = routes_with_anomalies[~routes_with_anomalies['IsAnomaly']]['LonDiff']
    anomaly_lon = routes_with_anomalies[routes_with_anomalies['IsAnomaly']]['LonDiff']

    axes[2].hist(normal_lon, bins=30, alpha=0.5, density=True, label='Normal', color='blue')
    axes[2].hist(anomaly_lon, bins=30, alpha=0.5, density=True, label='Anomalous', color='red')
    axes[2].set_title('Longitude Difference Distribution: Normal vs Anomalous Routes')
    axes[2].set_xlabel('Longitude Difference')
    axes[2].set_ylabel('Density')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('anomaly_comparisons/metrics_distribution.png')
    plt.close()

    # Create a scatter plot of routes colored by anomaly status
    plt.figure(figsize=(12, 10))

    # Normal routes in blue, anomalies in red
    normal_routes = routes_with_anomalies[~routes_with_anomalies['IsAnomaly']]
    anomalous_routes = routes_with_anomalies[routes_with_anomalies['IsAnomaly']]

    plt.scatter(normal_routes['LonDiff'], normal_routes['LatDiff'],
                alpha=0.5, s=normal_routes['Distance'] / 50, c='blue', label='Normal')
    plt.scatter(anomalous_routes['LonDiff'], anomalous_routes['LatDiff'],
                alpha=0.7, s=anomalous_routes['Distance'] / 50, c='red', label='Anomalous')

    plt.title('Route Patterns: Normal vs Anomalous (size indicates distance)')
    plt.xlabel('Longitude Difference')
    plt.ylabel('Latitude Difference')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('anomaly_comparisons/route_scatter.png')
    plt.close()

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot normal routes
    ax.scatter(normal_routes['LonDiff'], normal_routes['LatDiff'], normal_routes['Distance'],
               alpha=0.5, c='blue', label='Normal')

    # Plot anomalous routes
    ax.scatter(anomalous_routes['LonDiff'], anomalous_routes['LatDiff'], anomalous_routes['Distance'],
               alpha=0.7, c='red', label='Anomalous')

    ax.set_title('3D Visualization of Routes: Normal vs Anomalous')
    ax.set_xlabel('Longitude Difference')
    ax.set_ylabel('Latitude Difference')
    ax.set_zlabel('Distance (km)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('anomaly_comparisons/route_3d_scatter.png')
    plt.close()

    print("\n" + "\033[92m-\033[0m" * 50)
    print(f"\033[92m{'Visualizations Completed!':^50}\033[0m")
    print("\033[92m-\033[0m" * 50)
    print("\nVisualization images generated and saved to the 'anomaly_comparisons' directory")

    # Save anomalous routes to a file for further investigation
    print("\nSaving anomalous routes to file for further investigation...")
    anomalous_routes.to_csv('anomaly_comparisons/anomalous_routes.csv', index=False)
    print(f"Saved {len(anomalous_routes)} anomalous routes to 'anomaly_comparisons/anomalous_routes.csv'")


if __name__ == "__main__":
    main()
