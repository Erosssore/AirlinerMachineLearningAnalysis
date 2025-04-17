# visualizations.py
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_curve, auc


def visualize_airport_network(G, airports_df, sample_size=100):
    """
    Visualize a sample of the airport network with a Robinson map overlay
    """
    # Check if required packages are installed, if not, try to install them
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy package is required for map overlay. Installing...")
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cartopy"])
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

    # Get the largest connected component
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc)

    # Sample nodes to avoid overcrowding
    sampled_nodes = list(G_sub.nodes())[:sample_size]
    G_sample = G_sub.subgraph(sampled_nodes)

    # Set up the plot with map projection
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.Robinson())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.5)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Create a dictionary of positions from latitude and longitude
    pos = {}
    valid_nodes = []

    for node in G_sample.nodes():
        airport_data = airports_df[airports_df['Airport ID'] == node]
        if not airport_data.empty and not pd.isna(airport_data['Latitude'].values[0]) and not pd.isna(
                airport_data['Longitude'].values[0]):
            # Store the position
            pos[node] = (airport_data['Longitude'].values[0], airport_data['Latitude'].values[0])
            valid_nodes.append(node)

    # Create a subgraph of nodes with valid positions
    G_valid = G_sample.subgraph(valid_nodes)

    # Draw the nodes
    for node in G_valid.nodes():
        lon, lat = pos[node]
        ax.plot(lon, lat, 'o', transform=ccrs.PlateCarree(),
                markersize=5, color='blue', alpha=0.7)

    # Draw the edges
    for edge in G_valid.edges():
        source, target = edge
        if source in pos and target in pos:
            start_lon, start_lat = pos[source]
            end_lon, end_lat = pos[target]

            # Draw great circle route
            ax.plot([start_lon, end_lon], [start_lat, end_lat],
                    transform=ccrs.Geodetic(),
                    color='red', linewidth=0.6, alpha=0.3)

    # Add title and adjust layout
    plt.title(f'Global Airport Network (Sample of {len(G_valid)} airports)', fontsize=14)

    # Save and show
    plt.savefig('airport_link_predictions/airport_network_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()



def visualize_feature_importance(model, feature_names):
    """
    Visualize feature importance from the trained model
    """
    # Get feature importances from the model
    importances = model.feature_importances_

    # Sort features by importance
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Airport Link Prediction')
    plt.tight_layout()
    plt.savefig('airport_link_predictions/feature_importance.png', dpi=300)
    plt.close()


def visualize_roc_curve(y_test, y_pred_proba):
    """
    Visualize the ROC curve for the model

    Parameters:
    ----------
    y_test : array-like
        True binary labels
    y_pred_proba : array-like
        Probability estimates for the positive class (if 2D array, column 1 will be used)
    """
    # Extract the probability for the positive class if y_pred_proba is 2D
    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
        # Use the second column (index 1) which typically contains positive class probabilities
        y_pred_proba_pos = y_pred_proba[:, 1]
    else:
        y_pred_proba_pos = y_pred_proba

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_pos)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('airport_link_predictions/roc_curve.png', dpi=300)
    plt.close()


def visualize_top_predicted_routes(G, X_test, y_test, y_pred_proba, airports_df, test_pairs,
                                   threshold=0.7, top_n=25, show_probability_range=False): # show_probability_range = False for top values
    """
    Visualize the top predicted new routes on a proper world map with cartographic features

    Parameters:
    ----------
    G : networkx.DiGraph
        The airport network graph
    X_test : array-like
        Test features
    y_test : array-like
        True binary labels
    y_pred_proba : array-like
        Probability estimates for each class
    airports_df : pandas.DataFrame
        DataFrame containing airport information including coordinates
    test_pairs : list of tuples
        List of airport ID pairs being tested
    threshold : float, default=0.7
        Probability threshold for considering a prediction positive
    top_n : int, default=10
        Number of top predictions to visualize
    show_probability_range : bool, default=True
        If True, will select routes across a range of probabilities
    """
    # Import required libraries
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        CARTOPY_AVAILABLE = True
    except ImportError:
        print("Cartopy package is required for map overlay. Installing...")
        import sys
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cartopy"])
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            CARTOPY_AVAILABLE = True
        except:
            print("Failed to install Cartopy. Using simplified visualization.")
            CARTOPY_AVAILABLE = False

    # Get indices of airport pairs that don't have a connection but are predicted to have one
    non_connected_indices = np.where((y_test == 0) & (y_pred_proba[:, 1] > threshold))[0]

    if len(non_connected_indices) == 0:
        print(f"No predictions with probability > {threshold} found. Lowering threshold...")
        # Try with a lower threshold if no predictions meet the criteria
        threshold = 0.5
        non_connected_indices = np.where((y_test == 0) & (y_pred_proba[:, 1] > threshold))[0]

        if len(non_connected_indices) == 0:
            print("No suitable predictions found. Try lowering the threshold further.")
            return

    # Sort by prediction probability
    sorted_indices = non_connected_indices[np.argsort(y_pred_proba[non_connected_indices, 1])[::-1]]

    # Use the top predictions
    selected_indices = sorted_indices[:min(top_n, len(sorted_indices))]

    if show_probability_range and len(sorted_indices) >= top_n * 2:
        # Select routes across the probability spectrum to show variation
        # Take some from the top, some from the middle, some from the bottom
        top_third = sorted_indices[:len(sorted_indices) // 3]
        mid_third = sorted_indices[len(sorted_indices) // 3:2 * len(sorted_indices) // 3]
        bottom_third = sorted_indices[2 * len(sorted_indices) // 3:]

        # Distribute the top_n selections across these ranges
        n_top = max(top_n // 3, 1)
        n_mid = max(top_n // 3, 1)
        n_bottom = max(top_n - n_top - n_mid, 1)

        selected_indices = list(top_third[:n_top]) + \
                           list(mid_third[:n_mid]) + \
                           list(bottom_third[:n_bottom])
    else:
        # Just take the top predictions
        selected_indices = sorted_indices[:min(top_n, len(sorted_indices))]

    # Get the actual airport pairs and their probabilities
    top_pairs = [test_pairs[i] for i in selected_indices]
    top_probas = [y_pred_proba[i, 1] for i in selected_indices]

    # Print the probability distribution for inspection
    output = " | ".join([f"Route {i + 1}: {prob:.4f}" for i, prob in enumerate(top_probas)])
    print(output)

    if CARTOPY_AVAILABLE:
        # Create a figure with a proper map projection
        plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=ccrs.Robinson())

        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAKES, alpha=0.5)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        # Dictionary to keep track of labels to avoid overlapping
        labeled_airports = set()

        # Create colormap for a proper probability gradient
        norm = plt.Normalize(min(top_probas), max(top_probas))
        cmap = plt.cm.YlOrRd

        # Plot each predicted route
        for (source, target), probability in zip(top_pairs, top_probas):
            source_data = airports_df[airports_df['Airport ID'] == source]
            target_data = airports_df[airports_df['Airport ID'] == target]

            if source_data.empty or target_data.empty:
                continue

            # Get coordinates and names
            source_lon = source_data['Longitude'].values[0]
            source_lat = source_data['Latitude'].values[0]
            target_lon = target_data['Longitude'].values[0]
            target_lat = target_data['Latitude'].values[0]

            # Skip if any coordinates are missing
            if pd.isna(source_lon) or pd.isna(source_lat) or pd.isna(target_lon) or pd.isna(target_lat):
                continue

            # Get airport names (with fallback)
            source_name = source_data['Name'].values[0] if 'Name' in source_data.columns else f"Airport {source}"
            target_name = target_data['Name'].values[0] if 'Name' in target_data.columns else f"Airport {target}"

            # Get IATA codes if available
            source_code = source_data['IATA'].values[0] if 'IATA' in source_data.columns and not pd.isna(
                source_data['IATA'].values[0]) else ""
            target_code = target_data['IATA'].values[0] if 'IATA' in target_data.columns and not pd.isna(
                target_data['IATA'].values[0]) else ""

            # Add source code to name if available
            if source_code:
                source_label = f"{source_code}"
            else:
                source_label = source_name[:15] + '...' if len(source_name) > 15 else source_name

            # Add target code to name if available
            if target_code:
                target_label = f"{target_code}"
            else:
                target_label = target_name[:15] + '...' if len(target_name) > 15 else target_name

            # Determine line color based on probability (using normalized colormap)
            color = cmap(norm(probability))

            # Draw the great circle route
            ax.plot([source_lon, target_lon], [source_lat, target_lat],
                    transform=ccrs.Geodetic(),
                    color=color, linewidth=2, alpha=0.7)

            # Plot airports as points
            ax.plot(source_lon, source_lat, 'o', transform=ccrs.PlateCarree(),
                    markersize=7, color='blue')
            ax.plot(target_lon, target_lat, 'o', transform=ccrs.PlateCarree(),
                    markersize=7, color='blue')

            # Add labels if they haven't been added yet
            if source not in labeled_airports:
                ax.text(source_lon + 1, source_lat + 1, source_label,
                        transform=ccrs.PlateCarree(),
                        fontsize=9, ha='left', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
                labeled_airports.add(source)

            if target not in labeled_airports:
                ax.text(target_lon + 1, target_lat + 1, target_label,
                        transform=ccrs.PlateCarree(),
                        fontsize=9, ha='left', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
                labeled_airports.add(target)

        # Add a colorbar to show the probability scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                            pad=0.05, shrink=0.5, label='Prediction Probability')

        # Add title
        if show_probability_range:
            plt.title(f'Predicted New Routes Across Probability Range ({threshold:.2f} - {max(top_probas):.2f})',
                      fontsize=14, pad=10)
        else:
            plt.title(f'Top {len(selected_indices)} Predicted New Routes (Probability > {threshold:.2f})',
                      fontsize=14, pad=10)
    else:
        # Fallback visualization for systems without cartopy
        plt.figure(figsize=(12, 8))

        # Create a scatter plot of airports
        airport_ids = set()
        for source, target in top_pairs:
            airport_ids.add(source)
            airport_ids.add(target)

        # Filter airports dataframe for only those in our routes
        plot_airports = airports_df[airports_df['Airport ID'].isin(airport_ids)]

        # Plot airports
        plt.scatter(plot_airports['Longitude'], plot_airports['Latitude'],
                    s=30, c='blue', alpha=0.7, label='Airports')

        # Create colormap
        norm = plt.Normalize(min(top_probas), max(top_probas))
        cmap = plt.cm.YlOrRd

        # Plot routes
        for (source, target), probability in zip(top_pairs, top_probas):
            source_data = airports_df[airports_df['Airport ID'] == source]
            target_data = airports_df[airports_df['Airport ID'] == target]

            if source_data.empty or target_data.empty:
                continue

            # Get coordinates
            source_lon = source_data['Longitude'].values[0]
            source_lat = source_data['Latitude'].values[0]
            target_lon = target_data['Longitude'].values[0]
            target_lat = target_data['Latitude'].values[0]

            # Skip if any coordinates are missing
            if pd.isna(source_lon) or pd.isna(source_lat) or pd.isna(target_lon) or pd.isna(target_lat):
                continue

            # Get color based on probability
            color = cmap(norm(probability))

            # Draw the route as a straight line (simplification)
            plt.plot([source_lon, target_lon], [source_lat, target_lat],
                     color=color, linewidth=1.5, alpha=0.7)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Prediction Probability')

        # Add title and axis labels
        if show_probability_range:
            plt.title(f'Predicted New Routes Across Probability Range ({threshold:.2f} - {max(top_probas):.2f})')
        else:
            plt.title(f'Top {len(selected_indices)} Predicted New Routes (Probability > {threshold:.2f})')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)

    # Save and show the figure
    plt.savefig('airport_link_predictions/top_predicted_routes_map.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_degree_distribution(G):
    """
    Visualize the degree distribution of the airport network

    Parameters:
    ----------
    G : networkx.DiGraph
        The airport network graph
    """
    degrees = [d for n, d in G.degree()]

    if not degrees:
        print("The graph has no nodes or edges to visualize.")
        return

    plt.figure(figsize=(10, 6))

    # Determine appropriate number of bins based on data
    n_bins = min(50, len(set(degrees)))

    # Plot histogram
    plt.hist(degrees, bins=n_bins, alpha=0.7)
    plt.xlabel('Degree (Number of Connections)')
    plt.ylabel('Number of Airports')
    plt.title('Airport Connection Distribution')

    # Add a log-log plot of the degree distribution as inset
    degree_count = Counter(degrees)
    x = np.array(list(degree_count.keys()))
    y = np.array(list(degree_count.values()))

    # Only add the inset if we have enough distinct degree values
    if len(x) > 5:
        # Create inset axes
        plt.axes([0.55, 0.55, 0.3, 0.3])
        plt.loglog(x, y, 'ro', markersize=3)
        plt.xlabel('Degree (log)')
        plt.ylabel('Count (log)')
        plt.grid(True, alpha=0.3)
        # Don't use tight_layout when we have an inset
        use_tight_layout = False
    else:
        use_tight_layout = True

    # Only apply tight_layout if we don't have an inset
    if use_tight_layout:
        plt.tight_layout()

    plt.savefig('airport_link_predictions/degree_distribution.png', dpi=300)
    plt.close()

