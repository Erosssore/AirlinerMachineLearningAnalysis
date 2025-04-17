import pandas as pd
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import random
import subprocess
import os


"""
Generate visualization_data.pkl file to be used by run_visualizations.py
"""

print("\n" + "\033[94m-\033[0m" * 50)
print(f"\033[94m{'Training Random Forest for Link Predictions':^50}\033[0m")
print("\033[94m-\033[0m" * 50)

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

# Convert ID columns to integers where possible
airports_df['AirportID'] = pd.to_numeric(airports_df['AirportID'], errors='coerce')
routes_df['SourceAirportID'] = pd.to_numeric(routes_df['SourceAirportID'], errors='coerce')
routes_df['DestinationAirportID'] = pd.to_numeric(routes_df['DestinationAirportID'], errors='coerce')

# Debug information
print(f"Airport ID data type: {airports_df['AirportID'].dtype}")
print(f"Source Airport ID data type: {routes_df['SourceAirportID'].dtype}")
print(f"Destination Airport ID data type: {routes_df['DestinationAirportID'].dtype}")

# Create a set of valid airport IDs for faster lookup
valid_airport_ids = set(airports_df['AirportID'].dropna().astype(int).tolist())
print(f"Number of valid airport IDs: {len(valid_airport_ids)}")

# Create a network graph
G = nx.DiGraph()

# Add airports as nodes
valid_airports = 0
for _, airport in airports_df.iterrows():
    if not pd.isna(airport['AirportID']):
        airport_id = int(airport['AirportID'])
        G.add_node(airport_id,
                   name=airport['Name'],
                   iata=airport['IATA'],
                   lat=airport['Latitude'],
                   lon=airport['Longitude'],
                   country=airport['Country'])
        valid_airports += 1

print(f"Valid airports added as nodes: {valid_airports}")

# Add routes as edges
valid_routes = 0
invalid_routes = 0
for _, route in routes_df.iterrows():
    if not pd.isna(route['SourceAirportID']) and not pd.isna(route['DestinationAirportID']):
        try:
            src_id = int(route['SourceAirportID'])
            dst_id = int(route['DestinationAirportID'])

            if src_id in valid_airport_ids and dst_id in valid_airport_ids:
                G.add_edge(src_id, dst_id, airline=route['Airline'], stops=route['Stops'])
                valid_routes += 1
            else:
                invalid_routes += 1
        except (ValueError, TypeError):
            invalid_routes += 1

print(f"Valid routes added as edges: {valid_routes}")
print(f"Invalid routes skipped: {invalid_routes}")

# Network stats
print(f"Number of airports (nodes): {G.number_of_nodes()}")
print(f"Number of routes (edges): {G.number_of_edges()}")

# Check if the graph has edges
if G.number_of_edges() == 0:
    # If no edges were found, try to identify if there are any potential routes
    sample_routes = routes_df.head(10).to_string()
    print(f"Sample routes from dataset:\n{sample_routes}")
    raise ValueError("No valid edges found in the graph. Check your data files and their formats.")

# Calculate network metrics for each node
node_metrics = {}
for node in G.nodes():
    node_metrics[node] = {
        'degree': G.degree(node),
        'in_degree': G.in_degree(node),
        'out_degree': G.out_degree(node),
        'betweenness': 0,  # Placeholder for betweenness centrality
        'pagerank': 0  # Placeholder for PageRank
    }

# Compute centrality for the largest connected component to make computation feasible
largest_cc = max(nx.weakly_connected_components(G), key=len)
G_sub = G.subgraph(largest_cc)

print(f"Size of largest connected component: {len(largest_cc)} nodes")

# Calculate betweenness centrality (computationally expensive)
# Using a subset sample of nodes for demonstration
sample_size = min(1000, len(G_sub.nodes()))  # Taking a small subset for demonstration
betweenness = nx.betweenness_centrality(G_sub, k=sample_size, normalized=True)
pagerank = nx.pagerank(G_sub)

# Update node metrics with centrality measures
for node in G_sub.nodes():
    if node in node_metrics:
        node_metrics[node]['betweenness'] = betweenness.get(node, 0)
        node_metrics[node]['pagerank'] = pagerank.get(node, 0)

# Link prediction: Create training data
# IMPROVED: Directly collect edges from the graph rather than sampling nodes first
connected_pairs = []
non_connected_pairs = []

# Get edges from the largest connected component
edges = list(G_sub.edges())
if len(edges) == 0:
    raise ValueError("No edges found in the largest connected component.")

# Limit the number of edges we'll use to ensure balanced sampling
max_edges = min(5000, len(edges))
connected_pairs = random.sample(edges, max_edges)

# Now collect an equal number of non-connected pairs
# Get a list of nodes from the largest connected component
nodes_list = list(G_sub.nodes())

# Create a set of existing edges for faster lookup
existing_edges = set(G_sub.edges())

# Find non-connected pairs
attempts = 0
max_attempts = len(nodes_list) * 10  # Avoid infinite loop
while len(non_connected_pairs) < max_edges and attempts < max_attempts:
    source = random.choice(nodes_list)
    target = random.choice(nodes_list)
    if source != target and (source, target) not in existing_edges:
        non_connected_pairs.append((source, target))
    attempts += 1

# If we couldn't find enough non-connected pairs, reduce the connected pairs
if len(non_connected_pairs) < len(connected_pairs):
    connected_pairs = random.sample(connected_pairs, len(non_connected_pairs))

# Check if we have both classes
print(f"Connected pairs: {len(connected_pairs)}")
print(f"Non-connected pairs: {len(non_connected_pairs)}")

if len(connected_pairs) == 0 or len(non_connected_pairs) == 0:
    # Try a more aggressive approach to find connected pairs by sampling actual edges
    print("Unable to balance dataset with previous approach. Trying direct edge sampling...")

    # Get all edges from the graph
    all_edges = list(G.edges())
    connected_pairs = random.sample(all_edges, min(1000, len(all_edges)))

    # Get random non-connected pairs
    all_nodes = list(G.nodes())
    non_connected_pairs = []
    existing_edges = set(G.edges())

    attempts = 0
    while len(non_connected_pairs) < len(connected_pairs) and attempts < 10000:
        source = random.choice(all_nodes)
        target = random.choice(all_nodes)
        if source != target and (source, target) not in existing_edges:
            non_connected_pairs.append((source, target))
        attempts += 1

    # Ensure we have equal numbers
    min_pairs = min(len(connected_pairs), len(non_connected_pairs))
    if min_pairs == 0:
        raise ValueError("Cannot create a balanced dataset. The graph structure may be problematic.")

    connected_pairs = connected_pairs[:min_pairs]
    non_connected_pairs = non_connected_pairs[:min_pairs]

    print(f"After direct sampling - Connected pairs: {len(connected_pairs)}")
    print(f"After direct sampling - Non-connected pairs: {len(non_connected_pairs)}")

# Final check for dataset balance
if len(connected_pairs) == 0 or len(non_connected_pairs) == 0:
    raise ValueError("Cannot create a balanced dataset. Please check your data and graph structure.")

# Combine the pairs
all_pairs = connected_pairs + non_connected_pairs

# Create feature vectors
features = []
labels = []

for source, target in all_pairs:
    # Make sure we have metrics for both nodes
    if source in node_metrics and target in node_metrics:
        # Features: node metrics for source and target
        feature_vector = [
            node_metrics[source]['degree'],
            node_metrics[source]['in_degree'],
            node_metrics[source]['out_degree'],
            node_metrics[source]['pagerank'],
            node_metrics[target]['degree'],
            node_metrics[target]['in_degree'],
            node_metrics[target]['out_degree'],
            node_metrics[target]['pagerank']
        ]

        # Label: 1 if edge exists, 0 otherwise
        label = 1 if (source, target) in existing_edges else 0

        features.append(feature_vector)
        labels.append(label)

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Check class distribution
unique_classes, counts = np.unique(y, return_counts=True)
print(f"Class distribution: {dict(zip(unique_classes, counts))}")

# Ensure we have a balanced dataset
if len(unique_classes) < 2:
    raise ValueError("Dataset is still imbalanced: only one class present. Check your graph structure.")

# Train-test split
# Store the airport pairs along with features
airport_pairs = np.array(all_pairs)  # Make sure this is defined based on your existing code

# Modify the train_test_split to also split the pairs
X_train, X_test, y_train, y_test, train_pairs, test_pairs = train_test_split(
    X, y, airport_pairs, test_size=0.2, random_state=42, stratify=y)


# Train a Random Forest for link prediction
link_pred_model = RandomForestClassifier(n_estimators=100, random_state=42)
link_pred_model.fit(X_train, y_train)

# Safe prediction with error handling
try:
    # Check number of classes in the model
    n_classes = len(link_pred_model.classes_)
    print(f"Number of classes in model: {n_classes}")

    if n_classes > 1:
        # Binary classification, can access second column
        y_pred_proba = link_pred_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"Link Prediction AUC: {auc_score:.3f}")
    else:
        # Single class, use different evaluation
        print("Warning: Only one class present in the model. Cannot calculate AUC.")
        y_pred = link_pred_model.predict(X_test)
        print(classification_report(y_test, y_pred))
except Exception as e:
    print(f"Error during prediction: {e}")
    # Use direct predictions instead of probabilities
    y_pred = link_pred_model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Feature importance
feature_names = [
    'Source Degree', 'Source In-Degree', 'Source Out-Degree', 'Source PageRank',
    'Target Degree', 'Target In-Degree', 'Target Out-Degree', 'Target PageRank'
]

feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': link_pred_model.feature_importances_
})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importances['Feature'], feature_importances['Importance'])
plt.title('Feature Importance for Link Prediction')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.close()

# Save data for visualizations
visualization_data = {
    'G': G,
    'airports_df': airports_df,
    'X_test': X_test,
    'y_test': y_test,
    'y_pred_proba': link_pred_model.predict_proba(X_test) if len(link_pred_model.classes_) > 1 else None,
    'feature_names': feature_names,
    'link_pred_model': link_pred_model,
    'test_pairs': test_pairs
}

# Save the data
with open('visualization_data.pkl', 'wb') as f:
    pickle.dump(visualization_data, f)

print("Data saved for visualizations.")

# Execute run_visualizations.py
print("\n" + "\033[94m-\033[0m" * 50)
print(f"\033[94m{'Starting run_visualizations.py':^50}\033[0m")
print("\033[94m-\033[0m" * 50)
try:
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_visualizations.py')
    subprocess.run(['python', script_path], check=True)
    print("\n" + "\033[92m-\033[0m" * 50)
    print(f"\033[92m{'Visualizations Completed!':^50}\033[0m")
    print("\033[92m-\033[0m" * 50)
except subprocess.CalledProcessError as e:
    print()
    print(f"\033[91mError running visualizations: {e}\033[0m")
except FileNotFoundError:
    print("\n" + "\033[91m-\033[0m" * 50)
    print(f"\033[91m{'Visualization script not found. Make sure run_visualizations.py is in the same directory.':^50}\033[0m")
    print("\033[91m-\033[0m" * 50)


