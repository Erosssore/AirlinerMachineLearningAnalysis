# run_visualizations.py
import pickle
from visualizations import (
    visualize_airport_network,
    visualize_feature_importance,
    visualize_roc_curve,
    visualize_top_predicted_routes,
    visualize_degree_distribution
)

# Load the data
with open('visualization_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Check if we need to rename columns in airports_df
if 'airports_df' in data and isinstance(data['airports_df'], object):
    # Print available columns to help with debugging
    print("Available columns in airports_df:", data['airports_df'].columns.tolist())

    # Look for potential column names that might be used for Airport ID
    airport_id_columns = [col for col in data['airports_df'].columns
                          if 'airport' in col.lower() and 'id' in col.lower()]

    # If we found a potential match and 'Airport ID' is not already in columns
    if airport_id_columns and 'Airport ID' not in data['airports_df'].columns:
        # Rename the first matching column to 'Airport ID'
        data['airports_df'] = data['airports_df'].rename(columns={airport_id_columns[0]: 'Airport ID'})
        print(f"Renamed '{airport_id_columns[0]}' to 'Airport ID'")
    # If no match is found but we need this column
    elif 'Airport ID' not in data['airports_df'].columns:
        # Create an Airport ID column based on index as fallback
        data['airports_df']['Airport ID'] = data['airports_df'].index
        print("Created 'Airport ID' column based on DataFrame index")

# Run all visualizations
print("\n" + "-" * 50)
print("Generating Airport Network".center(50))
print("-" * 50)
visualize_airport_network(data['G'], data['airports_df'], sample_size=100)

print("\n" + "-" * 50)
print("Generating Feature Importance".center(50))
print("-" * 50)
visualize_feature_importance(data['link_pred_model'], data['feature_names'])

print("\n" + "-" * 50)
print("Generating ROC Curve".center(50))
print("-" * 50)
visualize_roc_curve(data['y_test'], data['y_pred_proba'])

print("\n" + "-" * 50)
print("Generating Top Predicted Routes".center(50))
print("-" * 50 + "\n")
visualize_top_predicted_routes(
    data['G'], data['X_test'], data['y_test'],
    data['y_pred_proba'], data['airports_df'], data['test_pairs']
)

print("\n" + "-" * 50)
print("Generating Degree Distribution".center(50))
print("-" * 50)
visualize_degree_distribution(data['G'])

