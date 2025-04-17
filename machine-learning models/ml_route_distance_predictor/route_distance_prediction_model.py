import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

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

# Merge airport details with routes
routes_with_src = routes_df.merge(
    airports_df[["AirportID", "Latitude", "Longitude", "Country"]],
    left_on="SourceAirportID",
    right_on="AirportID",
    suffixes=("", "_src")
)

routes_with_both = routes_with_src.merge(
    airports_df[["AirportID", "Latitude", "Longitude", "Country"]],
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

# Feature engineering
routes_with_both['SameCountry'] = (routes_with_both['Country_src'] == routes_with_both['Country_dest']).astype(int)

# Prepare data for classification: predicting if a route has stops
# Encode categorical features
le_airline = LabelEncoder()
routes_with_both['Airline_encoded'] = le_airline.fit_transform(routes_with_both['Airline'])

# Create features and target
X = routes_with_both[['Distance', 'SameCountry', 'Airline_encoded']].values
y = (routes_with_both['Stops'] > 0).astype(int)  # Binary: direct flight or has stops

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importances = pd.DataFrame({
    'Feature': ['Distance', 'Same Country', 'Airline'],
    'Importance': clf.feature_importances_
})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importances['Feature'], feature_importances['Importance'])
plt.title('Feature Importance for Predicting Stops')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()