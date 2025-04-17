import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math
import os
import traceback

# Get absolute path for visualization directory
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, "airport_traffic_predictions")

# Create directory for visualizations if it doesn't exist
os.makedirs("./airport_traffic_predictions", exist_ok=True)

# Helper function for saving visualizations with error handling
def save_visualization(fig, filename, close_after=True):
    """Save a matplotlib figure with error handling"""
    full_path = os.path.join(output_dir, filename)
    try:
        fig.savefig(full_path)
        print(f"Successfully saved: {full_path}")
        if close_after:
            plt.close(fig)
        return True
    except Exception as e:
        print(f"Error saving {filename}: {str(e)}")
        traceback.print_exc()
        return False


# Load data
try:
    airports_file = "../../data/airports.dat"
    routes_file = "../../data/routes.dat"

    # Load the data with the appropriate headers
    airports_cols = [
        "AirportID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude",
        "Altitude", "Timezone", "DST", "TzDatabaseTimeZone", "Type", "Source"
    ]
    airports_df = pd.read_csv(airports_file, header=None, names=airports_cols)

    # Convert AirportID to integer type to avoid merge issues
    airports_df['AirportID'] = pd.to_numeric(airports_df['AirportID'], errors='coerce')
    airports_df = airports_df.dropna(subset=['AirportID'])  # Remove rows where conversion failed
    airports_df['AirportID'] = airports_df['AirportID'].astype(int)  # Convert to int

    routes_cols = [
        "Airline", "AirlineID", "SourceAirport", "SourceAirportID",
        "DestinationAirport", "DestinationAirportID", "Codeshare",
        "Stops", "Equipment"
    ]
    routes_df = pd.read_csv(routes_file, header=None, names=routes_cols)

    # Convert airport IDs to numeric in routes_df
    routes_df['SourceAirportID'] = pd.to_numeric(routes_df['SourceAirportID'], errors='coerce')
    routes_df['DestinationAirportID'] = pd.to_numeric(routes_df['DestinationAirportID'], errors='coerce')
except Exception as e:
    print(f"Error loading data: {e}")
    traceback.print_exc()
    raise

try:
    # Calculate airport traffic metrics
    airport_traffic = pd.DataFrame({
        'OutboundRoutes': routes_df['SourceAirportID'].value_counts(),
        'InboundRoutes': routes_df['DestinationAirportID'].value_counts()
    })
    airport_traffic = airport_traffic.fillna(0)
    airport_traffic['TotalRoutes'] = airport_traffic['OutboundRoutes'] + airport_traffic['InboundRoutes']

    # Convert the index to int64 to match AirportID type
    airport_traffic.index = airport_traffic.index.astype('int64')

    # Merge with airport data
    airport_features = airports_df.merge(
        airport_traffic,
        left_on='AirportID',
        right_index=True,
        how='left'
    )
    airport_features = airport_features.fillna({'OutboundRoutes': 0, 'InboundRoutes': 0, 'TotalRoutes': 0})
except Exception as e:
    print(f"Error calculating airport traffic metrics: {e}")
    traceback.print_exc()
    raise


# Create dummy variables for region/continent
def assign_region(country):
    if country in ['United States', 'Canada', 'Mexico']:
        return 'North America'
    elif country in ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela']:
        return 'South America'
    elif country in ['China', 'Japan', 'South Korea', 'India', 'Singapore', 'Thailand', 'Malaysia', 'Indonesia']:
        return 'Asia Pacific'
    elif country in ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'Netherlands', 'Switzerland', 'Belgium']:
        return 'Europe'
    elif country in ['United Arab Emirates', 'Qatar', 'Saudi Arabia', 'Israel', 'Turkey', 'Egypt']:
        return 'Middle East'
    elif country in ['South Africa', 'Nigeria', 'Kenya', 'Ethiopia', 'Morocco', 'Tunisia']:
        return 'Africa'
    else:
        return 'Other'


try:
    airport_features['Region'] = airport_features['Country'].apply(assign_region)
    region_dummies = pd.get_dummies(airport_features['Region'], prefix='Region')
    airport_features = pd.concat([airport_features, region_dummies], axis=1)
except Exception as e:
    print(f"Error creating region dummies: {e}")
    traceback.print_exc()


# Haversine distance function for calculating distances between coordinates
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r



# Calculate distance to nearest major city or economic center
def calculate_city_proximity(airports_df):
    """Calculate distance to nearest major economic centers"""
    try:
        # Coordinates of 40 major global economic centers
        economic_centers = {
            'New York City': (40.7128, -74.0060),
            'London': (51.5074, -0.1278),
            'Tokyo': (35.6895, 139.6917),
            'Shanghai': (31.2304, 121.4737),
            'Hong Kong': (22.3193, 114.1694),
            'Singapore': (1.3521, 103.8198),
            'Paris': (48.8566, 2.3522),
            'Frankfurt': (50.1109, 8.6821),
            'Sydney': (-33.8688, 151.2093),
            'Toronto': (43.6511, -79.3839),
            'San Francisco': (37.7749, -122.4194),
            'Los Angeles': (34.0522, -118.2437),
            'Beijing': (39.9042, 116.4074),
            'Seoul': (37.5665, 126.9780),
            'Dubai': (25.2048, 55.2708),
            'Mumbai': (19.0760, 72.8777),
            'São Paulo': (-23.5505, -46.6333),
            'Zurich': (47.3769, 8.5417),
            'Chicago': (41.8781, -87.6298),
            'Amsterdam': (52.3676, 4.9041),
            'Milan': (45.4642, 9.1900),
            'Madrid': (40.4168, -3.7038),
            'Bangkok': (13.7563, 100.5018),
            'Istanbul': (41.0082, 28.9784),
            'Johannesburg': (-26.2041, 28.0473),
            'Mexico City': (19.4326, -99.1332),
            'Moscow': (55.7558, 37.6173),
            'Jakarta': (-6.2088, 106.8456),
            'Riyadh': (24.7136, 46.6753),
            'Abu Dhabi': (24.4539, 54.3773),
            'Doha': (25.276987, 51.520008),
            'Kuala Lumpur': (3.1390, 101.6869),
            'Lagos': (6.5244, 3.3792),
            'Santiago': (-33.4489, -70.6693),
            'Buenos Aires': (-34.6037, -58.3816),
            'Vienna': (48.2082, 16.3738),
            'Brussels': (50.8503, 4.3517),
            'Warsaw': (52.2297, 21.0122),
            'Copenhagen': (55.6761, 12.5683),
            'Stockholm': (59.3293, 18.0686),
            'Oslo': (59.9139, 10.7522)

        }

        # Calculate distances from each airport to each economic center
        distances = []
        nearest_centers = []

        for _, airport in airports_df.iterrows():
            airport_lat = airport['Latitude']
            airport_lon = airport['Longitude']

            # Calculate distances to all centers using haversine
            dist_to_centers = []
            for center_name, (center_lat, center_lon) in economic_centers.items():
                dist = haversine(airport_lat, airport_lon, center_lat, center_lon)
                dist_to_centers.append((dist, center_name))

            # Find minimum distance and corresponding center
            min_dist, nearest_center = min(dist_to_centers, key=lambda x: x[0])

            distances.append(min_dist)
            nearest_centers.append(nearest_center)

        # Add to dataframe
        airports_df['Distance_to_EconCenter'] = distances
        airports_df['Nearest_EconCenter'] = nearest_centers

        return airports_df
    except Exception as e:
        print(f"Error calculating city proximity: {e}")
        traceback.print_exc()
        # Add default values in case of error
        airports_df['Distance_to_EconCenter'] = 0
        airports_df['Nearest_EconCenter'] = 'Unknown'
        return airports_df


# Calculate network-based metrics from the existing routes data
def calculate_network_metrics(airport_df, routes_df):
    """Calculate network metrics including enhanced hub score"""
    print("Calculating network centrality metrics...")

    # Initialize dictionaries to store route counts
    dest_counts = {}
    routes_per_airport = {}
    inbound_routes = {}
    outbound_routes = {}

    # Process routes to count destinations and total routes
    for _, route in routes_df.iterrows():
        source_id = route['SourceAirportID']
        dest_id = route['DestinationAirportID']

        # Skip invalid entries
        if pd.isna(source_id) or pd.isna(dest_id):
            continue

        # Count outbound routes
        if source_id not in routes_per_airport:
            routes_per_airport[source_id] = 0
            dest_counts[source_id] = set()
            outbound_routes[source_id] = 0

        routes_per_airport[source_id] += 1
        dest_counts[source_id].add(dest_id)
        outbound_routes[source_id] += 1

        # Count inbound routes
        if dest_id not in inbound_routes:
            inbound_routes[dest_id] = 0
        inbound_routes[dest_id] += 1

    # Convert dest_counts from sets to integers
    for airport_id in dest_counts:
        dest_counts[airport_id] = len(dest_counts[airport_id])

    # Initialize columns for network metrics - explicitly create as float type
    airport_df['Unique_Destinations'] = 0
    airport_df['Connection_Ratio'] = 0.0  # Explicitly float
    airport_df['Hub_Score'] = 0.0  # Explicitly float
    airport_df['InboundRoutes'] = 0
    airport_df['OutboundRoutes'] = 0
    airport_df['TotalRoutes'] = 0

    # Ensure columns are of float type for those that need to hold float values
    airport_df['Connection_Ratio'] = airport_df['Connection_Ratio'].astype(float)
    airport_df['Hub_Score'] = airport_df['Hub_Score'].astype(float)

    # Create component scores for potential analysis
    hub_component_scores = []

    # Calculate network metrics for each airport
    for idx, airport in airport_df.iterrows():
        airport_id = airport['AirportID']

        # Number of unique destinations
        unique_dests = dest_counts.get(airport_id, 0)
        airport_df.at[idx, 'Unique_Destinations'] = unique_dests

        # Inbound and outbound routes
        inbound = inbound_routes.get(airport_id, 0)
        outbound = outbound_routes.get(airport_id, 0)
        total_routes = inbound + outbound

        airport_df.at[idx, 'InboundRoutes'] = inbound
        airport_df.at[idx, 'OutboundRoutes'] = outbound
        airport_df.at[idx, 'TotalRoutes'] = total_routes

        # Ratio of connections to total routes
        ratio = unique_dests / max(outbound, 1)  # Avoid division by zero
        airport_df.at[idx, 'Connection_Ratio'] = float(ratio)  # Explicitly cast to float

        # Enhanced Hub Score Calculation
        # Basic balance ratio - balanced hubs have similar in/out flows
        balance_ratio = min(inbound, outbound) / max(max(inbound, outbound), 1)

        # Scale factor - larger airports should score higher if they maintain balance
        scale_factor = math.log10(total_routes + 1) / 4  # Logarithmic scaling

        # Network centrality - how many other airports can be reached from here
        network_reach = unique_dests / max(len(airport_df), 1)  # As a ratio of all possible destinations

        # Combined hub score (weighted components)
        hub_score_value = (
                (0.3 * balance_ratio) +  # Balance between in/out
                (0.4 * scale_factor) +  # Size of operation
                (0.3 * network_reach)  # Network coverage
        )
        airport_df.at[idx, 'Hub_Score'] = float(hub_score_value)  # Explicitly cast to float

        # Store component scores for potential analysis
        hub_component_scores.append({
            'airport_id': airport_id,
            'balance_ratio': balance_ratio,
            'scale_factor': scale_factor,
            'network_reach': network_reach,
            'total_hub_score': hub_score_value
        })

    # Store component scores as a DataFrame for reference
    hub_components_df = pd.DataFrame(hub_component_scores)

    # Save component scores for deeper analysis if needed
    hub_components_df.to_csv(os.path.join(output_dir, 'hub_components.csv'), index=False)

    return airport_df



# Calculate coastal classification based on proximity to shoreline
def calculate_coastal_classification(airports_df):
    """Classify airports as coastal or inland based on latitude/longitude"""
    try:
        # This is a simplified approach - in a production environment you'd use
        # actual coastline data for more accuracy
        coastal_classification = []

        for _, airport in airports_df.iterrows():
            # This is a very simplified example - ideally you would have actual coastline data
            # For now, we'll just use a dummy classification based on airports near common
            # coastal longitudes/latitudes around continents

            lat = airport['Latitude']
            lon = airport['Longitude']

            # Very simplified coastal detection
            # Examples: US East/West coasts, European Atlantic coast, etc.
            is_coastal = (
                    (-125 <= lon <= -115 and 30 <= lat <= 50) or  # US West Coast
                    (-85 <= lon <= -65 and 25 <= lat <= 45) or  # US East Coast
                    (-10 <= lon <= 3 and 35 <= lat <= 60) or  # Western Europe
                    (120 <= lon <= 150 and 30 <= lat <= 45)  # East Asia
            )

            coastal_classification.append(1 if is_coastal else 0)

        airports_df['Is_Coastal'] = coastal_classification

        return airports_df
    except Exception as e:
        print(f"Error calculating coastal classification: {e}")
        traceback.print_exc()
        # Add default values in case of error
        airports_df['Is_Coastal'] = 0
        return airports_df


# Simulate population data for demonstration purposes
def simulate_population_proximity(airports_df):
    """
    Simulate population data for airports based on proximity to major urban centers.

    Args:
        airports_df (pandas.DataFrame): DataFrame containing airport information

    Returns:
        pandas.DataFrame: DataFrame with added population proximity columns
    """
    try:
        # In a real implementation, you would use actual population grid data
        # This function creates simulated population values proportional to:
        # 1. Proximity to known major urban centers
        # 2. Total routes (assuming airports serve populations proportionally)

        # Reference major city coordinates with population
        # As of 2025 to the millionth estimate
        major_cities = [
            # 25 Asian Cities
            {'name': 'Tokyo', 'lat': 35.6828, 'lon': 139.7595, 'pop': 37_036_200},
            {'name': 'Delhi', 'lat': 28.6139, 'lon': 77.2090, 'pop': 34_665_600},
            {'name': 'Shanghai', 'lat': 31.2304, 'lon': 121.4737, 'pop': 30_482_100},
            {'name': 'Dhaka', 'lat': 23.8103, 'lon': 90.4125, 'pop': 24_652_900},
            {'name': 'Beijing', 'lat': 39.9042, 'lon': 116.4074, 'pop': 22_596_500},
            {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777, 'pop': 22_089_000},
            {'name': 'Osaka', 'lat': 34.6937, 'lon': 135.5023, 'pop': 18_921_600},
            {'name': 'Chongqing', 'lat': 29.5630, 'lon': 106.5516, 'pop': 18_171_200},
            {'name': 'Karachi', 'lat': 24.8607, 'lon': 67.0011, 'pop': 18_076_800},
            {'name': 'Istanbul', 'lat': 41.0082, 'lon': 28.9784, 'pop': 16_236_700},
            {'name': 'Kolkata', 'lat': 22.5726, 'lon': 88.3639, 'pop': 15_845_200},
            {'name': 'Manila', 'lat': 14.5995, 'lon': 120.9842, 'pop': 15_230_600},
            {'name': 'Guangzhou', 'lat': 23.1291, 'lon': 113.2644, 'pop': 14_878_700},
            {'name': 'Lahore', 'lat': 31.5497, 'lon': 74.3436, 'pop': 14_825_800},
            {'name': 'Tianjin', 'lat': 39.3434, 'lon': 117.3616, 'pop': 14_704_100},
            {'name': 'Shenzhen', 'lat': 22.5431, 'lon': 114.0579, 'pop': 13_545_400},
            {'name': 'Chennai', 'lat': 13.0827, 'lon': 80.2707, 'pop': 12_336_000},
            {'name': 'Jakarta', 'lat': -6.2088, 'lon': 106.8456, 'pop': 11_634_100},
            {'name': 'Bangkok', 'lat': 13.7563, 'lon': 100.5018, 'pop': 11_391_700},
            {'name': 'Hyderabad', 'lat': 17.3850, 'lon': 78.4867, 'pop': 11_337_900},
            {'name': 'Nanjing', 'lat': 32.0603, 'lon': 118.7969, 'pop': 10_174_900},
            {'name': 'Seoul', 'lat': 37.5665, 'lon': 126.9780, 'pop': 10_025_800},
            {'name': 'Chengdu', 'lat': 30.5728, 'lon': 104.0668, 'pop': 9_998_870},
            {'name': 'Ho Chi Minh City', 'lat': 10.7769, 'lon': 106.7009, 'pop': 9_816_320},
            {'name': 'Tehran', 'lat': 35.6892, 'lon': 51.3890, 'pop': 9_729_740},
            # 26 European Cities
            {'name': 'London', 'lat': 51.5074, 'lon': -0.1278, 'pop': 9_000_000},
            {'name': 'Istanbul', 'lat': 41.0082, 'lon': 28.9784, 'pop': 15_840_900},
            {'name': 'Moscow', 'lat': 55.7558, 'lon': 37.6173, 'pop': 12_737_400},
            {'name': 'Paris', 'lat': 48.8566, 'lon': 2.3522, 'pop': 11_346_800},
            {'name': 'Madrid', 'lat': 40.4168, 'lon': -3.7038, 'pop': 6_810_530},
            {'name': 'Barcelona', 'lat': 41.3851, 'lon': 2.1734, 'pop': 5_733_250},
            {'name': 'Saint Petersburg', 'lat': 59.9343, 'lon': 30.3351, 'pop': 5_597_340},
            {'name': 'Rome', 'lat': 41.9028, 'lon': 12.4964, 'pop': 4_347_100},
            {'name': 'Berlin', 'lat': 52.5200, 'lon': 13.4050, 'pop': 3_580_190},
            {'name': 'Milan', 'lat': 45.4642, 'lon': 9.1900, 'pop': 3_167_450},
            {'name': 'Athens', 'lat': 37.9838, 'lon': 23.7275, 'pop': 3_155_320},
            {'name': 'Lisbon', 'lat': 38.7169, 'lon': -9.1399, 'pop': 3_028_270},
            {'name': 'Manchester', 'lat': 53.4808, 'lon': -2.2426, 'pop': 2_832_580},
            {'name': 'Kyiv', 'lat': 50.4501, 'lon': 30.5234, 'pop': 2_797_553},
            {'name': 'Birmingham', 'lat': 52.4862, 'lon': -1.8904, 'pop': 2_704_620},
            {'name': 'Naples', 'lat': 40.8518, 'lon': 14.2681, 'pop': 2_182_170},
            {'name': 'Brussels', 'lat': 50.8503, 'lon': 4.3517, 'pop': 2_141_520},
            {'name': 'Minsk', 'lat': 53.9006, 'lon': 27.5590, 'pop': 2_070_930},
            {'name': 'Vienna', 'lat': 48.2082, 'lon': 16.3738, 'pop': 2_005_500},
            {'name': 'Turin', 'lat': 45.0703, 'lon': 7.6869, 'pop': 1_809_850},
            {'name': 'Warsaw', 'lat': 52.2297, 'lon': 21.0122, 'pop': 1_800_230},
            {'name': 'Hamburg', 'lat': 53.5511, 'lon': 9.9937, 'pop': 1_787_710},
            {'name': 'Lyon', 'lat': 45.7640, 'lon': 4.8357, 'pop': 1_787_230},
            {'name': 'Budapest', 'lat': 47.4979, 'lon': 19.0402, 'pop': 1_782_240},
            {'name': 'Bucharest', 'lat': 44.4268, 'lon': 26.1025, 'pop': 1_758_700},
            # 11 US Cities
            {'name': 'New York City', 'lat': 40.7128, 'lon': -74.0060, 'pop': 18_800_000},
            {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437, 'pop': 13_000_000},
            {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298, 'pop': 9_500_000},
            {'name': 'Dallas', 'lat': 32.7767, 'lon': -96.7970, 'pop': 7_500_000},
            {'name': 'Houston', 'lat': 29.7604, 'lon': -95.3698, 'pop': 7_000_000},
            {'name': 'Washington DC', 'lat': 38.9072, 'lon': -77.0369, 'pop': 6_300_000},
            {'name': 'Miami', 'lat': 25.7617, 'lon': -80.1918, 'pop': 6_200_000},
            {'name': 'Philadelphia', 'lat': 39.9526, 'lon': -75.1652, 'pop': 6_100_000},
            {'name': 'Atlanta', 'lat': 33.7490, 'lon': -84.3880, 'pop': 6_000_000},
            {'name': 'San Francisco', 'lat': 37.7749, 'lon': -122.4194, 'pop': 4_700_000},
            {'name': 'Boston', 'lat': 42.3601, 'lon': -71.0589, 'pop': 4_900_000},
            # 10 South American Cities
            {'name': 'São Paulo', 'lat': -23.5505, 'lon': -46.6333, 'pop': 22_990_000},
            {'name': 'Lima', 'lat': -12.0464, 'lon': -77.0428, 'pop': 12_123_000},
            {'name': 'Bogotá', 'lat': 4.7110, 'lon': -74.0721, 'pop': 11_344_000},
            {'name': 'Rio de Janeiro', 'lat': -22.9068, 'lon': -43.1729, 'pop': 13_634_000},
            {'name': 'Santiago', 'lat': -33.4489, 'lon': -70.6693, 'pop': 7_445_000},
            {'name': 'Belo Horizonte', 'lat': -19.9167, 'lon': -43.9345, 'pop': 6_101_000},
            {'name': 'Brasília', 'lat': -15.8267, 'lon': -47.9218, 'pop': 5_156_000},
            {'name': 'Fortaleza', 'lat': -3.7172, 'lon': -38.5433, 'pop': 4_079_000},
            {'name': 'Medellín', 'lat': 6.2442, 'lon': -75.5812, 'pop': 3_999_000},
            {'name': 'Guayaquil', 'lat': -2.1700, 'lon': -79.9224, 'pop': 3_735_000},
            # 10 African Cities
            {'name': 'Lagos', 'lat': 6.5244, 'lon': 3.3792, 'pop': 24_000_000},
            {'name': 'Cairo', 'lat': 30.0444, 'lon': 31.2357, 'pop': 23_074_200},
            {'name': 'Kinshasa', 'lat': -4.4419, 'lon': 15.2663, 'pop': 17_400_000},
            {'name': 'Dar es Salaam', 'lat': -6.7924, 'lon': 39.2083, 'pop': 8_000_000},
            {'name': 'Johannesburg', 'lat': -26.2041, 'lon': 28.0473, 'pop': 6_600_000},
            {'name': 'Nairobi', 'lat': -1.2921, 'lon': 36.8219, 'pop': 5_700_000},
            {'name': 'Abidjan', 'lat': 5.3599, 'lon': -4.0083, 'pop': 5_300_000},
            {'name': 'Alexandria', 'lat': 31.2001, 'lon': 29.9187, 'pop': 5_200_000},
            {'name': 'Addis Ababa', 'lat': 9.0300, 'lon': 38.7400, 'pop': 5_000_000},
            {'name': 'Kano', 'lat': 12.0022, 'lon': 8.5919, 'pop': 4_900_000},
            # 5 Australian Cities
            {'name': 'Sydney', 'lat': -33.8688, 'lon': 151.2093, 'pop': 5_450_000},
            {'name': 'Melbourne', 'lat': -37.8136, 'lon': 144.9631, 'pop': 5_290_000},
            {'name': 'Brisbane', 'lat': -27.4698, 'lon': 153.0251, 'pop': 2_560_000},
            {'name': 'Perth', 'lat': -31.9505, 'lon': 115.8605, 'pop': 2_180_000},
            {'name': 'Adelaide', 'lat': -34.9285, 'lon': 138.6007, 'pop': 1_390_000},
            # 4 Middle East Cities
            {'name': 'Tehran', 'lat': 35.6892, 'lon': 51.3890, 'pop': 9_729_740},
            {'name': 'Baghdad', 'lat': 33.3152, 'lon': 44.3661, 'pop': 9_500_000},
            {'name': 'Riyadh', 'lat': 24.7136, 'lon': 46.6753, 'pop': 7_600_000},
            {'name': 'Jeddah', 'lat': 21.4858, 'lon': 39.1925, 'pop': 5_400_000},
        ]

        # Define population radius parameters
        radius_ranges = [
            {'name': 'Population_50km', 'radius': 50, 'noise_scale': 100000, 'min_factor': 1.0},
            {'name': 'Population_100km', 'radius': 100, 'noise_scale': 200000, 'min_factor': 1.2},
            {'name': 'Population_200km', 'radius': 200, 'noise_scale': 500000, 'min_factor': 1.5}
        ]

        # Initialize result dictionary
        population_results = {range_param['name']: [] for range_param in radius_ranges}

        # Process each airport
        for _, airport in airports_df.iterrows():
            airport_lat = airport['Latitude']
            airport_lon = airport['Longitude']

            # Store calculated population values for each radius
            radius_values = {}

            # Calculate population for each radius range
            for range_param in radius_ranges:
                radius_name = range_param['name']
                radius = range_param['radius']
                population_value = 0

                # Sum contributions from all cities within range
                for city in major_cities:
                    dist = haversine(airport_lat, airport_lon, city['lat'], city['lon'])

                    # Add population based on distance from city (linear falloff)
                    if dist <= radius:
                        population_value += city['pop'] * (1 - dist / radius)

                # Add realistic noise
                population_value = max(0, population_value + np.random.normal(0, range_param['noise_scale']))
                radius_values[radius_name] = population_value

            # Ensure logical population progression (larger radius must have more population)
            # Start from the smallest radius and enforce minimum factors
            sorted_ranges = sorted(radius_ranges, key=lambda x: x['radius'])
            for i in range(1, len(sorted_ranges)):
                smaller_range = sorted_ranges[i - 1]['name']
                current_range = sorted_ranges[i]['name']
                min_factor = sorted_ranges[i]['min_factor']

                # Ensure larger radius has at least min_factor times the population of smaller radius
                radius_values[current_range] = max(
                    radius_values[current_range],
                    radius_values[smaller_range] * min_factor
                )

            # Add calculated values to results
            for range_param in radius_ranges:
                population_results[range_param['name']].append(int(radius_values[range_param['name']]))

        # Add population columns to dataframe
        for range_param in radius_ranges:
            airports_df[range_param['name']] = population_results[range_param['name']]

        return airports_df

    except Exception as e:
        print(f"Error simulating population proximity: {e}")
        traceback.print_exc()

        # Add default values in case of error
        for range_param in radius_ranges:
            airports_df[range_param['name']] = 0

        return airports_df


# Apply all the enhancements to the airport features with error handling
try:
    print("Calculating city proximity metrics...")
    airport_features = calculate_city_proximity(airport_features)

    print("Calculating network metrics...")
    airport_features = calculate_network_metrics(airport_features, routes_df)

    print("Simulating population proximity data...")
    airport_features = simulate_population_proximity(airport_features)
except Exception as e:
    print(f"Error during feature calculation: {e}")
    traceback.print_exc()

try:
    # Create enhanced features for training
    enhanced_features = [
        'Latitude', 'Longitude', 'Altitude',
        'Distance_to_EconCenter', 'Unique_Destinations', 'Connection_Ratio', 'Hub_Score',
        'Is_Coastal', 'Population_50km', 'Population_100km', 'Population_200km',
        'Region_North America', 'Region_South America', 'Region_Asia Pacific',
        'Region_Europe', 'Region_Middle East', 'Region_Africa', 'Region_Other'
    ]

    # Use only available columns
    available_features = [col for col in enhanced_features if col in airport_features.columns]
    print(f"Using {len(available_features)} features: {available_features}")

    X = airport_features[available_features].values
    y = airport_features['TotalRoutes'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest regressor
    print("Training Random Forest regressor...")
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
except Exception as e:
    print(f"Error during model training: {e}")
    traceback.print_exc()
    raise

# Plot feature importance with error handling
try:
    feature_importances = pd.DataFrame({
        'Feature': available_features,
        'Importance': regr.feature_importances_
    })
    feature_importances = feature_importances.sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'])
    plt.title('Feature Importance for Airport Traffic Prediction', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    save_visualization(plt.gcf(), 'feature_importance.png')
except Exception as e:
    print(f"Error creating feature importance plot: {e}")
    traceback.print_exc()

# Visualize predictions vs actual with error handling
try:
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.xlabel('Actual Traffic (Total Routes)', fontsize=14)
    plt.ylabel('Predicted Traffic (Total Routes)', fontsize=14)
    plt.title('Actual vs Predicted Airport Traffic', fontsize=16)
    plt.tight_layout()
    save_visualization(plt.gcf(), 'prediction_accuracy.png')
except Exception as e:
    print(f"Error creating prediction accuracy plot: {e}")
    traceback.print_exc()

# Generate Map Visualizations

# 1. Relationship between population and airport traffic
try:
    plt.figure(figsize=(12, 8))
    plt.scatter(airport_features['Population_100km'], airport_features['TotalRoutes'],
                alpha=0.5, c=airport_features['Hub_Score'], cmap='viridis')
    plt.colorbar(label='Hub Score')
    plt.xlabel('Population within 100km', fontsize=14)
    plt.ylabel('Total Routes', fontsize=14)
    plt.title('Population vs. Airport Traffic', fontsize=16)
    plt.tight_layout()
    save_visualization(plt.gcf(), 'population_vs_traffic.png')
except Exception as e:
    print(f"Error creating population vs traffic plot: {e}")
    traceback.print_exc()

# 2. Traffic distribution by region
try:
    regional_traffic = airport_features.groupby('Region')['TotalRoutes'].agg(['mean', 'median', 'count'])
    regional_traffic = regional_traffic.sort_values('mean', ascending=False)

    plt.figure(figsize=(12, 8))
    ax = regional_traffic['mean'].plot(kind='bar', color='skyblue')
    plt.axhline(y=regional_traffic['mean'].mean(), color='r', linestyle='-', label='Global Average')
    plt.title('Average Airport Traffic by Region', fontsize=16)
    plt.xlabel('Region', fontsize=14)
    plt.ylabel('Average Number of Routes', fontsize=14)
    plt.legend()
    plt.xticks(rotation=45)

    # Add count annotations
    for i, v in enumerate(regional_traffic['count']):
        ax.text(i, 5, f"n={v}", ha='center', fontsize=10)

    plt.tight_layout()
    save_visualization(plt.gcf(), 'regional_traffic.png')
except Exception as e:
    print(f"Error creating regional traffic plot: {e}")
    traceback.print_exc()

# 3. Relationship between Distance to Economic Center and Traffic
try:
    plt.figure(figsize=(12, 8))
    plt.scatter(airport_features['Distance_to_EconCenter'], airport_features['TotalRoutes'],
                alpha=0.6, c=airport_features['Population_100km'], cmap='plasma',
                norm=plt.Normalize(vmin=0, vmax=20000000))
    plt.colorbar(label='Population within 100km')
    plt.xlabel('Distance to Nearest Economic Center (km)', fontsize=14)
    plt.ylabel('Total Routes', fontsize=14)
    plt.title('Distance to Economic Centers vs. Airport Traffic', fontsize=16)
    plt.tight_layout()
    save_visualization(plt.gcf(), 'distance_vs_traffic.png')
except Exception as e:
    print(f"Error creating distance vs traffic plot: {e}")
    traceback.print_exc()

# 4. Visualization for Hub Score Component
try:
    plt.figure(figsize=(12, 8))

    # Get top 20 airports by hub score
    top_hubs = airport_features.nlargest(20, 'Hub_Score')

    # Load hub components data
    hub_components_df = pd.read_csv(os.path.join(output_dir, 'hub_components.csv'))

    # Filter to top hub airports
    top_components = hub_components_df[hub_components_df['airport_id'].isin(top_hubs['AirportID'])]
    top_components = pd.merge(
        top_components,
        top_hubs[['AirportID', 'IATA']],
        left_on='airport_id',
        right_on='AirportID'
    )
    top_components = top_components.sort_values('total_hub_score', ascending=False)

    # Create stacked bar chart
    bar_width = 0.8
    index = np.arange(len(top_components))

    # Plot each component with its contribution to the score
    plt.bar(index, top_components['balance_ratio'] * 0.3, bar_width,
            label='Balance Factor (30%)', color='royalblue')
    plt.bar(index, top_components['scale_factor'] * 0.4, bar_width,
            bottom=top_components['balance_ratio'] * 0.3,
            label='Scale Factor (40%)', color='darkorange')
    plt.bar(index, top_components['network_reach'] * 0.3, bar_width,
            bottom=(top_components['balance_ratio'] * 0.3) + (top_components['scale_factor'] * 0.4),
            label='Network Reach (30%)', color='forestgreen')

    # Set labels and title
    plt.title('Hub Score Components for Top 20 Airports', fontsize=16)
    plt.xlabel('Airport', fontsize=14)
    plt.ylabel('Hub Score Component Contribution', fontsize=14)
    plt.xticks(index, top_components['IATA'], rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    save_visualization(plt.gcf(), 'hub_score_components.png')
except Exception as e:
    print(f"Error creating hub component visualization: {e}")
    traceback.print_exc()


# 5. Altitude effect on airport traffic
try:
    plt.figure(figsize=(12, 8))
    plt.scatter(airport_features['Altitude'], airport_features['TotalRoutes'],
                alpha=0.6, c=airport_features['Population_100km'], cmap='viridis')
    plt.colorbar(label='Population within 100km')
    plt.xlabel('Altitude (feet)', fontsize=14)
    plt.ylabel('Total Routes', fontsize=14)
    plt.title('Altitude vs. Airport Traffic', fontsize=16)
    plt.tight_layout()
    save_visualization(plt.gcf(), 'altitude_vs_traffic.png')
except Exception as e:
    print(f"Error creating altitude vs traffic plot: {e}")
    traceback.print_exc()

print(f"Visualization process completed. Check {output_dir} for saved files.")

# Create a summary of insights
try:
    top_10_important_features = feature_importances.head(10)
    print("\nTop 10 most important features for predicting airport traffic:")
    for idx, row in top_10_important_features.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    # Save feature importance values to a text file for reference
    with open(os.path.join(output_dir, 'feature_importance_summary.txt'), 'w') as f:
        f.write("Top 10 most important features for predicting airport traffic:\n")
        for idx, row in top_10_important_features.iterrows():
            f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    print(f"Feature importance summary saved to {output_dir}/feature_importance_summary.txt")
except Exception as e:
    print(f"Error creating summary: {e}")
    traceback.print_exc()

# Cartopy Visualizations

try:
    print("Generating Cartopy global visualizations...")

    # Import necessary libraries for cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


    # Function to create a base map with Robinson projection
    def create_robinson_map(figsize=(15, 10)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

        # Add natural earth features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

        # Add gridlines
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle=':')

        # Set global extent
        ax.set_global()

        return fig, ax


    # 1. Global Airport Traffic Distribution
    fig, ax = create_robinson_map()

    # Filter to top 500 airports by traffic to avoid overcrowding
    top_airports = airport_features.nlargest(500, 'TotalRoutes')

    # Create scatter plot with size proportional to traffic
    scatter = ax.scatter(
        top_airports['Longitude'],
        top_airports['Latitude'],
        transform=ccrs.PlateCarree(),
        c=top_airports['TotalRoutes'],
        s=top_airports['TotalRoutes'] / 20,  # Scale down size for visibility
        cmap='viridis',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('Total Routes', fontsize=12)

    plt.title('Global Distribution of Airport Traffic (Top 500 Airports)', fontsize=16)
    save_visualization(fig, 'global_airport_traffic.png')

    # 2. Regional Hub Analysis - Update to use new Hub Score
    fig, ax = create_robinson_map()

    # Filter for significant hub airports (high hub score and traffic)
    hub_threshold = np.percentile(airport_features['Hub_Score'], 80)
    traffic_threshold = np.percentile(airport_features['TotalRoutes'], 80)

    hub_airports = airport_features[
        (airport_features['Hub_Score'] >= hub_threshold) &
        (airport_features['TotalRoutes'] >= traffic_threshold)
        ]

    # Create scatter with size proportional to hub score
    scatter = ax.scatter(
        hub_airports['Longitude'],
        hub_airports['Latitude'],
        transform=ccrs.PlateCarree(),
        c=hub_airports['Hub_Score'],
        s=hub_airports['TotalRoutes'] / 10,
        cmap='spring',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('Hub Score (Balance, Scale, Network)', fontsize=12)

    plt.title('Global Distribution of Major Airport Hubs', fontsize=16)
    save_visualization(fig, 'global_airport_hubs.png')

    # 3. Population Density vs Airport Traffic
    fig, ax = create_robinson_map()

    # Filter for airports with significant population and traffic
    pop_airports = airport_features.nlargest(400, 'Population_100km')

    # Create scatter with size proportional to traffic and color by population
    scatter = ax.scatter(
        pop_airports['Longitude'],
        pop_airports['Latitude'],
        transform=ccrs.PlateCarree(),
        c=pop_airports['Population_100km'],
        s=pop_airports['TotalRoutes'] / 15,
        cmap='YlOrRd',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('Population within 100km', fontsize=12)

    plt.title('Population Density and Airport Traffic', fontsize=16)
    save_visualization(fig, 'population_and_traffic_global.png')

    # 4. Major Routes Network Visualization (simplified version)
    fig, ax = create_robinson_map(figsize=(18, 12))

    # Get the top 50 airports by traffic for network visualization
    top_network_airports = airport_features.nlargest(50, 'TotalRoutes')

    # Plot the nodes (airports)
    ax.scatter(
        top_network_airports['Longitude'],
        top_network_airports['Latitude'],
        transform=ccrs.PlateCarree(),
        s=top_network_airports['TotalRoutes'] / 10,
        c='orchid',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.8,
        zorder=3
    )

    # Filter routes to only those connecting top airports (to avoid overwhelming the map)
    top_airport_ids = top_network_airports['AirportID'].tolist()

    top_routes = routes_df[
        (routes_df['SourceAirportID'].isin(top_airport_ids)) &
        (routes_df['DestinationAirportID'].isin(top_airport_ids))
        ]

    # Create a dictionary to lookup airport coordinates
    airport_coord_dict = dict(zip(
        airport_features['AirportID'],
        zip(airport_features['Longitude'], airport_features['Latitude'])
    ))

    # Plot up to 200 major routes to avoid cluttering
    route_count = 0
    max_routes = 200

    # Get the most frequent routes first
    route_pairs = top_routes.groupby(['SourceAirportID', 'DestinationAirportID']).size().reset_index(name='count')
    route_pairs = route_pairs.sort_values('count', ascending=False)

    for _, route in route_pairs.iterrows():
        if route_count >= max_routes:
            break

        source_id = route['SourceAirportID']
        dest_id = route['DestinationAirportID']

        # Skip if we don't have coordinates
        if source_id not in airport_coord_dict or dest_id not in airport_coord_dict:
            continue

        # Get coordinates
        source_lon, source_lat = airport_coord_dict[source_id]
        dest_lon, dest_lat = airport_coord_dict[dest_id]

        # Draw route line
        ax.plot(
            [source_lon, dest_lon],
            [source_lat, dest_lat],
            transform=ccrs.Geodetic(),
            color='skyblue',
            alpha=0.3,
            linewidth=0.8,
            zorder=2
        )

        route_count += 1

    plt.title('Major Global Air Route Network', fontsize=16)
    save_visualization(fig, 'global_route_network.png')

    # 5. Predicted vs Actual Traffic by Geographic Region
    fig, ax = create_robinson_map()

    # Get prediction accuracy by region
    X_for_viz = airport_features[available_features].values
    y_for_viz = airport_features['TotalRoutes'].values
    y_pred_viz = regr.predict(X_for_viz)

    # Calculate prediction error
    airport_features['PredictedTraffic'] = y_pred_viz
    airport_features['PredictionError'] = (airport_features['PredictedTraffic'] - airport_features['TotalRoutes']) / (
                airport_features['TotalRoutes'] + 1)  # Avoid division by zero

    # Filter to airports with significant traffic
    traffic_viz_airports = airport_features[airport_features['TotalRoutes'] > 20].nlargest(300, 'TotalRoutes')

    # Create scatter with color representing prediction error
    scatter = ax.scatter(
        traffic_viz_airports['Longitude'],
        traffic_viz_airports['Latitude'],
        transform=ccrs.PlateCarree(),
        c=traffic_viz_airports['PredictionError'],
        s=traffic_viz_airports['TotalRoutes'] / 15,
        cmap='coolwarm',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
        vmin=-0.5,  # Underprediction
        vmax=0.5,  # Overprediction
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('Prediction Error (negative = underprediction)', fontsize=12)

    plt.title('Geographic Distribution of Traffic Prediction Accuracy', fontsize=16)
    save_visualization(fig, 'prediction_error_global.png')

except Exception as e:
    print(f"Error creating Cartopy visualizations: {e}")
    traceback.print_exc()


print("\nAnalysis complete.")
