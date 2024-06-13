import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gurobipy import Model, GRB, quicksum

# Load coordinates from the CSV file
file_path_juan = "/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A1_1500_1.csv"
file_path_daila = "/Users/dailagencarelli/Documents/Design4Transport&Logistic/instance_set/Set A/A1_1500_1.csv"

coordinates_district = pd.read_csv(file_path_juan)

# Split coordinates into pickup and delivery
pickup_coords = coordinates_district.iloc[:1500].copy()
delivery_coords = coordinates_district.iloc[1500:].copy()

# Split into districts
pickup_coords_A = pickup_coords[pickup_coords['Origin'] == 1].copy()
pickup_coords_B = pickup_coords[pickup_coords['Origin'] == 2].copy()
delivery_coords_A = delivery_coords[delivery_coords['Origin'] == 1].copy()
delivery_coords_B = delivery_coords[delivery_coords['Origin'] == 2].copy()

# Combine pickup and delivery coordinates for each district
coords_A = pd.concat([pickup_coords_A, delivery_coords_A], axis=0).copy()
coords_B = pd.concat([pickup_coords_B, delivery_coords_B], axis=0).copy()

# Translate coordinates for district B
translation_vector = np.array([300, 300])
coords_B[['X', 'Y']] += translation_vector

# Calculate centroids for each district
hub_A = coords_A[['X', 'Y']].mean()
hub_B = coords_B[['X', 'Y']].mean()

# Perform KMeans clustering
def perform_kmeans(coords, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters)
    coords['Cluster'] = kmeans.fit_predict(coords[['X', 'Y']])
    satellites = coords.groupby('Cluster')[['X', 'Y']].mean()
    return coords, satellites

coords_A, satellites_A = perform_kmeans(coords_A)
coords_B, satellites_B = perform_kmeans(coords_B)

# Plot clustered coordinates with hub and satellites
def plot_clustered_coordinates(coords, hub, satellites, district_name):
    plt.figure(figsize=(10, 6))
    for cluster in range(4):
        cluster_coords = coords[coords['Cluster'] == cluster]
        plt.scatter(cluster_coords['X'], cluster_coords['Y'], label=f'Cluster {cluster+1}')
    plt.scatter(hub['X'], hub['Y'], color='blue', marker='x', s=100, label='Hub')
    plt.scatter(satellites['X'], satellites['Y'], color='black', marker='*', s=100, label='Satellites')
    plt.title(f"Clustered Coordinates District {district_name} with Hub and Satellites")
    plt.grid(True)
    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

plot_clustered_coordinates(coords_A, hub_A, satellites_A, 'A')
plot_clustered_coordinates(coords_B, hub_B, satellites_B, 'B')

# Pickup to satellite assignment
Pickups_to_satellites = []
for cluster_id, satellite_coords in satellites_A.iterrows():
    cluster_pickups = coords_A[coords_A['Cluster'] == cluster_id][['X', 'Y']]
    for pickup_id in cluster_pickups.index:
        Pickups_to_satellites.append([pickup_id, cluster_id])

print("Pickup-Delivery pairs:")
print(Pickups_to_satellites)

# Filter and process pickup coordinates
pickup_coords_A = pickup_coords_A.dropna(subset=['Early', 'Latest']).copy()
pickup_time_windows_A = list(pickup_coords_A[['Early', 'Latest']].itertuples(index=False, name=None))
pickup_demands_A = pickup_coords_A['Demand'].values

print("These are the pickup time windows", pickup_time_windows_A)
print("This is the demand per pickup point", pickup_demands_A)
print("Total demand for district A:", pickup_demands_A.sum())

# Define truck capacity and demand
truck_capacity = 100
cost_per_unit_distance = 1
daily_demand_A_to_B = pickup_coords_A['Demand'].sum()
daily_demand_B_to_A = pickup_coords_B['Demand'].sum()

# Define sets and parameters for optimization
H = [0, 1]  # Hubs
R = [(i, j) for i in H for j in H if i != j]  # Routes
V = range(10)  # Vehicles
P = range(1500)  # Packages
D = {0: daily_demand_A_to_B, 1: daily_demand_B_to_A}  # Demand at each hub
C = {v: truck_capacity for v in V}  # Capacity of each vehicle
M = {(i, j): np.linalg.norm(hub_A - hub_B) if (i, j) in R else 0 for i in H for j in H}  # Distance matrix

# Gurobi optimization model
model = Model('ParcelAssignment')

x = model.addVars(R, V, vtype=GRB.BINARY, name="x")
f = model.addVars(R, V, vtype=GRB.CONTINUOUS, name="f")

model.setObjective(quicksum(M[i, j] * x[i, j, v] * cost_per_unit_distance for (i, j) in R for v in V), GRB.MINIMIZE)

# Constraints

# 1. Flow conservation
for i in H:
    model.addConstr(
        quicksum(f[i, j, v] for j in H if (i, j) in R for v in V) ==
        quicksum(f[j, i, v] for j in H if (j, i) in R for v in V) + D[i],
        name=f"flow_conservation_{i}"
    )

# 2. Capacity constraints
for (i, j) in R:
    for v in V:
        model.addConstr(f[i, j, v] <= C[v] * x[i, j, v], name=f"capacity_{i}_{j}_{v}")

# 3. Positive or zero flow variables
for (i, j) in R:
    for v in V:
        model.addConstr(f[i, j, v] >= 0, name=f"positive_flow_{i}_{j}_{v}")

# 4. Demand satisfaction
for (i, j) in R:
    model.addConstr(
        quicksum(C[v] * x[i, j, v] for v in V) >= sum(D.values()),
        name=f"demand_satisfaction_{i}_{j}"
    )

# 5. Binary decision variables
for (i, j) in R:
    for v in V:
        model.addConstr(x[i, j, v] <= 1, name=f"x_binary_{i}_{j}_{v}")

# Optimize model
model.optimize()

if model.status == GRB.OPTIMAL:
    trucks_A_to_B = sum(x[i, j, v].x for (i, j) in R for v in V if i == 0 and j == 1)
    trucks_B_to_A = sum(x[i, j, v].x for (i, j) in R for v in V if i == 1 and j == 0)
    print(f"Optimal number of trucks from A to B: {trucks_A_to_B}")
    print(f"Optimal number of trucks from B to A: {trucks_B_to_A}")

# Scenario analysis function
def scenario_analysis(truck_capacity, demand_multiplier):
    model = Model('ParcelAssignment')
    daily_demand_A_to_B_adj = daily_demand_A_to_B * demand_multiplier
    daily_demand_B_to_A_adj = daily_demand_B_to_A * demand_multiplier
    trucks_needed_A_to_B = int(np.ceil(daily_demand_A_to_B_adj / truck_capacity))
    trucks_needed_B_to_A = int(np.ceil(daily_demand_B_to_A_adj / truck_capacity))

    x_A_to_B = model.addVars(trucks_needed_A_to_B, vtype=GRB.BINARY, name="x_A_to_B")
    x_B_to_A = model.addVars(trucks_needed_B_to_A, vtype=GRB.BINARY, name="x_B_to_A")

    model.setObjective(quicksum(x_A_to_B) + quicksum(x_B_to_A), GRB.MINIMIZE)

    model.addConstr(quicksum(x_A_to_B) * truck_capacity >= daily_demand_A_to_B_adj, "Demand_A_to_B")
    model.addConstr(quicksum(x_B_to_A) * truck_capacity >= daily_demand_B_to_A_adj, "Demand_B_to_A")

    model.optimize()

    trucks_A_to_B = sum(x_A_to_B[i].x for i in range(trucks_needed_A_to_B))
    trucks_B_to_A = sum(x_B_to_A[i].x for i in range(trucks_needed_B_to_A))

    return trucks_A_to_B, trucks_B_to_A

# Example scenario analysis
scenarios = [
    {'truck_capacity': 100, 'demand_multiplier': 1.0},
    {'truck_capacity': 150, 'demand_multiplier': 1.0},
    {'truck_capacity': 200, 'demand_multiplier': 1.0},
    {'truck_capacity': 250, 'demand_multiplier': 1.0},
]

results = []
for scenario in scenarios:
    trucks_A_to_B, trucks_B_to_A = scenario_analysis(scenario['truck_capacity'], scenario['demand_multiplier'])
    results.append({
        'truck_capacity': scenario['truck_capacity'],
        'demand_multiplier': scenario['demand_multiplier'],
        'trucks_A_to_B': trucks_A_to_B,
        'trucks_B_to_A': trucks_B_to_A
    })

results_df = pd.DataFrame(results)

def plot_scenario_results(results_df):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(results_df.index, results_df['trucks_A_to_B'], marker='o', label='Trucks A to B')
    ax.plot(results_df.index, results_df['trucks_B_to_A'], marker='o', label='Trucks B to A')
    ax.set_xticks(results_df.index)
    ax.set_xticklabels([f"Capacity: {row['truck_capacity']}\nDemand Multiplier: {row['demand_multiplier']}" for _, row in results_df.iterrows()], rotation=45, ha='right')
    ax.set_title('Scenario Analysis Results')
    ax.set_xlabel('Scenarios')
    ax.set_ylabel('Number of Trucks')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

plot_scenario_results(results_df)
print(results_df)

# Extract the "Day-Ops" column
day_ops = coordinates_district['Day-Ops']

# Initialize an array to count the number of packages per day
day_counts = np.zeros(10)  # Assuming 10 days

# Go through the "Day-Ops" column and update the array
for day in day_ops:
    if not np.isnan(day) and 1 <= day <= 10:
        day_counts[int(day)-1] += 1

# Calculate the proportion of packages per day
day_proportions = day_counts / day_counts.sum()

# Distribute the trucks proportionally
trucks_per_day = np.round(day_proportions * 38)

# Adjust the distribution to make sure the total number of trucks is equal to 38
while trucks_per_day.sum() > 38:
    max_index = np.argmax(trucks_per_day)
    trucks_per_day[max_index] -= 1

while trucks_per_day.sum() < 38:
    min_index = np.argmin(trucks_per_day)
    trucks_per_day[min_index] += 1

print("The trucks needed for each day are:")

# Print the results in the desired format
for i, trucks in enumerate(trucks_per_day, start=1):
    print(f'Day {i}: {trucks}')