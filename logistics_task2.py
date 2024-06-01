import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gurobipy import Model, GRB, quicksum

# Read data
file_path_juan = "/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A1_1500_1.csv"
file_path_daila = "/Users/dailagencarelli/Documents/Design4Transport&Logistic/instance_set/Set A/A1_1500_1.csv"

coordinates_district = pd.read_csv(file_path_juan)

pickup_coords = coordinates_district.iloc[:1500].copy()  
delivery_coords = coordinates_district.iloc[1500:].copy()  

pickup_coords_A = pickup_coords[pickup_coords['Origin'] == 1].copy()
pickup_coords_B = pickup_coords[pickup_coords['Origin'] == 2].copy()
delivery_coords_A = delivery_coords[delivery_coords['Origin'] == 1].copy()
delivery_coords_B = delivery_coords[delivery_coords['Origin'] == 2].copy()

coords_A = pd.concat([pickup_coords_A, delivery_coords_A], axis=0).copy()

# Translation vector for city B
translation_vector = np.array([300, 300])

# Applying translation to the copies of the DataFrame
pickup_coords_B[['X', 'Y']] += translation_vector
delivery_coords_B[['X', 'Y']] += translation_vector

# Reconstruct coords_B after translation
coords_B = pd.concat([pickup_coords_B, delivery_coords_B], axis=0).copy()

# Perform KMeans clustering for district A
kmeans_A = KMeans(n_clusters=4)
kmeans_A.fit(coords_A[['X', 'Y']])
coords_A['Cluster'] = kmeans_A.labels_

# Perform KMeans clustering for district B
kmeans_B = KMeans(n_clusters=4)
kmeans_B.fit(coords_B[['X', 'Y']])
coords_B['Cluster'] = kmeans_B.labels_

# Calculate centroids for each district
hub_A = coords_A[['X', 'Y']].mean()
hub_B = coords_B[['X', 'Y']].mean()

# Calculate centroids for each cluster
satellites_A = coords_A.groupby('Cluster').mean()
satellites_B = coords_B.groupby('Cluster').mean()

def plot_clustered_coordinates_with_hub_and_satellites(coords, hub, satellites, district_name):
    plt.figure(figsize=(10, 6))
    for cluster in range(4):
        cluster_coords = coords[coords['Cluster'] == cluster]
        plt.scatter(cluster_coords['X'], cluster_coords['Y'], label=f'Cluster {cluster+1}')
    plt.scatter(hub['X'], hub['Y'], color='blue', marker='x', s=100, label='Hub')
    for i, sat in satellites.iterrows():
        plt.scatter(sat['X'], sat['Y'], color='black', marker='*', s=100, label=f'Satellite {i+1}')
    plt.title(f"Clustered Coordinates District {district_name} with Hub and Satellites")
    plt.grid(True)
    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

plot_clustered_coordinates_with_hub_and_satellites(coords_A, hub_A, satellites_A, 'A')
plot_clustered_coordinates_with_hub_and_satellites(coords_B, hub_B, satellites_B, 'B')

Pickups_to_satellites=[]
for cluster_id, satellite_coords in satellites_A.iterrows():
    cluster_pickups= coords_A[coords_A['Cluster'] == cluster_id][['X', 'Y']] # Filter pickup coordinates for the current cluster
    # Assign each pickup in the cluster to the corresponding satellite
    for pickup_id in cluster_pickups.index:
        Pickups_to_satellites.append([pickup_id, satellite_coords.name])

print("Pickup-Delivery pairs:")
print(Pickups_to_satellites)

# Time window
pickup_coords_A = pickup_coords_A.dropna(subset=['Early', 'Latest']).copy()
# Definizione dei vincoli di tempo (time windows) per i pickup
pickup_time_windows_A = []
for _, pickup in pickup_coords_A.iterrows():
    pickup_time_windows_A.append((pickup['Early'], pickup['Latest']))
print("These are the pickup time windows", pickup_time_windows_A)
pickup_demands_A = pickup_coords_A['Demand'].values
print("This is the demand per pickup point",pickup_demands_A)
pickup_demands_A_sum = np.sum(pickup_demands_A)
print("Total demand for district A:", pickup_demands_A_sum)

# Vehicle capacities
vehicle_capacity = 20
num_vehicles = 100
fixed_cost_per_km=1
fixed_cost_to_open=15
fixed_cost_per_vehicle=5

# Define truck capacity and demand
truck_capacity = 200 
daily_demand_A_to_B = pickup_coords_A['Demand'].sum()  # Total daily demand from A to B
daily_demand_B_to_A = pickup_coords_B['Demand'].sum()  # Total daily demand from B to A

# Number of trucks needed (considering rounding up)
trucks_needed_A_to_B = int(np.ceil(daily_demand_A_to_B / truck_capacity))
trucks_needed_B_to_A = int(np.ceil(daily_demand_B_to_A / truck_capacity))

# Create Gurobi model
model = Model('ParcelAssignment')

# Variables
x_A_to_B = model.addVars(trucks_needed_A_to_B, vtype=GRB.BINARY, name="x_A_to_B")
x_B_to_A = model.addVars(trucks_needed_B_to_A, vtype=GRB.BINARY, name="x_B_to_A")

# Objective: Minimize the number of trucks used
model.setObjective(quicksum(x_A_to_B) + quicksum(x_B_to_A), GRB.MINIMIZE)

# Constraints
model.addConstr(quicksum(x_A_to_B) * truck_capacity >= daily_demand_A_to_B, "Demand_A_to_B")
model.addConstr(quicksum(x_B_to_A) * truck_capacity >= daily_demand_B_to_A, "Demand_B_to_A")

# Optimize model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal number of trucks from A to B: {sum(x_A_to_B[i].x for i in range(trucks_needed_A_to_B))}")
    print(f"Optimal number of trucks from B to A: {sum(x_B_to_A[i].x for i in range(trucks_needed_B_to_A))}")

# For more detailed analysis, print the values of each variable
for v in model.getVars():
    print(f"{v.varName}: {v.x}")
