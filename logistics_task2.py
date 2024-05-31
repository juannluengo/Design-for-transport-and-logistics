# coordinates_district = pd.read_csv("/Users/dailagencarelli/Documents/Design4Transport&Logistic/instance_set/Set A/A1_1500_1.csv")
# coordinates_district = pd.read_csv("/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A1_1500_1.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linprog

# Read data
coordinates_district = pd.read_csv("/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A1_1500_1.csv")
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

# Pickups to satellites assignment
Pickups_to_satellites = []
for cluster_id, satellite_coords in satellites_A.iterrows():
    cluster_pickups = coords_A[coords_A['Cluster'] == cluster_id][['X', 'Y']] # Filter pickup coordinates for the current cluster
    for pickup_id in cluster_pickups.index:
        Pickups_to_satellites.append([pickup_id, satellite_coords.name])

print("Pickup-Delivery pairs:")
print(Pickups_to_satellites)

# Time window
pickup_coords_A = pickup_coords_A.dropna(subset=['Early', 'Latest']).copy()
pickup_time_windows_A = [(row['Early'], row['Latest']) for _, row in pickup_coords_A.iterrows()]
print("These are the pickup time windows", pickup_time_windows_A)
pickup_demands_A = pickup_coords_A['Demand'].values
print("This is the demand per pickup point", pickup_demands_A)
pickup_demands_A_sum = np.sum(pickup_demands_A)
print("Total demand for district A:", pickup_demands_A_sum)

# Truck capacities and demands
truck_capacity = 100
num_trucks = 10000  # Significantly increase the number of trucks
fixed_cost_per_km = 1
fixed_cost_per_truck = 5

hub_A_demand = pickup_demands_A_sum
hub_B_demand = np.sum(delivery_coords_B['Demand'].values)

# Linear programming setup
c = []  # Costs array
A_eq = []  # Equality constraints matrix
b_eq = []  # Equality constraints vector

# Decision variables (binary)
x = []

# Cost setup
for t in range(num_trucks):
    c.append(fixed_cost_per_truck)

# Capacity and utilization constraints
for t in range(num_trucks):
    row = [0] * num_trucks
    row[t] = truck_capacity
    A_eq.append(row)
    b_eq.append(0)  # Initially set to 0, will be adjusted later

# Demand satisfaction constraint
demand_constraint = [1] * num_trucks
A_eq.append(demand_constraint)
b_eq.append(max(hub_A_demand, hub_B_demand))

# Constraint to limit the number of trucks used
truck_usage_constraint = [1] * num_trucks
A_eq.append(truck_usage_constraint)
b_eq.append((hub_A_demand + hub_B_demand) // truck_capacity)

# Bounds for decision variables (0 or 1)
bounds = [(0, 1) for _ in range(num_trucks)]

# Linear programming problem solving
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

# Results
if result.success:
    print("Optimal truck assignments:", result.x)
else:
    print("No optimal solution found.")

print("Minimized cost:", result.fun)
