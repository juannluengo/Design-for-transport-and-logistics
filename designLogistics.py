import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import random
import gurobipy as gp
from gurobipy import GRB
random.seed(42)
np.random.seed(42)

# Load data and display pickup and delivery points for District A
# coordinates_districtA = pd.read_csv("/Users/dailagencarelli/Documents/Design4Transport&Logistic/instance_set/Set A/A1_1500_1.csv")
coordinates_districtA = pd.read_csv("/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A1_1500_1.csv")
pickup_coordsA = coordinates_districtA.iloc[:500]
delivery_coordsA = coordinates_districtA.iloc[1500:2000]
plt.figure(figsize=(10,6))
plt.scatter(pickup_coordsA['X'], pickup_coordsA['Y'], color='blue', label='Pickups')
plt.scatter(delivery_coordsA['X'], delivery_coordsA['Y'], color='red', label='Deliveries')
plt.title("Coordinates district A")
plt.grid(True)
plt.legend()
plt.show()

# Load data and display pickup and delivery points for District B
# coordinates_districtB = pd.read_csv("/Users/dailagencarelli/Documents/Design4Transport&Logistic/instance_set/Set B/B1_1500_1.csv")
coordinates_districtB = pd.read_csv("/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set B/B1_1500_1.csv")
pickup_coordsB = coordinates_districtB.iloc[:500]
delivery_coordsB = coordinates_districtB.iloc[1500:2000]
# Translate District B
translation_vector = np.array([300, 300])
coordinates_districtB[['X', 'Y']] = coordinates_districtB[['X', 'Y']] + translation_vector
plt.figure(figsize=(10,6))
plt.scatter(pickup_coordsB['X'], pickup_coordsB['Y'], color='blue', label='Pickups')
plt.scatter(delivery_coordsB['X'], delivery_coordsB['Y'], color='red', label='Deliveries')
plt.title("Coordinates district B")
plt.grid(True)
plt.legend()
plt.show()

# Function to calculate the Elbow curve
def calculate_elbow(data, k_range):
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

# Function to plot the Elbow curve
def plot_elbow(inertias, k_range, title):
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, '-o')
    plt.title(title)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances')
    plt.grid(True)
    plt.show()

# Function to plot clusters with satellites and hubs
def plot_clusters_with_hubs_satellites(data, centroids, satellite_locations, hub_locations, labels, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter([loc[0] for loc in satellite_locations], [loc[1] for loc in satellite_locations], c='red', marker='x', s=100, label='Potential Satellites')
    plt.scatter([loc[0] for loc in hub_locations], [loc[1] for loc in hub_locations], c='blue', marker='o', s=100, label='Potential Hubs')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()

# Generate potential hubs for District A and B
num_hubs_per_cluster = 50
def generate_random_hub_locations(centroids, num_hubs_per_cluster):
    hub_locations = []
    for center in centroids:
        for _ in range(num_hubs_per_cluster):
            hub_x = random.uniform(center[0] - 25, center[0] + 25)  
            hub_y = random.uniform(center[1] - 25, center[1] + 25)  
            hub_locations.append((hub_x, hub_y))
    return hub_locations

# Generate potential satellites for District A and B
num_satellites_per_cluster = 100
def generate_random_satellite_locations(centroids, num_hubs_per_cluster):
    satellite_locations = []
    for center in centroids:
        for _ in range(num_satellites_per_cluster):
            hub_x = random.uniform(center[0] - 25, center[0] + 25)  
            hub_y = random.uniform(center[1] - 25, center[1] + 25)  
            satellite_locations.append((hub_x, hub_y))
    return satellite_locations

# Calculate the Elbow curve for District A
k_range_A = range(1, 21)
inertias_A = calculate_elbow(coordinates_districtA[['X', 'Y']], k_range_A)
plot_elbow(inertias_A, k_range_A, title='Elbow Method for optimal clusters in District A')

# Calculate the Elbow curve for District B
k_range_B = range(1, 21)
inertias_B = calculate_elbow(coordinates_districtB[['X', 'Y']], k_range_B)
plot_elbow(inertias_B, k_range_B, title='Elbow Method for optimal cluster in District B')

# Clustering for District A
k_optimal_districtA = 4
kmeans_A = KMeans(n_clusters=k_optimal_districtA)
kmeans_A.fit(coordinates_districtA[['X', 'Y']])
centroids_A = kmeans_A.cluster_centers_
labels_A = kmeans_A.labels_
satellite_locations_A = generate_random_satellite_locations(centroids_A, num_satellites_per_cluster)
hub_locations_A = generate_random_hub_locations(centroids_A, num_hubs_per_cluster)
plot_clusters_with_hubs_satellites(coordinates_districtA[['X', 'Y']], centroids_A, satellite_locations_A, hub_locations_A, labels_A, title='Clusters with Satellites and Hubs for District A')

# Clustering for District B
k_optimal_districtB = 4
kmeans_B = KMeans(n_clusters=k_optimal_districtB)
kmeans_B.fit(coordinates_districtB[['X', 'Y']])
centroids_B = kmeans_B.cluster_centers_
labels_B = kmeans_B.labels_
satellite_locations_B = generate_random_satellite_locations(centroids_B, num_satellites_per_cluster)
hub_locations_B = generate_random_hub_locations(centroids_B, num_hubs_per_cluster)
plot_clusters_with_hubs_satellites(coordinates_districtB[['X', 'Y']], centroids_B, satellite_locations_B, hub_locations_B, labels_B, title='Clusters with Satellites and Hubs for district B')

from scipy.spatial.distance import cdist

def calculate_distances(set1, set2):
    distances = cdist(set1, set2, 'euclidean')
    return distances

dist_pickupsA_satellitesA = calculate_distances(pickup_coordsA[['X', 'Y']], satellite_locations_A)
dist_satellitesA_hubA = calculate_distances(satellite_locations_A, hub_locations_A)
dist_hubA_hubB = calculate_distances(hub_locations_A, hub_locations_B)
dist_hubB_satellitesB = calculate_distances(hub_locations_B, satellite_locations_B)
dist_satellitesB_deliveriesB = calculate_distances(satellite_locations_B, delivery_coordsB[['X', 'Y']])
dist_pickupsB_satellitesB = calculate_distances(pickup_coordsB[['X', 'Y']], satellite_locations_B)
dist_satellitesB_hubB = calculate_distances(satellite_locations_B, hub_locations_B)
dist_hubA_satellitesA= calculate_distances(hub_locations_A, satellite_locations_A)
dist_hubB_hubA = dist_hubA_hubB.T  # Transpose distances between hubA and hubB to get hubB to hubA
dist_satellitesA_deliveriesA = calculate_distances(satellite_locations_A, delivery_coordsA[['X', 'Y']])
print("Distance between hubs A and B:", dist_hubA_hubB)

m = gp.Model()
num_customers_per_district = 500
capacity_truck = 100
capacity_bike = 10

# Binary variables for nodes, hubs, and satellites for District A
x_A = m.addVars(num_customers_per_district, num_satellites_per_cluster * k_optimal_districtA, vtype=GRB.BINARY, name="x_A")
w_A = m.addVars(num_satellites_per_cluster * k_optimal_districtA, num_customers_per_district, vtype=GRB.BINARY, name="w_A")
z_A = m.addVars(num_satellites_per_cluster * k_optimal_districtA, num_hubs_per_cluster * k_optimal_districtA, vtype=GRB.BINARY, name="z_A")

# Binary variables for nodes, hubs, and satellites for District B
x_B = m.addVars(num_customers_per_district, num_satellites_per_cluster * k_optimal_districtB, vtype=GRB.BINARY, name="x_B")
w_B = m.addVars(num_satellites_per_cluster * k_optimal_districtB, num_customers_per_district, vtype=GRB.BINARY, name="w_B")
z_B = m.addVars(num_satellites_per_cluster * k_optimal_districtB, num_hubs_per_cluster * k_optimal_districtB, vtype=GRB.BINARY, name="z_B")

# Activation variables for satellites
zk_A = m.addVars(num_satellites_per_cluster * k_optimal_districtA, vtype=GRB.BINARY, name="zk_A")
zk_B = m.addVars(num_satellites_per_cluster * k_optimal_districtB, vtype=GRB.BINARY, name="zk_B")
# Activation variables for hubs
h_A = m.addVars(num_hubs_per_cluster * k_optimal_districtA, vtype=GRB.BINARY, name="h_A")
h_B = m.addVars(num_hubs_per_cluster * k_optimal_districtB, vtype=GRB.BINARY, name="h_B")

hub_assignment = m.addVars(num_hubs_per_cluster * k_optimal_districtA, num_hubs_per_cluster * k_optimal_districtB, vtype=GRB.BINARY, name="hub_assignment")

# Cost and speed parameters
gk_A = 5   # Fixed cost to open a satellite in district A
gk_B = 5   # Fixed cost to open a satellite in district B
gh_A = 30   # Fixed cost to open a hub in district A
gh_B = 30   # Fixed cost to open a hub in district B
Vt = 50      # Speed of truck
Vb = 15      # Speed of bicycle
cost_km_with_truck=10
cost_km_with_bike=1

# Objective function for District A
obj_A = gp.quicksum(dist_pickupsA_satellitesA[i, j] * x_A[i, j] *cost_km_with_bike for i in range(num_customers_per_district) for j in range(num_satellites_per_cluster * k_optimal_districtA)) 
obj_A += gp.quicksum(dist_satellitesA_hubA[i, j] * z_A[i, j] *cost_km_with_truck for i in range(num_satellites_per_cluster * k_optimal_districtA) for j in range(num_hubs_per_cluster * k_optimal_districtA))
obj_A += gp.quicksum(dist_hubA_hubB[i, j] * hub_assignment[i, j]*cost_km_with_truck for i in range(num_hubs_per_cluster * k_optimal_districtA) for j in range(num_hubs_per_cluster * k_optimal_districtB))
obj_A += gp.quicksum(dist_satellitesA_deliveriesA[i, j] * w_A[i, j]*cost_km_with_bike for i in range(num_satellites_per_cluster * k_optimal_districtA) for j in range(num_customers_per_district))
obj_A +=gp.quicksum(gk_A * zk_A[k] for k in range(num_satellites_per_cluster * k_optimal_districtA))
obj_A +=gp.quicksum(gh_A * h_A[k] for k in range(num_hubs_per_cluster * k_optimal_districtA))

# Objective function for District B with translation
obj_B = gp.quicksum(dist_pickupsB_satellitesB[i, j] * x_B[i, j]*cost_km_with_bike for i in range(num_customers_per_district) for j in range(num_satellites_per_cluster * k_optimal_districtB))  
obj_B += gp.quicksum(dist_satellitesB_hubB[i, j] * z_B[i, j]*cost_km_with_truck for i in range(num_satellites_per_cluster * k_optimal_districtB) for j in range(num_hubs_per_cluster * k_optimal_districtB))
obj_B += gp.quicksum(dist_hubB_hubA[i, j] * hub_assignment[i, j]*cost_km_with_truck for i in range(num_hubs_per_cluster * k_optimal_districtB) for j in range(num_hubs_per_cluster * k_optimal_districtA))
obj_B += gp.quicksum(dist_satellitesB_deliveriesB[i, j] * w_A[i, j]*cost_km_with_bike for i in range(num_satellites_per_cluster * k_optimal_districtB) for j in range(num_customers_per_district))
obj_B += gp.quicksum(gk_B * zk_B[k] for k in range(num_satellites_per_cluster * k_optimal_districtB))
obj_B +=gp.quicksum(gh_A * h_B[k] for k in range(num_hubs_per_cluster * k_optimal_districtA))

m.setObjective(obj_A+obj_B, GRB.MINIMIZE)

for j in range(num_hubs_per_cluster * k_optimal_districtA):
    m.addConstr(gp.quicksum(x_A[i, j] for i in range(num_customers_per_district)) <= capacity_truck, name=f"truck_capacity_A_{j}")
for j in range(num_hubs_per_cluster * k_optimal_districtB):
    m.addConstr(gp.quicksum(x_B[i, j] for i in range(num_customers_per_district)) <= capacity_truck, name=f"truck_capacity_B_{j}")

for j in range(num_satellites_per_cluster * k_optimal_districtA):
    m.addConstr(gp.quicksum(w_A[j, i] for i in range(num_customers_per_district)) <= capacity_bike, name=f"bike_capacity_A_{j}")
for j in range(num_satellites_per_cluster * k_optimal_districtB):
    m.addConstr(gp.quicksum(w_B[j, i] for i in range(num_customers_per_district)) <= capacity_bike, name=f"bike_capacity_B_{j}")

for i in range(num_customers_per_district):
    m.addConstr(gp.quicksum(x_A[i, j] for j in range(num_satellites_per_cluster * k_optimal_districtA)) == 1, name=f"customer_assignment_A_{i}")
    m.addConstr(gp.quicksum(x_B[i, j] for j in range(num_satellites_per_cluster * k_optimal_districtB)) == 1, name=f"customer_assignment_B_{i}")

for i in range(num_satellites_per_cluster * k_optimal_districtA):
    m.addConstr(gp.quicksum(z_A[i, j] for j in range(num_hubs_per_cluster * k_optimal_districtA)) == 1, name=f"satellite_hub_link_A_{i}")
for i in range(num_satellites_per_cluster * k_optimal_districtB):
    m.addConstr(gp.quicksum(z_B[i, j] for j in range(num_hubs_per_cluster * k_optimal_districtB)) == 1, name=f"satellite_hub_link_B_{i}")

# Satellites can be used only if active:
for i in range(num_satellites_per_cluster * k_optimal_districtA):
    for j in range(num_customers_per_district):
        m.addConstr(x_A[j, i] <= zk_A[i], name=f"open_satellite_A_{i}_{j}")
for i in range(num_satellites_per_cluster * k_optimal_districtB):
    for j in range(num_customers_per_district):
        m.addConstr(x_B[j, i] <= zk_B[i], name=f"open_satellite_B_{i}_{j}")

# Hubs can be used only if active:
for i in range(num_hubs_per_cluster * k_optimal_districtA):
    for j in range(num_satellites_per_cluster * k_optimal_districtA):
        m.addConstr(z_A[j, i] <= h_A[i], name=f"open_hub_A_{i}_{j}")
for i in range(num_hubs_per_cluster * k_optimal_districtB):
    for j in range(num_satellites_per_cluster * k_optimal_districtB):
        m.addConstr(z_B[j, i] <= h_B[i], name=f"open_hub_B_{i}_{j}")

for i in range(num_hubs_per_cluster * k_optimal_districtA):
    for j in range(num_hubs_per_cluster * k_optimal_districtB):
        m.addConstr(hub_assignment[i, j] <= h_A[i], name=f"hub_assignment_A_{i}_{j}")
        m.addConstr(hub_assignment[i, j] <= h_B[j], name=f"hub_assignment_B_{i}_{j}")

# One hub per district
m.addConstr(gp.quicksum(h_A[i] for i in range(num_hubs_per_cluster * k_optimal_districtA)) == 1, name="one_hub_A")
m.addConstr(gp.quicksum(h_B[i] for i in range(num_hubs_per_cluster * k_optimal_districtB)) == 1, name="one_hub_B")

# One satellite per cluster
for cluster in range(k_optimal_districtA):
    m.addConstr(gp.quicksum(zk_A[cluster * num_satellites_per_cluster + i] for i in range(num_satellites_per_cluster)) == 1, name=f"single_satellite_A_cluster_{cluster}")

for cluster in range(k_optimal_districtB):
    m.addConstr(gp.quicksum(zk_B[cluster * num_satellites_per_cluster + i] for i in range(num_satellites_per_cluster)) == 1, name=f"single_satellite_B_cluster_{cluster}")

# Optimize the model
m.optimize()

# Output the solutions
optimal_satellites_A = []
optimal_satellites_B = []
optimal_hubs_A = []
optimal_hubs_B = []

# Activated satellites for District A
for k in range(len(satellite_locations_A)):
    if zk_A[k].X > 0.5:  
        optimal_satellites_A.append(satellite_locations_A[k])

# Activated satellites for District B
for k in range(len(satellite_locations_B)):
    if zk_B[k].X > 0.5:  
        optimal_satellites_B.append(satellite_locations_B[k])

# Activated hubs for District A
for k in range(len(hub_locations_A)):
    if h_A[k].X > 0.5:  
        optimal_hubs_A.append(hub_locations_A[k])

# Activated hubs for District B
for k in range(len(hub_locations_B)):
    if h_B[k].X > 0.5:  
        optimal_hubs_B.append(hub_locations_B[k])

# Print the values of the variables zk_A and zk_B
print("zk_A:")
for k in range(num_satellites_per_cluster * k_optimal_districtA):
    print(f"Satellite {k}: {zk_A[k].X}")

print("zk_B:")
for k in range(num_satellites_per_cluster * k_optimal_districtB):
    print(f"Satellite {k}: {zk_B[k].X}")

# Print the values of the variables h_A and h_B
print("h_A:")
for k in range(num_hubs_per_cluster * k_optimal_districtA):
    print(f"Hub {k}: {h_A[k].X}")

print("h_B:")
for k in range(num_hubs_per_cluster * k_optimal_districtB):
    print(f"Hub {k}: {h_B[k].X}")

print("Satellites for district A:", optimal_satellites_A)
print("Satellites for district B:", optimal_satellites_B)
print("Hub for district A:", optimal_hubs_A)
print("Hub for district B:", optimal_hubs_B)

import matplotlib.pyplot as plt

def plot_final_map(coordinates, centroids, satellite_locations, hub_locations, labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(coordinates.iloc[:, 0], coordinates.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter([loc[0] for loc in satellite_locations], [loc[1] for loc in satellite_locations], c='red', marker='x', s=100, label='Optimal Satellites')
    plt.scatter([loc[0] for loc in hub_locations], [loc[1] for loc in hub_locations], c='blue', marker='o', s=100, label='Optimal Hubs')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='s', s=200, label='Cluster Centroids')
    plt.title('Final Map with Optimal Satellite and Hub Locations')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot the final map for District A
plot_final_map(coordinates_districtA[['X', 'Y']], centroids_A, optimal_satellites_A, optimal_hubs_A, labels_A)

# Plot the final map for District B
plot_final_map(coordinates_districtB[['X', 'Y']], centroids_B, optimal_satellites_B, optimal_hubs_B, labels_B)

#/usr/local/bin/python3 /Users/dailagencarelli/Desktop/Design-for-transport-and-logistics/hubs+satellites.py
