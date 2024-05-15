import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random


#data visualization
coordinates_districtA = pd.read_csv("/Users/dailagencarelli/Documents/Design4Transport&Logistic/instance_set/Set A/A1_1500_1.csv")
pickup_coordsA= coordinates_districtA.iloc[:500]
delivery_coordsA=coordinates_districtA.iloc[1500:2000]
plt.figure(figsize=(10,6))
plt.scatter(pickup_coordsA['X'], pickup_coordsA['Y'], color='blue', label='Pickups')
plt.scatter(delivery_coordsA['X'], delivery_coordsA['Y'], color='red', label='Deliveries')
plt.title("Coordinates district A")
plt.grid(True)
plt.legend()
plt.show()
coordinates_districtB = pd.read_csv("/Users/dailagencarelli/Documents/Design4Transport&Logistic/instance_set/Set B/B1_1500_1.csv")
pickup_coordsB= coordinates_districtB.iloc[:500]
delivery_coordsB=coordinates_districtB.iloc[1500:2000]
plt.figure(figsize=(10,6))
plt.scatter(pickup_coordsB['X'], pickup_coordsB['Y'], color='blue', label='Pickups')
plt.scatter(delivery_coordsB['X'], delivery_coordsB['Y'], color='red', label='Deliveries')
plt.title("Coordinates district B")
plt.grid(True)
plt.legend()
plt.show()

#BUILDING CLUSTERS

# Function to calculate the Elbow curve data
def calculate_elbow(data, k_range):
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

# Function to plot the Elbow curve
def plot_elbow(inertias, k_range):
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, '-o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances')
    plt.grid(True)
    plt.show()

def plot_clusters(data, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red', s=300)
    plt.title('Clusters')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


def generate_random_satellite_locations(centroids,num_satellites_per_cluster):
    satellite_locations = []
    for center in centroids:
        for _ in range(num_satellites_per_cluster):
            satellite_x = random.uniform(0, 100)
            satellite_y = random.uniform(0, 100)
            satellite_locations.append((satellite_x, satellite_y))
    return satellite_locations

def plot_clusters_with_satellites(data, centroids, satellite_locations, labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter([loc[0] for loc in satellite_locations], [loc[1] for loc in satellite_locations], c='red', marker='x', s=100, label='Potentail Satellites')
    #plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='blue', s=300, label='Centroids')
    plt.title('Clusters with Satellites')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()

# Calculating the Elbow curve District A
k_range = range(1, 21)
inertias = calculate_elbow(coordinates_districtA[['X', 'Y']], k_range)
plot_elbow(inertias, k_range)

# Calculating the Elbow curve District A
k_range = range(1, 21)
inertias = calculate_elbow(coordinates_districtB[['X', 'Y']], k_range)
plot_elbow(inertias, k_range)


# Clusters for district A
k_optimal_districtA = 4 
plot_clusters(coordinates_districtA[['X', 'Y']], k_optimal_districtA)

# Clusters for district B
k_optimal_districtB = 4  
plot_clusters(coordinates_districtB[['X', 'Y']], k_optimal_districtB)

# Generazione dei satelliti per il distretto A e B
num_satellites_per_cluster = 50

# KMeans clustering per trovare i centroidi del distretto A
kmeans_A = KMeans(n_clusters=k_optimal_districtA)
kmeans_A.fit(coordinates_districtA[['X', 'Y']])
centroids_A = kmeans_A.cluster_centers_
labels_A = kmeans_A.labels_

# Generazione delle posizioni dei satelliti per il distretto A
satellite_locations_A = generate_random_satellite_locations(centroids_A, num_satellites_per_cluster)

# KMeans clustering per trovare i centroidi del distretto B
kmeans_B = KMeans(n_clusters=k_optimal_districtB)
kmeans_B.fit(coordinates_districtB[['X', 'Y']])
centroids_B = kmeans_B.cluster_centers_
labels_B = kmeans_B.labels_

# Generazione delle posizioni dei satelliti per il distretto B
satellite_locations_B = generate_random_satellite_locations(centroids_B, num_satellites_per_cluster)

# Plot delle posizioni dei satelliti per il distretto A con divisione in cluster
plot_clusters_with_satellites(coordinates_districtA[['X', 'Y']], centroids_A, satellite_locations_A, labels_A)

# Plot delle posizioni dei satelliti per il distretto B con divisione in cluster
plot_clusters_with_satellites(coordinates_districtB[['X', 'Y']], centroids_B, satellite_locations_B, labels_B)

from scipy.spatial.distance import cdist

# Calcolo delle distanze tra clienti e satelliti
dist_pickup_to_satellite_A = cdist(pickup_coordsA[['X', 'Y']], satellite_locations_A)
dist_pickup_to_satellite_B = cdist(pickup_coordsB[['X', 'Y']], satellite_locations_B)
dist_delivery_to_satellite_A = cdist(delivery_coordsA[['X', 'Y']], satellite_locations_A)
dist_delivery_to_satellite_B = cdist(delivery_coordsB[['X', 'Y']], satellite_locations_B)
# print("dist_pickup_to_satellite_A",dist_pickup_to_satellite_A)
# print("dist_pickup_to_satellite_B",dist_pickup_to_satellite_B)
# print("dist_delivery_to_satellite_A",dist_delivery_to_satellite_A)
# print("dist_delivery_to_satellite_B",dist_delivery_to_satellite_B)
# Definizione delle variabili Gurobi
m = gp.Model()
num_customers_per_district = 500
capacity_truck = 10
# Variabili binarie
x_A = m.addVars(num_customers_per_district, num_satellites_per_cluster * k_optimal_districtA, vtype=GRB.BINARY, name="x_A")
w_A = m.addVars(num_satellites_per_cluster * k_optimal_districtA, num_customers_per_district, vtype=GRB.BINARY, name="w_A")

x_B = m.addVars(num_customers_per_district, num_satellites_per_cluster * k_optimal_districtB, vtype=GRB.BINARY, name="x_B")
w_B = m.addVars(num_satellites_per_cluster * k_optimal_districtB, num_customers_per_district, vtype=GRB.BINARY, name="w_B")

# Variabili di attivazione dei satelliti
zk_A = m.addVars(num_satellites_per_cluster * k_optimal_districtA, vtype=GRB.BINARY, name="zk_A")
zk_B = m.addVars(num_satellites_per_cluster * k_optimal_districtB, vtype=GRB.BINARY, name="zk_B")

# Parametri di costo e velocità
gk_A = 5   # Fixed cost to open a satellite in district A
gk_B = 5   # Fixed cost to open a satellite in district B
Vt = 50      # Speed of truck
Vb = 15      # Speed of bicycle

# Funzione obiettivo per il distretto A
obj_A = gp.quicksum(dist_pickup_to_satellite_A[i, j] / Vt * x_A[i, j] for i in range(num_customers_per_district) for j in range(num_satellites_per_cluster * k_optimal_districtA)) + gp.quicksum(gk_A * zk_A[k] for k in range(num_satellites_per_cluster * k_optimal_districtA))

# Funzione obiettivo per il distretto B
obj_B = gp.quicksum(dist_delivery_to_satellite_B[i, j] / Vt * x_B[i, j] for i in range(num_customers_per_district) for j in range(num_satellites_per_cluster * k_optimal_districtB)) + gp.quicksum(gk_B * zk_B[k] for k in range(num_satellites_per_cluster * k_optimal_districtB))

# Aggiungiamo la funzione obiettivo al modello
m.setObjective(obj_A + obj_B, GRB.MINIMIZE)

# Vincoli per il distretto A
for i in range(num_customers_per_district):
    m.addConstr(gp.quicksum(x_A[i, j] for j in range(num_satellites_per_cluster * k_optimal_districtA)) == 1)

for k in range(num_satellites_per_cluster * k_optimal_districtA):
    m.addConstr(gp.quicksum(w_A[k, h] for h in range(num_customers_per_district)) <= zk_A[k] * capacity_truck)

# Vincoli per il distretto B
for i in range(num_customers_per_district):
    m.addConstr(gp.quicksum(x_B[i, j] for j in range(num_satellites_per_cluster * k_optimal_districtB)) == 1)

for k in range(num_satellites_per_cluster * k_optimal_districtB):
    m.addConstr(gp.quicksum(w_B[k, h] for h in range(num_customers_per_district)) <= zk_B[k] * capacity_truck)

# Vincoli per garantire che ogni cluster sia servito da almeno un satellite
for cluster in range(k_optimal_districtB):
    m.addConstr(
        gp.quicksum(
            zk_A[cluster * num_satellites_per_cluster + k] for k in range(num_satellites_per_cluster)
        ) >= 1,
        f"cluster_{cluster}_A"
    )
    m.addConstr(
        gp.quicksum(
            zk_B[cluster * num_satellites_per_cluster + k] for k in range(num_satellites_per_cluster)
        ) >= 1,
        f"cluster_{cluster}_B"
    )


# Ottimizzazione del modello
m.optimize()

# Output delle soluzioni
optimal_satellites_A = []
optimal_satellites_B = []

for k in range(len(satellite_locations_A)):
    if zk_A[k].X > 0:  # Se il satellite è attivato
        optimal_satellites_A.append(satellite_locations_A[k])
        

for k in range(len(satellite_locations_B)):
    if zk_B[k].X > 0:  # Se il satellite è attivato
        optimal_satellites_B.append(satellite_locations_B[k])

# Stampa dei valori delle variabili zk_A e zk_B
print("Valori delle variabili zk_A:")
for k in range(num_satellites_per_cluster * k_optimal_districtA):
    print(f"Satellite {k}: {zk_A[k].X}")

print("Valori delle variabili zk_B:")
for k in range(num_satellites_per_cluster * k_optimal_districtB):
    print(f"Satellite {k}: {zk_B[k].X}")
print("A", optimal_satellites_A)
print("B", optimal_satellites_B)

# Plot delle posizioni ottimali dei satelliti per il distretto A
plt.figure(figsize=(10, 6))
plt.scatter(coordinates_districtA['X'], coordinates_districtA['Y'], c=labels_A, cmap='viridis', alpha=0.5)
plt.scatter([loc[0] for loc in optimal_satellites_A], [loc[1] for loc in optimal_satellites_A], c='red', marker='x', s=100, label='Optimal Satellites')
plt.scatter(centroids_A[:, 0], centroids_A[:, 1], marker='*', c='blue', s=300, label='Centroids')
plt.title('Optimal Satellite Locations in District A')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()

# Plot delle posizioni ottimali dei satelliti per il distretto B
plt.figure(figsize=(10, 6))
plt.scatter(coordinates_districtB['X'], coordinates_districtB['Y'], c=labels_B, cmap='viridis', alpha=0.5)
plt.scatter([loc[0] for loc in optimal_satellites_B], [loc[1] for loc in optimal_satellites_B], c='red', marker='x', s=100, label='Optimal Satellites')
plt.scatter(centroids_B[:, 0], centroids_B[:, 1], marker='*', c='blue', s=300, label='Centroids')
plt.title('Optimal Satellite Locations in District B')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()

#/usr/local/bin/python3 /Users/dailagencarelli/Desktop/Design-for-transport-and-logistics/Gurobi.py
#df = pd.read_csv('/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A1_1500_1.csv')
