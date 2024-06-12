import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
import gurobipy as gp
from gurobipy import GRB

# Funzione per calcolare l'Elbow curve
def calculate_elbow(data, k_range,random_state=42):
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k,random_state=random_state)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

# Funzione per visualizzare l'Elbow curve
def plot_elbow(inertias, k_range,title):
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, '-o')
    plt.title(title)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances')
    plt.grid(True)
    plt.show()


def plot_clusters_with_hubs_satellites(data, centroids, satellite_locations, hub_locations, labels, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(satellite_locations[:, 0], satellite_locations[:, 1], c='red', marker='x', s=100, label='Potential Satellites')
    plt.scatter(hub_locations[:, 0], hub_locations[:, 1], c='blue', marker='o', s=100, label='Potential Hubs')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()


def generate_random_locations(centroids, delta=25, num_points=50):
    hub_locations = []
    for center in centroids:
        min_p = center-delta
        max_p = center+delta
        hub_locations.append(np.random.uniform(min_p, max_p, size=(num_points, 2)))
    return np.vstack(hub_locations)



def find_elbow_point(indices, inertias):
    # Normalize the indices and inertias
    indices_normalized = (indices - np.min(indices)) / (np.max(indices) - np.min(indices))
    inertias_normalized = (inertias - np.min(inertias)) / (np.max(inertias) - np.min(inertias))
    
    # Create points from the normalized values
    points = np.array(list(zip(indices_normalized, inertias_normalized)))

    # Calculate the line connecting the first and last points
    line_vector = points[-1] - points[0]
    line_vector = line_vector / np.linalg.norm(line_vector)
    
    # Calculate the distances of all points to the line
    projections = np.dot(points - points[0], line_vector)
    projected_points = points[0] + np.outer(projections, line_vector)
    distances = np.linalg.norm(points - projected_points, axis=1)
    
    # Find the index of the maximum distance
    elbow_index = np.argmax(distances)
    
    return indices[elbow_index], inertias[elbow_index]

def generate_initial_solution(points, num_satellites_per_cluster=50, num_hubs_per_cluster=100):
    hubs = []
    satellites = []
    clusters = []
    for points_district in points:
        # Calcolo dell'Elbow curve per il distretto A
        k_range = range(1, 8)
        #inertias = calculate_elbow(points[['X', 'Y']], k_range)
        inertias = calculate_elbow(points_district, k_range)
        elbow_k, elbow_inertia = find_elbow_point(k_range, inertias)
        print(f"Elbow Point: k = {elbow_k}, Inertia = {elbow_inertia}")
        plot_elbow(inertias, k_range, title='Elbow Method for optimal clusters in District A')

        # Clustering per il distretto A
        # k_optimal_district = get_optimal_elbow(inerties, k_range)
        k_optimal_district = find_elbow_point(k_range, inertias)[0]
        kmeans = KMeans(n_clusters=k_optimal_district)
        
        #kmeans.fit(points_district[['X', 'Y']])
        kmeans.fit(points_district)
        clusters.append(kmeans.labels_)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        satellite_locations = generate_random_locations(centroids, delta=25, num_points=num_satellites_per_cluster)
        hub_locations = generate_random_locations(centroids, delta=25, num_points=num_hubs_per_cluster)
        hubs.append(hub_locations)
        satellites.append(satellite_locations)
        plot_clusters_with_hubs_satellites(points_district, centroids, satellite_locations, hub_locations, labels, title='Clusters with Satellites and Hubs')

        #plot_clusters_with_hubs_satellites(points_district[['X', 'Y']], centroids, satellite_locations, hub_locations, labels, title='Clusters with Satellites and Hubs for District A')
    return hubs, satellites, clusters



def optimize_hubs_satellites(points, fixed_cost_hub=5000, fixed_cost_satellite=5000):
    model = gp.Model("hub_satellite_optimization")

    hubs_final = []
    satellites_final = []
    clusters = []

    hubs, satellites, clusters = generate_initial_solution(points, num_satellites_per_cluster=50, num_hubs_per_cluster=100)

    for points_district, hubs_district, satellites_district in zip(points, hubs, satellites):
        # Numero di punti
        n_points = points_district.shape[0]
        
        # Matrici di distanza
        dist_points_to_satellites = cdist(points_district, satellites_district)
        dist_satellites_to_hubs = cdist(satellites_district, hubs_district)
        bike_cost=1
        truck_cost=2

        # Variabili di decisione
        x = model.addVars(n_points, len(satellites_district), vtype=GRB.BINARY, name="assign_points_to_satellites")
        y = model.addVars(len(satellites_district), len(hubs_district), vtype=GRB.BINARY, name="assign_satellites_to_hubs")
        z = model.addVars(len(hubs_district), vtype=GRB.BINARY, name="open_hubs")
        w = model.addVars(len(satellites_district), vtype=GRB.BINARY, name="open_satellites")

        # Funzione obiettivo: Minimizzare il costo totale
        model.setObjective(
            gp.quicksum(dist_points_to_satellites[i, j] * x[i, j]*bike_cost for i in range(n_points) for j in range(len(satellites_district)))
            + gp.quicksum(dist_satellites_to_hubs[j, k] * y[j, k]*truck_cost for j in range(len(satellites_district)) for k in range(len(hubs_district)))
            + fixed_cost_satellite * gp.quicksum(w[j] for j in range(len(satellites_district)))
            + fixed_cost_hub * gp.quicksum(z[k] for k in range(len(hubs_district))),
            GRB.MINIMIZE
        )

        # Vincoli
        # Ogni punto deve essere assegnato a esattamente un satellite
        model.addConstrs((gp.quicksum(x[i, j] for j in range(len(satellites_district))) == 1 for i in range(n_points)), name="assign_points")

        # Ogni satellite deve essere assegnato a esattamente un hub
        model.addConstrs((gp.quicksum(y[j, k] for k in range(len(hubs_district))) == 1 for j in range(len(satellites_district))), name="assign_satellites")

        # Aprire un satellite se qualsiasi punto è assegnato ad esso
        model.addConstrs((x[i, j] <= w[j] for i in range(n_points) for j in range(len(satellites_district))), name="open_satellite_if_assigned")

        # Aprire un hub se qualsiasi satellite è assegnato ad esso
        model.addConstrs((y[j, k] <= z[k] for j in range(len(satellites_district)) for k in range(len(hubs_district))), name="open_hub_if_assigned")

        # Risolvi il modello
        model.optimize()

        # Estrazione dei risultati
        hubs_district = np.array([hubs_district[k] for k in range(len(hubs_district)) if z[k].X > 0.5])
        satellites_district = np.array([satellites_district[j] for j in range(len(satellites_district)) if w[j].X > 0.5])

        hubs_final.append(hubs_district)
        satellites_final.append(satellites_district)
        #print('hubs',hubs)
        #print(satellites)

    return hubs_final, satellites_final, clusters

# Funzione di visualizzazione
def plot_optimized_solution(points, hubs, satellites, clusters):
    for points_district, hubs_district, satellites_district, labels in zip(points, hubs, satellites, clusters):
        plt.figure(figsize=(10, 6))
        plt.scatter(points_district[:, 0], points_district[:, 1], c=labels, cmap='viridis', label='Points', alpha=0.5)
        plt.scatter(satellites_district[:, 0], satellites_district[:, 1], c='red', marker='x', s=100, label='Satellites')
        plt.scatter(hubs_district[:, 0], hubs_district[:, 1], c='blue', marker='o', s=100, label='Hubs')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.show()

