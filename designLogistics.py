import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Load the dataset
df = pd.read_csv('/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A1_1500_1.csv')

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

 # We create a variable to save the city hub location
city_hub_location = None

# Function to plot clusters for all locations including the city hub
def plot_clusters(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    data['Cluster'] = kmeans.fit_predict(data[['X', 'Y']])
    centroids = kmeans.cluster_centers_

    # Calculate the weighted gravity center of the centroids
    cluster_sizes = data['Cluster'].value_counts().sort_index().values
    weighted_centroids = centroids * cluster_sizes[:, np.newaxis]
    gravity_center = weighted_centroids.sum(axis=0) / cluster_sizes.sum()

    # Save the city hub location
    city_hub_location = gravity_center

    plt.figure(figsize=(10, 6))
    plt.scatter(data['X'], data['Y'], c=data['Cluster'], cmap='viridis', marker='o', label='Locations')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', label='Satellites', marker='X')
    plt.scatter(gravity_center[0], gravity_center[1], s=200, c='red', label='City Hub', marker='*')
    plt.title(f'Unified Pickup and Delivery Locations with {n_clusters} Clusters')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

    return data, centroids

# Function to generate random coordinates within each cluster
def generate_coordinates_per_cluster(data, centroids, n_points_per_cluster=10):
    generated_points = {}

    for cluster_id, centroid in enumerate(centroids):
        # Extract all data points in the current cluster
        cluster_data = data[data['Cluster'] == cluster_id]
        if cluster_data.empty:
            continue

        # Calculate the bounding box of the cluster
        min_x, max_x = cluster_data['X'].min(), cluster_data['X'].max()
        min_y, max_y = cluster_data['Y'].min(), cluster_data['Y'].max()

        # Generate random points within the bounding box
        generated_points[cluster_id] = np.column_stack([
            np.random.uniform(min_x, max_x, n_points_per_cluster),
            np.random.uniform(min_y, max_y, n_points_per_cluster)
        ])

    return generated_points

# Calculating the Elbow curve
k_range = range(1, 21)
inertias = calculate_elbow(df[['X', 'Y']], k_range)

# Plotting the Elbow curve
plot_elbow(inertias, k_range)

# Input for the optimal number of clusters based on the Elbow plot
optimal_k = int(input("Enter the optimal number of clusters based on the elbow plot: "))

# Plotting clusters using an optimal k based on user input
clustered_data, centroids = plot_clusters(df, n_clusters=optimal_k)

# Generate 10 new coordinates per cluster
generated_coordinates = generate_coordinates_per_cluster(clustered_data, centroids, n_points_per_cluster=10)

# Display generated coordinates
for cluster, coordinates in generated_coordinates.items():
    print(f"Cluster {cluster} generated coordinates:\n", coordinates)

def optimize_location(df, generated_coordinates, centroids, city_hub_location, f, g):
    # Create a new model
    model = gp.Model()

    # Decision variables
    x = {}  # Binary variable indicating whether pickup/delivery location i is assigned to satellite k
    y = {}  # Binary variable indicating whether satellite k is active (connected to the city hub) and serving pickup/delivery location j
    z = {}  # Binary variable indicating whether pickup/delivery location j is directly connected to satellite l
    w = {}  # Binary variable indicating whether satellite l is active and connected to pickup/delivery location k'

    # Define variables and constraints
    for i in range(len(df)):
        for k in range(len(centroids)):
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{k}')

    for k in range(len(centroids)):
        for j in range(len(df)):
            y[k, j] = model.addVar(vtype=GRB.BINARY, name=f'y_{k}_{j}')

    for j in range(len(df)):
        for l in range(len(centroids)):
            z[j, l] = model.addVar(vtype=GRB.BINARY, name=f'z_{j}_{l}')

    for l in range(len(centroids)):
        for k_prime in range(len(centroids)):
            w[l, k_prime] = model.addVar(vtype=GRB.BINARY, name=f'w_{l}_{k_prime}')

    # Capacity constraints
    for k in range(len(centroids)):
        model.addConstr(gp.quicksum(x[i, k] * df.iloc[i]['Demand'] for i in range(len(df))) <= 25, f"Satellite_capacity_{k}")

    model.addConstr(gp.quicksum(y[k, j] * df.iloc[j]['Demand'] for k in range(len(centroids)) for j in range(len(df))) <= 100, "City_hub_capacity")

    # Connectivity constraints
    for i in range(len(df)):
        model.addConstr(gp.quicksum(x[i, k] for k in range(len(centroids))) == 1, f"Location_assignment_{i}")

    for k in range(len(centroids)):
        model.addConstr(gp.quicksum(y[k, j] for j in range(len(df))) <= 1, f"Satellite_activation_{k}")

    for j in range(len(df)):
        for l in range(len(centroids)):
            model.addConstr(y[l, j] <= z[j, l], f"Satellite_connection_{j}_{l}")

    for l in range(len(centroids)):
        for k_prime in range(len(centroids)):
            model.addConstr(w[l, k_prime] <= y[l, :].sum(), f"Satellite_active_connection_{l}_{k_prime}")

    # Objective function
    obj = (gp.quicksum(df.iloc[i]['Distance_to_satellite'][k] * x[i, k] * f for i in range(len(df)) for k in range(len(centroids))) +
           gp.quicksum(df.iloc[j]['Distance_to_city_hub'][l] * y[l, j] * f for l in range(len(centroids)) for j in range(len(df))) +
           gp.quicksum(df.iloc[j]['Distance_to_city_hub'][l] * z[j, l] * f for l in range(len(centroids)) for j in range(len(df))) +
           gp.quicksum(df.iloc[k_prime]['Distance_to_city_hub'][l] * w[l, k_prime] * f for l in range(len(centroids)) for k_prime in range(len(centroids))) +
           gp.quicksum(df.iloc[k_prime]['Distance_to_city_hub'][l] * w[l, k_prime] * g for l in range(len(centroids)) for k_prime in range(len(centroids))) +
           gp.quicksum(df.iloc[j]['Demand'] * y[l, j] * g for l in range(len(centroids)) for j in range(len(df))) +
           gp.quicksum(df.iloc[j]['Demand'] * z[j, l] * g for l in range(len(centroids)) for j in range(len(df))))

    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize model
    model.optimize()

    # Print optimal solution
    if model.status == GRB.OPTIMAL:
        print('\nOptimal solution found:')
        for v in model.getVars():
            if v.x > 0:
                print(f'{v.varName}: {v.x}')
        print('Objective value:', model.objVal)
    else:
        print('No solution found')


optimize_location(df, generated_coordinates, centroids, city_hub_location, f=1, g=1)
