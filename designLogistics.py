import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np  # Import numpy for array manipulations

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

# Function to plot clusters for all locations including the city hub
def plot_clusters(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    data['Cluster'] = kmeans.fit_predict(data[['X', 'Y']])
    centroids = kmeans.cluster_centers_

    # Calculate the weighted gravity center of the centroids
    cluster_sizes = data['Cluster'].value_counts().sort_index().values
    weighted_centroids = centroids * cluster_sizes[:, np.newaxis] 
    gravity_center = weighted_centroids.sum(axis=0) / cluster_sizes.sum()

    plt.figure(figsize=(10, 6))
    plt.scatter(data['X'], data['Y'], c=data['Cluster'], cmap='viridis', marker='o', label='Locations')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', label='Satellites', marker='X')
    plt.scatter(gravity_center[0], gravity_center[1], s=200, c='red', label='City Hub', marker='*')
    plt.title(f'Unified Pickup and Delivery Locations with {n_clusters} Clusters')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

# Calculating the Elbow curve
k_range = range(1, 21)  # Checking from 1 to 20 clusters
inertias = calculate_elbow(df[['X', 'Y']], k_range)

# Plotting the Elbow curve
plot_elbow(inertias, k_range)

# Input for the optimal number of clusters based on the Elbow plot
optimal_k = int(input("Enter the optimal number of clusters based on the elbow plot: "))

# Plotting clusters using an optimal k based on user input
plot_clusters(df, n_clusters=optimal_k)
