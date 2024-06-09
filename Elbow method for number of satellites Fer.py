import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from gurobipy import Model, GRB, quicksum

# Read Excel file into DataFrame
df = pd.read_excel('/Users/fernandomarti/Desktop/1CM130 Design for transport and logistics/TASK 1/Code/A1_1500_1.xlsx')

# Elbow Method
coordinates = df[['X', 'Y']]
sum_of_squared_distances = []
K = range(1, 20)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(coordinates)
    sum_of_squared_distances.append(kmeans.inertia_)

# Plot the Elbow
plt.figure(figsize=(10, 6))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# Assuming the optimal number of clusters (k) is determined to be 5
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
df['Cluster'] = kmeans.fit_predict(coordinates)
df['Cluster_Center_X'] = df['Cluster'].apply(lambda x: kmeans.cluster_centers_[x][0])
df['Cluster_Center_Y'] = df['Cluster'].apply(lambda x: kmeans.cluster_centers_[x][1])

# Plot orders and clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['X'], df['Y'], c=df['Cluster'], cmap='viridis', label='Orders')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='*', s=100, label='Satellites')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Orders and Satellites')
plt.legend()
plt.grid(True)
plt.show()

# Check cluster sizes and reassign if necessary
cluster_counts = Counter(df['Cluster'])
for cluster, count in cluster_counts.items():
    if count > 50:
        over_capacity_orders = df[df['Cluster'] == cluster]
        for idx, row in over_capacity_orders.iterrows():
            distances = np.linalg.norm(kmeans.cluster_centers_ - np.array([row['X'], row['Y']]), axis=1)
            sorted_indices = np.argsort(distances)
            for nearest_cluster in sorted_indices:
                if cluster_counts[nearest_cluster] < 50:
                    df.at[idx, 'Cluster'] = nearest_cluster
                    cluster_counts[cluster] -= 1
                    cluster_counts[nearest_cluster] += 1
                    break

# Calculate costs
df['Distance'] = np.sqrt((df['X'] - df['Cluster_Center_X'])**2 + (df['Y'] - df['Cluster_Center_Y'])**2)
df['Transport_Cost'] = df['Distance'] * 1  # 1€/distance unit
total_transport_cost = df['Transport_Cost'].sum()
num_satellites = df['Cluster'].nunique()
fixed_cost = num_satellites * 5000  # 5000€ per satellite
total_cost = total_transport_cost + fixed_cost

print(f"Number of satellites: {num_satellites}")
print(f"Total transport cost: {total_transport_cost:.2f}€")
print(f"Fixed cost: {fixed_cost:.2f}€")
print(f"Total cost: {total_cost:.2f}€")

