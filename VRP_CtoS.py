import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from gurobipy import Model, GRB, quicksum

np.random.seed(13)

# Read data
coordinates_district = pd.read_csv("/Users/dailagencarelli/Documents/Design4Transport&Logistic/instance_set/Set A/A1_1500_1.csv")
pickup_coords = coordinates_district.iloc[:1500].copy()  
delivery_coords = coordinates_district.iloc[1500:].copy()  

pickup_coords_A = pickup_coords[pickup_coords['Origin'] == 1].copy()
pickup_coords_B = pickup_coords[pickup_coords['Origin'] == 2].copy()
delivery_coords_A = delivery_coords[delivery_coords['Origin'] == 1].copy()
delivery_coords_B = delivery_coords[delivery_coords['Origin'] == 2].copy()

# SOLVE PROBLEM FOR DISTRICT A
pickup_coords_A = pickup_coords_A.dropna(subset=['X', 'Y', 'Early', 'Latest', 'Demand'])
day_1 = pickup_coords_A[pickup_coords_A['Day-Order'] == 1].copy()
kmeans_A = KMeans(n_clusters=4, random_state=13)
day_1['Cluster'] = kmeans_A.fit_predict(day_1[['X', 'Y']])
satellites_A = day_1.groupby('Cluster')[['X', 'Y']].mean()
print(satellites_A)
def plot_clustered_coordinates_with_hub_and_satellites(coords, satellites, district_name):
    plt.figure(figsize=(10, 6))
    for cluster in range(4):
        cluster_coords = coords[coords['Cluster'] == cluster]
        plt.scatter(cluster_coords['X'], cluster_coords['Y'], label=f'Cluster {cluster+1}')
    plt.scatter(satellites['X'], satellites['Y'], color='black', marker='*', s=100, label='Satellite')
    plt.title(f"Clustered Coordinates District {district_name} with Hub and Satellites")
    plt.grid(True)
    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

plot_clustered_coordinates_with_hub_and_satellites(day_1, satellites_A, 'A')

cluster_0_points = day_1[day_1['Cluster'] == 0][['X', 'Y']].values
for idx, point in enumerate(cluster_0_points):
    print(f"Index {idx}: {point}")
# Distance matrix between points
distance_matrix_points = cdist(cluster_0_points, cluster_0_points, metric='euclidean')
satellite_0_coord = satellites_A.loc[0].values.reshape(1, -1)
# Distance matrix between points and the satellite
distance_matrix_to_satellite = cdist(cluster_0_points, satellite_0_coord, metric='euclidean')

# Combine the matrices
# Create a full distance matrix including satellite as the first row and column
full_distance_matrix = np.zeros((len(cluster_0_points) + 1, len(cluster_0_points) + 1))
full_distance_matrix[1:, 1:] = distance_matrix_points
full_distance_matrix[0, 1:] = distance_matrix_to_satellite.flatten()
full_distance_matrix[1:, 0] = distance_matrix_to_satellite.flatten()
print(full_distance_matrix)
demand_0 = day_1[day_1['Cluster'] == 0][['Demand']].values
demand_0 = np.insert(demand_0, 0, 0)  # add satellite
print(demand_0)
window_0 = day_1[day_1['Cluster'] == 0][['Early', 'Latest']].values
satellite_time_window = np.array([[0, np.inf]])
window_0 = np.vstack([satellite_time_window, window_0])
print(window_0)
customers = len(cluster_0_points)
vehicle_capacity = 60
n_vehicles = 1  
cost_per_km = 1  
cost_open_satellite = 1000  
cost_fixed_vehicle = 50  

# CREATE THE MODEL
model = Model("CVRPTW")
# Variables
x = model.addVars(customers + 1, customers + 1, n_vehicles, vtype=GRB.BINARY, name="x")
u = model.addVars(customers + 1, vtype=GRB.CONTINUOUS, name="u")
s = model.addVar(vtype=GRB.BINARY, name="s")

model.setObjective(quicksum(cost_per_km * full_distance_matrix[i, j] * x[i, j, k] for i in range(customers + 1) for j in range(customers + 1) for k in range(n_vehicles)) +
    cost_open_satellite * s +
    quicksum(cost_fixed_vehicle * x[0, j, k] for j in range(1, customers + 1) for k in range(n_vehicles)),
    GRB.MINIMIZE
)

# Capacity constraints
for k in range(n_vehicles):
    model.addConstr(quicksum(demand_0[j] * x[i, j, k] for i in range(customers + 1) for j in range(1, customers + 1)) <= vehicle_capacity)

# Flow constraints
for j in range(1, customers + 1):
    model.addConstr(quicksum(x[i, j, k] for i in range(customers + 1) for k in range(n_vehicles)) == 1)
    model.addConstr(quicksum(x[j, i, k] for i in range(customers + 1) for k in range(n_vehicles)) == 1)

# Vincoli di tempo
for i in range(customers + 1):
    for j in range(1, customers + 1):
        if i != j:
            for k in range(n_vehicles):
                model.addConstr(u[j] >= u[i] + full_distance_matrix[i, j] - (1 - x[i, j, k]) * 10000)
for j in range(1, customers + 1):
    model.addConstr(u[j] >= window_0[j, 0])
    model.addConstr(u[j] <= window_0[j, 1])

# Vincolo di utilizzo del satellite
model.addConstr(quicksum(x[0, j, k] for j in range(1, customers + 1) for k in range(n_vehicles)) <= s * n_vehicles)
# Constraint to not go back to the same node
for k in range(n_vehicles):
    for i in range(customers + 1):
        model.addConstr(x[i, i, k] == 0)

# Each vehicle start and go back to satellite
for k in range(n_vehicles):
    model.addConstr(quicksum(x[0, j, k] for j in range(1, customers + 1)) == 1)  # start from satellite
    model.addConstr(quicksum(x[i, 0, k] for i in range(1, customers + 1)) == 1)  # Back to satellite


model.setParam(GRB.Param.TimeLimit, 500)
model.optimize()

# Results
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    if model.status == GRB.OPTIMAL:
        print('Costo ottimale: %g' % model.objVal)
    elif model.status == GRB.TIME_LIMIT:
        print('Tempo limite raggiunto, soluzione migliore trovata: %g' % model.objVal)
    
    solution = model.getAttr('x', x)
    for k in range(n_vehicles):
        print(f"\nVeicolo {k}:")
        current_node = 0
        route = [current_node]
        while True:
            for j in range(customers + 1):
                if solution[current_node, j, k] > 0.5:
                    route.append(j)
                    current_node = j
                    break
            if current_node == 0:
                break
        route_str = " -> ".join(map(str, route))
        print(f"Percorso: {route_str}")

    def plot_solution(solution, cluster_points, satellites, n_vehicles):
        plt.figure(figsize=(12, 8))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        # paths
        for k in range(n_vehicles):
            current_node = 0
            while True:
                for j in range(len(cluster_points) + 1):
                    if solution[current_node, j, k] > 0.5:
                        if current_node == 0:
                            start = satellites[0]
                        else:
                            start = cluster_points[current_node - 1]
                        if j == 0:
                            end = satellites[0]
                        else:
                            end = cluster_points[j - 1]
                        plt.plot([start[0], end[0]], [start[1], end[1]], colors[k % len(colors)] + '-', lw=2, label=f'Veicolo {k}' if current_node == 0 else "")
                        current_node = j
                        break
                if current_node == 0:
                    break
        # pickup
        for idx, point in enumerate(cluster_points):
            plt.scatter(point[0], point[1], c='black', marker='o')
            plt.annotate(f'{idx + 1}', (point[0], point[1]), textcoords="offset points", xytext=(0, 5), ha='center')
        # satellite
        plt.scatter(satellites[0, 0], satellites[0, 1], c='red', marker='*', s=200, label='Satellite')
        plt.annotate('Satellite', (satellites[0, 0], satellites[0, 1]), textcoords="offset points", xytext=(0, 5), ha='center')
        plt.title("Optimal path")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()
    plot_solution(solution, cluster_0_points, satellite_0_coord, n_vehicles)
else:
    print("No solution found")

# Print infeasible constraints

