import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


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
hub_A = pickup_coords_A[['X', 'Y']].mean()
def plot_clustered_coordinates_with_hub_and_satellites(coords,hub, satellites, district_name):
    plt.figure(figsize=(10, 6))
    for cluster in range(4):
        cluster_coords = coords[coords['Cluster'] == cluster]
        plt.scatter(cluster_coords['X'], cluster_coords['Y'], label=f'Cluster {cluster+1}')
    plt.scatter(hub['X'], hub['Y'], color='blue', marker='x',s=100, label='Hub')
    plt.scatter(satellites['X'], satellites['Y'], color='black', marker='*', s=100, label='Satellite')
    plt.title(f"Clustered Coordinates District {district_name} with Hub and Satellites")
    plt.grid(True)
    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

plot_clustered_coordinates_with_hub_and_satellites(day_1,hub_A, satellites_A, 'A')
print(satellites_A)
print(hub_A)
cost_per_km=3

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = [
        # fmt: off
        [0, 33.69761828, 34.09501094, 38.80391165, 35.09719152],
        [33.69761828, 0, 66.85048308, 44.39889374, 37.4327638],
        [34.09501094, 66.85048308, 0, 64.39545587, 51.01026351],
        [38.80391165, 44.39889374, 64.39545587, 0, 51.01026351],
        [35.09719152, 37.4327638, 51.01026351, 69.68356117, 0],
        # fmt: on
    ]
    data["demands"] = [0, 57, 69, 49, 55]
    data["vehicle_capacities"] = [150,150]
    data["num_vehicles"] = 2
    data["depot"] = 0
    data["fixed_vehicle_cost"] = 1000  
    data["fixed_hub_cost"] = 5000  
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    total_distance = 0
    total_load = 0
    total_cost = data["fixed_hub_cost"]  # Include costo fisso dell'hub
    print(f"Objective (excluding fixed costs): {solution.ObjectiveValue()}")
    cost_per_km = 10  # Definisci il costo per km percorso
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                from_node = manager.IndexToNode(previous_index)
                to_node = manager.IndexToNode(index)
                route_distance += data["distance_matrix"][from_node][to_node]
        
        # Aggiungi la distanza di ritorno all'hub
        if routing.IsEnd(index):
            last_node_index = manager.IndexToNode(previous_index)
            route_distance += data["distance_matrix"][last_node_index][data["depot"]]

        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route (including return to hub): {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        route_cost = route_distance * cost_per_km + data['fixed_vehicle_cost']
        plan_output += f"Cost of the route: {route_cost}\n"
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
        total_cost += route_cost
    print(f"Total distance of all routes: {total_distance}m")
    print(f"Total load of all routes: {total_load}")
    print(f"Total cost of all routes (including fixed costs): {total_cost}")
    print(f"Objective (including fixed costs): {total_cost}")

def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()
    cost_per_km = 10  # Definisci il costo per km percorso

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node] * cost_per_km

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Add penalties for nodes not visited.
    penalty = 1000
    for node in range(1, len(data["distance_matrix"])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(10)  # Increased time limit for better solutions

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
    else:
        print("No solution found!")

if __name__ == "__main__":
    main()
