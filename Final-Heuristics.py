import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dati forniti
satellites_A = [[28.21, 76.14], [80.02, 74.59], [25.5, 22.0], [81.13, 24.87]]
satellites_B = [[174.65, 70.27], [229.47, 75.61], [226.50, 26.45], [176.75, 20.85]]
hub_A = [[51.92, 47.90]]
hub_B = [[201.54, 49.87]]

coordinates_district = pd.read_csv("/Users/dailagencarelli/Desktop/Design-for-transport-and-logistics/A1_1500_1.csv")
coordinates_district.fillna(0, inplace=True)

pickup_coords = coordinates_district.iloc[:1500].copy()  
delivery_coords = coordinates_district.iloc[1500:].copy()  

pickup_coords_A = pickup_coords[pickup_coords['Origin'] == 1].copy()
pickup_coords_B = pickup_coords[pickup_coords['Origin'] == 2].copy()
delivery_coords_A = delivery_coords[delivery_coords['Origin'] == 1].copy()
delivery_coords_B = delivery_coords[delivery_coords['Origin'] == 2].copy()

max_point_A = max(pickup_coords_A['X'].max(), delivery_coords_A['X'].max())
min_point_A = min(pickup_coords_A['X'].min(), delivery_coords_A['X'].min())

pickup_coords_B['X'] += max_point_A + (max_point_A - min_point_A) * 0.5
delivery_coords_B['X'] += max_point_A + (max_point_A - min_point_A) * 0.5

# Neighborhoods
x_min_A = pickup_coords_A['X'].min()
x_max_A = pickup_coords_A['X'].max()
y_min_A = pickup_coords_A['Y'].min()
y_max_A = pickup_coords_A['Y'].max()

x_min_B = pickup_coords_B['X'].min()
x_max_B = pickup_coords_B['X'].max()
y_min_B = pickup_coords_B['Y'].min()
y_max_B = pickup_coords_B['Y'].max()

x_step_A = (x_max_A - x_min_A) / 2
y_step_A = (y_max_A - y_min_A) / 2

x_step_B = (x_max_B - x_min_B) / 2
y_step_B = (y_max_B - y_min_B) / 2

# Select day from keyborad
while True:
    try:
        day_input = int(input("Insert the interested day (1-7): "))
        if 1 <= day_input <= 7:
            break
        else:
            print("Insert a number between 1 and 7")
    except ValueError:
        print("Insert a valid number.")


pickup_coords_day_A = pickup_coords_A[pickup_coords_A['Day-Ops'] == day_input].copy()
delivery_coords_day_A = delivery_coords_A[delivery_coords_A['Day-Ops'] == day_input].copy()
pickup_coords_day_B = pickup_coords_B[pickup_coords_B['Day-Ops'] == day_input].copy()
delivery_coords_day_B = delivery_coords_B[delivery_coords_B['Day-Ops'] == day_input].copy()

# Separate in neighborhoods
def divide_into_quadrants(coords, x_min, x_max, y_min, y_max, x_step, y_step):
    quadrants = {}
    quadrants['Q1'] = coords[(coords['X'] >= x_min) & (coords['X'] < x_min + x_step) & (coords['Y'] >= y_min) & (coords['Y'] < y_min + y_step)]
    quadrants['Q2'] = coords[(coords['X'] >= x_min + x_step) & (coords['X'] <= x_max) & (coords['Y'] >= y_min) & (coords['Y'] < y_min + y_step)]
    quadrants['Q3'] = coords[(coords['X'] >= x_min) & (coords['X'] < x_min + x_step) & (coords['Y'] >= y_min + y_step) & (coords['Y'] <= y_max)]
    quadrants['Q4'] = coords[(coords['X'] >= x_min + x_step) & (coords['X'] <= x_max) & (coords['Y'] >= y_min + y_step) & (coords['Y'] <= y_max)]
    return quadrants

quadrants_A_pickup = divide_into_quadrants(pickup_coords_day_A, x_min_A, x_max_A, y_min_A, y_max_A, x_step_A, y_step_A)
quadrants_A_delivery = divide_into_quadrants(delivery_coords_day_A, x_min_A, x_max_A, y_min_A, y_max_A, x_step_A, y_step_A)
quadrants_B_pickup = divide_into_quadrants(pickup_coords_day_B, x_min_B, x_max_B, y_min_B, y_max_B, x_step_B, y_step_B)
quadrants_B_delivery = divide_into_quadrants(delivery_coords_day_B, x_min_B, x_max_B, y_min_B, y_max_B, x_step_B, y_step_B)

# Calculate Distance
def compute_euclidean_distance_matrix(locations):
    distances = np.zeros((len(locations), len(locations)))
    for i in range(len(locations)):
        for j in range(len(locations)):
            if i != j:
                distances[i][j] = np.linalg.norm(locations[i] - locations[j])
    return distances

# Distance matrix for point in a neighborhood
def calculate_distances_in_quadrant(quadrant_pickup, quadrant_delivery, satellites):
    if len(quadrant_pickup) == 0 and len(quadrant_delivery) == 0:
        return np.array([])

    # Coordinate X Y for piclup and deliveries
    pickup_coords = quadrant_pickup[['X', 'Y']].values
    delivery_coords = quadrant_delivery[['X', 'Y']].values

    all_points = np.concatenate([pickup_coords, delivery_coords, satellites])
    num_pickup = len(pickup_coords)
    num_delivery = len(delivery_coords)
    num_satellites = len(satellites)

    distances = np.zeros((len(all_points), len(all_points)))

    # Distance between all points
    for i in range(len(all_points)):
        for j in range(len(all_points)):
            if i != j:
                point_i = all_points[i]
                point_j = all_points[j]
                distance_ij = np.linalg.norm(point_i - point_j)

                # Penality if time window is not respected
                if i < num_pickup:  # i -pickup
                    early_i = quadrant_pickup.iloc[i]['Early']
                    late_i = quadrant_pickup.iloc[i]['Latest']
                elif i < num_pickup + num_delivery:  # i - delivery
                    early_i = quadrant_delivery.iloc[i - num_pickup]['Early']
                    late_i = quadrant_delivery.iloc[i - num_pickup]['Latest']
                else:  # i - satellite
                    early_i = -np.inf  # no time window for satellites
                    late_i = np.inf

                if j < num_pickup:  # j - pickup
                    early_j = quadrant_pickup.iloc[j]['Early']
                    late_j = quadrant_pickup.iloc[j]['Latest']
                elif j < num_pickup + num_delivery:  # j - delivery
                    early_j = quadrant_delivery.iloc[j - num_pickup]['Early']
                    late_j = quadrant_delivery.iloc[j - num_pickup]['Latest']
                else:  # j - satellite
                    early_j = -np.inf  
                    late_j = np.inf

                
                if late_i < early_j or late_j < early_i:
                    distance_ij += 1000  

                distances[i][j] = distance_ij

    return distances


# Nearest neighbor
def nearest_neighbor_vrp(start_point, delivery_points, pickup_points, vehicle_capacity, num_vehicles):
    all_points = delivery_points + pickup_points
    routes = [[] for _ in range(num_vehicles)]
    total_distances = [0 for _ in range(num_vehicles)]
    current_capacities = [0 for _ in range(num_vehicles)]
    current_points = [start_point for _ in range(num_vehicles)]  # Start from the satellite position

    unvisited = set(range(len(all_points)))

    while unvisited:
        for vehicle in range(num_vehicles):
            if not unvisited:
                break

            nearest_point = None
            nearest_distance = float('inf')

            for i in unvisited:
                point = all_points[i]
                distance = np.linalg.norm(np.array(current_points[vehicle][:2]) - np.array(point[:2]))

                # Check capacity constraint
                if current_capacities[vehicle] + point[2] > vehicle_capacity:
                    continue

                if distance < nearest_distance:
                    nearest_point = point
                    nearest_distance = distance
                    nearest_index = i

            if nearest_point is not None:
                routes[vehicle].append(nearest_point)
                total_distances[vehicle] += nearest_distance
                current_points[vehicle] = nearest_point
                current_capacities[vehicle] += nearest_point[2]
                unvisited.remove(nearest_index)

                if nearest_point in pickup_points:
                    print(f"Vehicle {vehicle+1}: Picking up package at {nearest_point}")
                elif nearest_point in delivery_points:
                    print(f"Vehicle {vehicle+1}: Delivering package at {nearest_point}")

            else:
                # Return to the satellite if no valid points left to visit
                total_distances[vehicle] += np.linalg.norm(np.array(current_points[vehicle][:2]) - np.array(start_point[:2]))
                current_points[vehicle] = start_point
                current_capacities[vehicle] = 0
                routes[vehicle].append(start_point)

    # Return to the satellite for each vehicle
    for vehicle in range(num_vehicles):
        total_distances[vehicle] += np.linalg.norm(np.array(current_points[vehicle][:2]) - np.array(start_point[:2]))
        routes[vehicle].append(start_point)

    return routes, total_distances




# Convert data for VRP
def convert_to_vrp_format(coords):
    formatted_coords = []
    
    for index, row in coords.iterrows():
        x, y, origin, demand, early, late = row['X'], row['Y'], row['Type'], row['Demand'], row['Early'], row['Latest']
        if origin == 3:  # pickup service
            formatted_coords.append((x, y, demand))
        elif origin == 2:  # delivery service
            formatted_coords.append((x, y, -demand))  # demand is negative for delivery
    
    return formatted_coords




vehicle_capacity = 20
num_vehicles = 1  
cost_bike=5
# VRP for city A
routes_A = {}
distances_A = {}

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    delivery_points = convert_to_vrp_format(quadrants_A_delivery[q])
    pickup_points = convert_to_vrp_format(quadrants_A_pickup[q])
    satellite_routes = []
    satellite_distances = []

    for satellite in satellites_A:
        routes, distances = nearest_neighbor_vrp(satellite, delivery_points, pickup_points, vehicle_capacity, num_vehicles)
        # satellite at the beginning of route
        routes_with_start = [[satellite] + route for route in routes]
        satellite_routes.append(routes_with_start)
        satellite_distances.append(distances)

    best_route_index = np.argmin([sum(d) for d in satellite_distances])
    routes_A[q] = satellite_routes[best_route_index]
    distances_A[q] = satellite_distances[best_route_index]

# VRP for city B
routes_B = {}
distances_B = {}

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    delivery_points = convert_to_vrp_format(quadrants_B_delivery[q])
    pickup_points = convert_to_vrp_format(quadrants_B_pickup[q])
    satellite_routes = []
    satellite_distances = []

    for satellite in satellites_B:
        routes, distances = nearest_neighbor_vrp(satellite, delivery_points, pickup_points, vehicle_capacity, num_vehicles)
        routes_with_start = [[satellite] + route for route in routes]
        satellite_routes.append(routes_with_start)
        satellite_distances.append(distances)

    best_route_index = np.argmin([sum(d) for d in satellite_distances])
    routes_B[q] = satellite_routes[best_route_index]
    distances_B[q] = satellite_distances[best_route_index]


# Results
print("Routes and distances for quadrants in City A:")
for q in routes_A:
    print(f"Quadrant {q}:")
    for i in range(num_vehicles):
        route = routes_A[q][i]
        total_distance = distances_A[q][i]
        print(f"  Vehicle {i+1} Route: {route}")
        print(f"  Vehicle {i+1} Total distance: {total_distance}")

print("\nRoutes and distances for quadrants in City B:")
for q in routes_B:
    print(f"Quadrant {q}:")
    for i in range(num_vehicles):
        route = routes_B[q][i]
        total_distance = distances_B[q][i]
        print(f"  Vehicle {i+1} Route: {route}")
        print(f"  Vehicle {i+1} Total distance: {total_distance}")
# Cost for city A
total_cost_A = 0
print("Routes and distances for quadrants in City A:")
for q in routes_A:
    print(f"Quadrant {q}:")
    for i in range(num_vehicles):
        route = routes_A[q][i]
        total_distance = distances_A[q][i]
        cost = total_distance * cost_bike
        total_cost_A += cost
        print(f"  Vehicle {i+1} Route: {route}")
        print(f"  Vehicle {i+1} Total cost: {cost}")

# Cost for city B
total_cost_B = 0
print("\nRoutes and distances for quadrants in City B:")
for q in routes_B:
    print(f"Quadrant {q}:")
    for i in range(num_vehicles):
        route = routes_B[q][i]
        total_distance = distances_B[q][i]
        cost = total_distance * cost_bike
        total_cost_B += cost
        print(f"  Vehicle {i+1} Route: {route}")
        print(f"  Vehicle {i+1} Total cost: {cost}")

# Total final cost 2nd echelon
total_cost_second_echelon = total_cost_A + total_cost_B

print(f"\nTotal 2nd echelon cost for City A (customer-satellites): {total_cost_A}")
print(f"Total 2nd echelon cost for City B (customer-satellites): {total_cost_B}")
print(f"Final 2nd echelon (customer-satellites)total cost: {total_cost_second_echelon}")


# Plot map with routes
plt.figure(figsize=(14, 10))

plt.scatter(pickup_coords_day_A['X'], pickup_coords_day_A['Y'], c='blue', label='Pickup City A', alpha=0.6)
plt.scatter(delivery_coords_day_A['X'], delivery_coords_day_A['Y'], c='cyan', label='Delivery City A', alpha=0.6)
plt.scatter(pickup_coords_day_B['X'], pickup_coords_day_B['Y'], c='red', label='Pickup City B', alpha=0.6)
plt.scatter(delivery_coords_day_B['X'], delivery_coords_day_B['Y'], c='orange', label='Delivery City B', alpha=0.6)

sat_A = np.array(satellites_A)
sat_B = np.array(satellites_B)
hub_A = np.array(hub_A)
hub_B = np.array(hub_B)

plt.scatter(sat_A[:, 0], sat_A[:, 1], c='green', marker='^', s=100, label='Satellites A')
plt.scatter(sat_B[:, 0], sat_B[:, 1], c='green', marker='^', s=100, label='Satellites B')
plt.scatter(hub_A[:, 0], hub_A[:, 1], c='purple', marker='x', s=100, label='Hub A')
plt.scatter(hub_B[:, 0], hub_B[:, 1], c='purple', marker='x', s=100, label='Hub B')

plt.plot([x_min_A, x_max_A], [y_min_A + y_step_A, y_min_A + y_step_A], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_A, x_max_A], [y_min_A + 2*y_step_A, y_min_A + 2*y_step_A], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_A + x_step_A, x_min_A + x_step_A], [y_min_A, y_max_A], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_A + 2*x_step_A, x_min_A + 2*x_step_A], [y_min_A, y_max_A], c='gray', linestyle='--', linewidth=2)


plt.plot([x_min_B, x_max_B], [y_min_B + y_step_B, y_min_B + y_step_B], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_B, x_max_B], [y_min_B + 2*y_step_B, y_min_B + 2*y_step_B], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_B + x_step_B, x_min_B + x_step_B], [y_min_B, y_max_B], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_B + 2*x_step_B, x_min_B + 2*x_step_B], [y_min_B, y_max_B], c='gray', linestyle='--', linewidth=2)

#city A
colors = ['g--', 'b--', 'm--']  
for q in routes_A:
    for vehicle in range(num_vehicles):
        route = routes_A[q][vehicle]
        route_coords = np.array([[point[0], point[1]] for point in route])  
        plt.plot(route_coords[:, 0], route_coords[:, 1], colors[vehicle % len(colors)])

#city B
for q in routes_B:
    for vehicle in range(num_vehicles):
        route = routes_B[q][vehicle]
        # Estrai le coordinate per il plot
        route_coords = np.array([[point[0], point[1]] for point in route])  # Extract X and Y coordinates
        plt.plot(route_coords[:, 0], route_coords[:, 1], colors[vehicle % len(colors)])

plt.legend()
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title(f'Map for day {day_input}')
plt.grid(True)
plt.show()

# Function to compute the total time for a given distance and speed
def compute_total_time(distance, speed):
    return distance / speed


def calculate_total_time_second_echelon(distances, speed_bike):
    total_time = 0
    for key in distances:
        for distance in distances[key]:
            total_time += distance / speed_bike
    
    return total_time



# Parcel sum per trip
def calcola_somma_demand(percorso):
    somma_demand = 0
    for step in percorso[1:-1]:  # only movment tuple
        if isinstance(step, tuple):
            somma_demand += step[2]  # add amount of parcels on tuple
    return somma_demand

# sum of demand per each neighbor in a city
def calcola_somma_demand_citta(routes):
    somma_demand_citta = {}
    for quadrant, route_data in routes.items():
        somma_demand = 0
        for percorso in route_data:
            somma_demand += calcola_somma_demand(percorso)
        somma_demand_citta[quadrant] = somma_demand
    return somma_demand_citta

# Demand sum for each city
somma_demand_citta_A = calcola_somma_demand_citta(routes_A)
somma_demand_citta_B = calcola_somma_demand_citta(routes_B)

# Results for both cities
print("Demand City A:")
for quadrant, somma in somma_demand_citta_A.items():
    print(f"Quadrante {quadrant}: {somma}")

print("\nDemand City B:")
for quadrant, somma in somma_demand_citta_B.items():
    print(f"Area {quadrant}: {somma}")



#demand each satellite
demand_A = [somma_demand_citta_A['Q1'], somma_demand_citta_A['Q2'], somma_demand_citta_A['Q3'], somma_demand_citta_A['Q4']]
demand_B = [somma_demand_citta_B['Q1'], somma_demand_citta_B['Q2'], somma_demand_citta_B['Q3'], somma_demand_citta_B['Q4']]

satellites_A = np.array([[28.21, 76.14], [80.02, 74.59], [25.5, 22.0], [81.13, 24.87]])
satellites_B = np.array([[174.65, 70.27], [229.47, 75.61], [226.50, 26.45], [176.75, 20.85]])
hub_A = np.array([[51.92, 47.90]])
hub_B = np.array([[201.54, 49.87]])


minivan_capacity = 100
num_minivans = 1
cost_minivan = 10

# Eucldean distance
def distanza(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# VRP with nearest neighbor and more vehicles
def vrp_nearest_neighbor_multi_vehicle(hub, satellites, demand, capacity, num_vehicles):
    n = len(satellites)
    visited = [False] * n
    routes = [[] for _ in range(num_vehicles)]
    vehicle_loads = [0] * num_vehicles
    current_positions = [hub] * num_vehicles
    vehicle_index = 0

    while not all(visited):
        nearest_dist = float('inf')
        nearest_index = -1
        
        for i in range(n):
            if not visited[i] and distanza(current_positions[vehicle_index], satellites[i]) < nearest_dist:
                nearest_dist = distanza(current_positions[vehicle_index], satellites[i])
                nearest_index = i

        if nearest_index == -1:
            break  # if there are not visitet satellites, stop

        if vehicle_loads[vehicle_index] + demand[nearest_index] <= capacity:
            routes[vehicle_index].append(nearest_index)
            vehicle_loads[vehicle_index] += demand[nearest_index]
            current_positions[vehicle_index] = satellites[nearest_index]
            visited[nearest_index] = True
        else:
            vehicle_index = (vehicle_index + 1) % num_vehicles
            # If tryed each vehicle without success, reload
            if all(vehicle_loads[v] + demand[nearest_index] > capacity for v in range(num_vehicles)):
                # Go back to hub, unload and start again
                routes[vehicle_index].append(-1)  # indicator to return to hub
                vehicle_loads[vehicle_index] = 0
                current_positions[vehicle_index] = hub
    
    
    for route in routes:
        while -1 in route:
            route.remove(-1)
    
    return routes



#KPI
# Total cost of the path
def calcola_costo_totale(hub, satellites, routes, cost_minivan):
    totale_distanza = 0
    for route in routes:
        if route:
            path = [hub] + [satellites[j] for j in route if j != -1] + [hub]
            for i in range(len(path) - 1):
                totale_distanza += distanza(path[i], path[i + 1])
    return totale_distanza * cost_minivan


routes_A = vrp_nearest_neighbor_multi_vehicle(hub_A[0], satellites_A, demand_A, minivan_capacity, num_minivans)
routes_B = vrp_nearest_neighbor_multi_vehicle(hub_B[0], satellites_B, demand_B, minivan_capacity, num_minivans)

# Total firt echelon cost
total_cost_first_echelon_A = calcola_costo_totale(hub_A[0], satellites_A, routes_A, cost_minivan)
total_cost_first_echelon_B = calcola_costo_totale(hub_B[0], satellites_B, routes_B, cost_minivan)

print(f"Total cost for first echelon city A: {total_cost_first_echelon_A}")
print(f"Total cost for first echelon city B: {total_cost_first_echelon_B}")

# Plot routes
def plot_routes_combined(hub_A, satellites_A, routes_A, hub_B, satellites_B, routes_B):
    plt.figure(figsize=(12, 8))
    plt.scatter(pickup_coords_day_A['X'], pickup_coords_day_A['Y'], c='blue', label='Pickup City A', alpha=0.6)
    plt.scatter(delivery_coords_day_A['X'], delivery_coords_day_A['Y'], c='cyan', label='Delivery City A', alpha=0.6)
    plt.scatter(pickup_coords_day_B['X'], pickup_coords_day_B['Y'], c='red', label='Pickup City B', alpha=0.6)
    plt.scatter(delivery_coords_day_B['X'], delivery_coords_day_B['Y'], c='orange', label='Delivery City B', alpha=0.6)
    plt.plot([x_min_A, x_max_A], [y_min_A + y_step_A, y_min_A + y_step_A], c='gray', linestyle='--', linewidth=2)
    plt.plot([x_min_A, x_max_A], [y_min_A + 2*y_step_A, y_min_A + 2*y_step_A], c='gray', linestyle='--', linewidth=2)
    plt.plot([x_min_A + x_step_A, x_min_A + x_step_A], [y_min_A, y_max_A], c='gray', linestyle='--', linewidth=2)
    plt.plot([x_min_A + 2*x_step_A, x_min_A + 2*x_step_A], [y_min_A, y_max_A], c='gray', linestyle='--', linewidth=2)
    plt.plot([x_min_B, x_max_B], [y_min_B + y_step_B, y_min_B + y_step_B], c='gray', linestyle='--', linewidth=2)
    plt.plot([x_min_B, x_max_B], [y_min_B + 2*y_step_B, y_min_B + 2*y_step_B], c='gray', linestyle='--', linewidth=2)
    plt.plot([x_min_B + x_step_B, x_min_B + x_step_B], [y_min_B, y_max_B], c='gray', linestyle='--', linewidth=2)
    plt.plot([x_min_B + 2*x_step_B, x_min_B + 2*x_step_B], [y_min_B, y_max_B], c='gray', linestyle='--', linewidth=2)
    plt.scatter(satellites_A[:, 0], satellites_A[:, 1], color='green',marker="^",s=100 ,label='Satellites A')
    plt.scatter(hub_A[0], hub_A[1], color='purple', marker= "x",s=100, label='Hub A')
    colors_A = ['blue', 'cyan']  # Vehicle colors
    for i, route in enumerate(routes_A):
        if route:
            path = [hub_A] + [satellites_A[j] for j in route if j != -1] + [hub_A]
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], color=colors_A[i % len(colors_A)])
    

    colors_B = ['red', 'orange']  # vehicle colors
    for i, route in enumerate(routes_B):
        if route:
            path = [hub_B] + [satellites_B[j] for j in route if j != -1] + [hub_B]
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], color=colors_B[i % len(colors_B)])
    plt.scatter(satellites_B[:, 0], satellites_B[:, 1], color='green',marker="^",s=100, label='Satellites B')
    plt.scatter(hub_B[0], hub_B[1], color='purple', marker="x", s=100, label='Hub B')
    plt.title('Routes for both cities')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid()
    plt.show()

# Plot routes 1st echelon
plot_routes_combined(hub_A[0], satellites_A, routes_A, hub_B[0], satellites_B, routes_B)


Final_cost_2_echelons= total_cost_second_echelon + total_cost_first_echelon_A + total_cost_first_echelon_B
print("The cost for the whole trips in the two echelon problem is:", Final_cost_2_echelons )


# Constants for speed of vehicles
speed_minivan = 800  # Speed of minivans in u/h
speed_bike = 300  # Speed of bicycles in u/h

# Function to calculate total time for minivans (first echelon)
def calculate_total_time_first_echelon(hub, satellites, routes, speed):
    total_distance = 0
    for route in routes:
        if route:
            path = [hub] + [satellites[j] for j in route if j != -1] + [hub]
            for i in range(len(path) - 1):
                total_distance += np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))

    return compute_total_time(total_distance, speed)

# Calculate routes and distances for minivans (first echelon) in both cities
routes_A_minivan = vrp_nearest_neighbor_multi_vehicle(hub_A[0], satellites_A, demand_A, minivan_capacity, num_minivans)
routes_B_minivan = vrp_nearest_neighbor_multi_vehicle(hub_B[0], satellites_B, demand_B, minivan_capacity, num_minivans)

service_time_sat=0.4
# Calculate total time for first echelon in both cities
total_time_first_echelon_A = calculate_total_time_first_echelon(hub_A[0], satellites_A, routes_A_minivan, speed_minivan)/num_minivans
total_time_first_echelon_B = calculate_total_time_first_echelon(hub_B[0], satellites_B, routes_B_minivan, speed_minivan)/num_minivans
total_time_first_echelon_A += (service_time_sat*len(satellites_A))
total_time_first_echelon_B += (service_time_sat*len(satellites_B))
print(f"Total time for first echelon city A: {total_time_first_echelon_A} hours")
print(f"Total time for first echelon city B: {total_time_first_echelon_B} hours")

service_time_customers= 0.1
# Calculate total time for bicycles (second echelon) in city A
total_time_second_echelon_A = calculate_total_time_second_echelon(distances_A, speed_bike)
total_time_second_echelon_A +=  (service_time_customers*(len(pickup_coords_day_A)+len(delivery_coords_day_A)))
total_time_second_echelon_A = total_time_second_echelon_A/(num_vehicles*4)
# Calculate total time for bicycles (second echelon) in city B
total_time_second_echelon_B = calculate_total_time_second_echelon(distances_B, speed_bike)/(num_vehicles)
total_time_second_echelon_B += (service_time_customers*(len(pickup_coords_day_B)+len(delivery_coords_day_B)))
total_time_second_echelon_B = total_time_second_echelon_B/(num_vehicles*4)

print(f"Total time for second echelon city A: {total_time_second_echelon_A} hours")
print(f"Total time for second echelon city B: {total_time_second_echelon_B} hours")

# Calculate total time for both echelons
total_time_A = total_time_first_echelon_A + total_time_second_echelon_A
total_time_B = total_time_first_echelon_B + total_time_second_echelon_B

print(f"\nTotal time for city A: {total_time_A} hours")
print(f"Total time for city B: {total_time_B} hours")

semitrailer_speed=1200
cost_semitrailer=20
# Compute Euclidean distance, time, and cost between hubs
def compute_distance_time_cost(hub_A, hub_B, speed, cost_per_km):
    hub_A = np.array(hub_A[0])
    hub_B = np.array(hub_B[0])
    distance = np.linalg.norm(hub_A - hub_B)
    time = distance / speed
    cost = distance * cost_per_km
    return distance, time, cost

service_time_hub=0.8
distance_hub, time_hub, cost_hub = compute_distance_time_cost(hub_A, hub_B, semitrailer_speed, cost_semitrailer)
time_hub+=service_time_hub*2
print(f"Distance between hubs: {distance_hub:.2f} u")
print(f"Time between hubs: {time_hub:.2f} hours")
print(f"Cost between hubs: {cost_hub:.2f} currency units")
Avarage_total_cost_for_the_company_to_manage_all_parcels= total_cost_first_echelon_A+total_cost_first_echelon_B+total_cost_A+total_cost_B+cost_hub
Avarage_Total_time_for_a_parcel_to_complete_its_journey= total_time_A+total_time_B+time_hub

print(f"Avarage total cost for the company to manage all parcels: {Avarage_total_cost_for_the_company_to_manage_all_parcels:.2f} currency units")
print(f"Avarage Total time for a parcel to complete its journey: {Avarage_Total_time_for_a_parcel_to_complete_its_journey:.2f} hours")



# Total distance for each city
total_distance_A = sum(sum(distances_A[q]) for q in distances_A)
total_distance_B = sum(sum(distances_B[q]) for q in distances_B)
print(f"Total distance traveled in City A: {total_distance_A}")
print(f"Total distance traveled in City B: {total_distance_B}")

#Total distance runned by company
total_distance_covered_by_company= total_distance_A+total_distance_B+distance_hub
print(f"Total distance traveled by company: {total_distance_covered_by_company}")
print("Total Distance for Satellite A:", total_distance_A)
print("Total Distance for Satellite B:", total_distance_B)

#capacity utilization for bikes and minivans


total_demand_A = sum(demand_A)
total_demand_B = sum(demand_B)


total_capacity_A_bike = num_vehicles*4 * vehicle_capacity
total_capacity_B_bike = num_vehicles*4 * vehicle_capacity

utilization_A_bike = total_demand_A / total_capacity_A_bike
utilization_B_bike = total_demand_B / total_capacity_B_bike

print(f"Utilization Capacity for bikes at city  A: {utilization_A_bike * 100:.2f}%")
print(f"Utilization Capacity for bikes at city B: {utilization_B_bike * 100:.2f}%")
print(f"Utilization Capacity for bikes: {((utilization_A_bike+utilization_B_bike)/2) * 100:.2f}%")
total_capacity_A_minivan = num_minivans * minivan_capacity
total_capacity_B_minivan = num_minivans * minivan_capacity

utilization_A_minivan = total_demand_A / total_capacity_A_minivan
utilization_B_minivan = total_demand_B / total_capacity_B_minivan

print(f"Utilization Capacity for minivans at city  A: {utilization_A_minivan * 100:.2f}%")
print(f"Utilization Capacity for minivans at city B: {utilization_B_minivan * 100:.2f}%")
print(f"Utilization Capacity for minivans: {((utilization_A_minivan+utilization_B_minivan)/2) * 100:.2f}%")