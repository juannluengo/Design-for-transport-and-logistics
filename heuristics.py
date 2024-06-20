import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dati forniti
satellites_A = [[28.21, 76.14], [80.02, 74.59], [25.5, 22.0], [81.13, 24.87]]
satellites_B = [[174.65, 70.27], [229.47, 75.61], [226.50, 26.45], [176.75, 20.85]]

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

# Definisci i limiti per dividere in 4 aree
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

# Seleziona il giorno
while True:
    try:
        day_input = int(input("Insert the interested day (1-7): "))
        if 1 <= day_input <= 7:
            break
        else:
            print("Insert a number between 1 and 7")
    except ValueError:
        print("Insert a valid number.")

# Seleziona gli ordini per il giorno specificato
pickup_coords_day_A = pickup_coords_A[pickup_coords_A['Day-Ops'] == day_input].copy()
delivery_coords_day_A = delivery_coords_A[delivery_coords_A['Day-Ops'] == day_input].copy()
pickup_coords_day_B = pickup_coords_B[pickup_coords_B['Day-Ops'] == day_input].copy()
delivery_coords_day_B = delivery_coords_B[delivery_coords_B['Day-Ops'] == day_input].copy()

# Dividi nei quadranti
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

# Calcolo della distanza euclidea
def compute_euclidean_distance_matrix(locations):
    distances = np.zeros((len(locations), len(locations)))
    for i in range(len(locations)):
        for j in range(len(locations)):
            if i != j:
                distances[i][j] = np.linalg.norm(locations[i] - locations[j])
    return distances

# Funzione per calcolare la matrice delle distanze per i punti in un quadrante
def calculate_distances_in_quadrant(quadrant_pickup, quadrant_delivery, satellites):
    if len(quadrant_pickup) == 0 and len(quadrant_delivery) == 0:
        return np.array([])

    all_points = np.concatenate([quadrant_pickup[['X', 'Y']].values, quadrant_delivery[['X', 'Y']].values, satellites])
    return compute_euclidean_distance_matrix(all_points)

# Calcola le distanze per ogni quadrante in città A e B
distance_matrices_A = {}
distance_matrices_B = {}

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    distance_matrices_A[q] = calculate_distances_in_quadrant(quadrants_A_pickup[q], quadrants_A_delivery[q], satellites_A)
    distance_matrices_B[q] = calculate_distances_in_quadrant(quadrants_B_pickup[q], quadrants_B_delivery[q], satellites_B)

# Funzione per calcolare il percorso usando l'algoritmo del più vicino vicino
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

                if distance < nearest_distance and current_capacities[vehicle] + point[2] <= vehicle_capacity:
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


# Converti i dati in un formato utilizzabile per VRP
def convert_to_vrp_format(coords):
    formatted_coords = []
    
    for index, row in coords.iterrows():
        x, y, origin, demand = row['X'], row['Y'], row['Type'], row['Demand']
        if origin == 3:  # Punto di pickup
            formatted_coords.append((x, y, demand))
        elif origin == 2:  # Punto di delivery
            formatted_coords.append((x, y, -demand))  # Domanda negativa per i punti di delivery
    
    return formatted_coords



vehicle_capacity = 10
num_vehicles = 3  # Ad esempio, 3 biciclette

# Risolvere il VRP per ciascun quadrante in città A con satelliti come punti di partenza
routes_A = {}
distances_A = {}

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    delivery_points = convert_to_vrp_format(quadrants_A_delivery[q])
    pickup_points = convert_to_vrp_format(quadrants_A_pickup[q])
    satellite_routes = []
    satellite_distances = []

    for satellite in satellites_A:
        routes, distances = nearest_neighbor_vrp(satellite, delivery_points, pickup_points, vehicle_capacity, num_vehicles)
        # Aggiungi il punto di partenza (satellite) all'inizio di ogni percorso
        routes_with_start = [[satellite] + route for route in routes]
        satellite_routes.append(routes_with_start)
        satellite_distances.append(distances)

    best_route_index = np.argmin([sum(d) for d in satellite_distances])
    routes_A[q] = satellite_routes[best_route_index]
    distances_A[q] = satellite_distances[best_route_index]

# Risolvere il VRP per ciascun quadrante in città B con satelliti come punti di partenza
routes_B = {}
distances_B = {}

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    delivery_points = convert_to_vrp_format(quadrants_B_delivery[q])
    pickup_points = convert_to_vrp_format(quadrants_B_pickup[q])
    satellite_routes = []
    satellite_distances = []

    for satellite in satellites_B:
        routes, distances = nearest_neighbor_vrp(satellite, delivery_points, pickup_points, vehicle_capacity, num_vehicles)
        # Aggiungi il punto di partenza (satellite) all'inizio di ogni percorso
        routes_with_start = [[satellite] + route for route in routes]
        satellite_routes.append(routes_with_start)
        satellite_distances.append(distances)

    best_route_index = np.argmin([sum(d) for d in satellite_distances])
    routes_B[q] = satellite_routes[best_route_index]
    distances_B[q] = satellite_distances[best_route_index]


# Stampa i risultati
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

# Plot della mappa con i percorsi
plt.figure(figsize=(14, 10))

# Plot pickup e delivery per città A
plt.scatter(pickup_coords_day_A['X'], pickup_coords_day_A['Y'], c='blue', label='Pickup City A', alpha=0.6)
plt.scatter(delivery_coords_day_A['X'], delivery_coords_day_A['Y'], c='cyan', label='Delivery City A', alpha=0.6)

# Plot pickup e delivery per città B
plt.scatter(pickup_coords_day_B['X'], pickup_coords_day_B['Y'], c='red', label='Pickup City B', alpha=0.6)
plt.scatter(delivery_coords_day_B['X'], delivery_coords_day_B['Y'], c='orange', label='Delivery City B', alpha=0.6)

# Plot satelliti e hub per città A e B
sat_A = np.array(satellites_A)
sat_B = np.array(satellites_B)

plt.scatter(sat_A[:, 0], sat_A[:, 1], c='green', marker='^', s=100, label='Satellites A')
plt.scatter(sat_B[:, 0], sat_B[:, 1], c='green', marker='s', s=100, label='Satellites B')

# Disegna i confini per le aree nella città A
plt.plot([x_min_A, x_max_A], [y_min_A + y_step_A, y_min_A + y_step_A], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_A, x_max_A], [y_min_A + 2*y_step_A, y_min_A + 2*y_step_A], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_A + x_step_A, x_min_A + x_step_A], [y_min_A, y_max_A], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_A + 2*x_step_A, x_min_A + 2*x_step_A], [y_min_A, y_max_A], c='gray', linestyle='--', linewidth=2)

# Disegna i confini per le aree nella città B
plt.plot([x_min_B, x_max_B], [y_min_B + y_step_B, y_min_B + y_step_B], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_B, x_max_B], [y_min_B + 2*y_step_B, y_min_B + 2*y_step_B], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_B + x_step_B, x_min_B + x_step_B], [y_min_B, y_max_B], c='gray', linestyle='--', linewidth=2)
plt.plot([x_min_B + 2*x_step_B, x_min_B + 2*x_step_B], [y_min_B, y_max_B], c='gray', linestyle='--', linewidth=2)

# Plot delle rotte per i quadranti della città A
colors = ['g--', 'b--', 'm--']  # Per diverse biciclette
for q in routes_A:
    for vehicle in range(num_vehicles):
        route = routes_A[q][vehicle]
        # Estrai le coordinate per il plot
        route_coords = np.array([[point[0], point[1]] for point in route])  # Extract X and Y coordinates
        plt.plot(route_coords[:, 0], route_coords[:, 1], colors[vehicle % len(colors)])

# Plot delle rotte per i quadranti della città B
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
