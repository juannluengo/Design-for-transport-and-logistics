# coordinates_district = pd.read_csv("/Users/dailagencarelli/Documents/Design4Transport&Logistic/instance_set/Set A/A1_1500_1.csv")
# coordinates_district = pd.read_csv("/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A1_1500_1.csv")

import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

# Read data
coordinates_district = pd.read_csv("/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A1_1500_1.csv")
pickup_coords = coordinates_district.iloc[:1500].copy()
delivery_coords = coordinates_district.iloc[1500:].copy()

pickup_coords_A = pickup_coords[pickup_coords['Origin'] == 1].copy()
pickup_coords_B = pickup_coords[pickup_coords['Origin'] == 2].copy()
delivery_coords_A = delivery_coords[delivery_coords['Origin'] == 1].copy()
delivery_coords_B = delivery_coords[delivery_coords['Origin'] == 2].copy()

coords_A = pd.concat([pickup_coords_A, delivery_coords_A], axis=0).copy()

# Translation vector for city B
translation_vector = np.array([300, 300])

# Applying translation to the copies of the DataFrame
pickup_coords_B[['X', 'Y']] += translation_vector
delivery_coords_B[['X', 'Y']] += translation_vector

# Reconstruct coords_B after translation
coords_B = pd.concat([pickup_coords_B, delivery_coords_B], axis=0).copy()

# Calculate centroids for each district
hub_A = coords_A[['X', 'Y']].mean()
hub_B = coords_B[['X', 'Y']].mean()

# Define truck capacity and demand
truck_capacity = 200 
daily_demand_A_to_B = pickup_coords_A['Demand'].sum()  # Total daily demand from A to B
daily_demand_B_to_A = pickup_coords_B['Demand'].sum()  # Total daily demand from B to A

# Number of trucks needed (considering rounding up)
trucks_needed_A_to_B = int(np.ceil(daily_demand_A_to_B / truck_capacity))
trucks_needed_B_to_A = int(np.ceil(daily_demand_B_to_A / truck_capacity))

# Create Gurobi model
model = Model('ParcelAssignment')

# Variables
x_A_to_B = model.addVars(trucks_needed_A_to_B, vtype=GRB.BINARY, name="x_A_to_B")
x_B_to_A = model.addVars(trucks_needed_B_to_A, vtype=GRB.BINARY, name="x_B_to_A")

# Objective: Minimize the number of trucks used
model.setObjective(quicksum(x_A_to_B) + quicksum(x_B_to_A), GRB.MINIMIZE)

# Constraints
model.addConstr(quicksum(x_A_to_B) * truck_capacity >= daily_demand_A_to_B, "Demand_A_to_B")
model.addConstr(quicksum(x_B_to_A) * truck_capacity >= daily_demand_B_to_A, "Demand_B_to_A")

# Optimize model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal number of trucks from A to B: {sum(x_A_to_B[i].x for i in range(trucks_needed_A_to_B))}")
    print(f"Optimal number of trucks from B to A: {sum(x_B_to_A[i].x for i in range(trucks_needed_B_to_A))}")

# For more detailed analysis, print the values of each variable
for v in model.getVars():
    print(f"{v.varName}: {v.x}")
