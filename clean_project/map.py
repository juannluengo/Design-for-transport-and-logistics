import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_map():
    # Read data
    coordinates_district = pd.read_csv("/Users/dailagencarelli/Desktop/Design-for-transport-and-logistics/A1_1500_1.csv")
    coordinates_district.fillna(0,inplace=True)
    print(coordinates_district)
    pickup_coords = coordinates_district.iloc[:1500].copy()  
    delivery_coords = coordinates_district.iloc[1500:].copy()  

    pickup_coords_A = pickup_coords[pickup_coords['Origin'] == 1].copy()
    pickup_coords_B = pickup_coords[pickup_coords['Origin'] == 2].copy()
    delivery_coords_A = delivery_coords[delivery_coords['Origin'] == 1].copy()
    delivery_coords_B = delivery_coords[delivery_coords['Origin'] == 2].copy()

    max_point_A = max(pickup_coords_A['X'].max(), delivery_coords_A['X'].max())
    min_point_A = min(pickup_coords_A['X'].min(), delivery_coords_A['X'].min())

    pickup_coords_B['X'] += max_point_A + (max_point_A-min_point_A)*0.1
    delivery_coords_B['X'] += max_point_A + (max_point_A-min_point_A)*0.1

    return (delivery_coords_A, delivery_coords_B), (pickup_coords_A, pickup_coords_B)
    

def plot_map(delivery, pickup, plot_A=True, plot_B=True):
    (delivery_coords_A, delivery_coords_B), (pickup_coords_A, pickup_coords_B) = delivery, pickup
    plt.figure(figsize=(10,6))
    plt.title("Coordinates districts A and B")
    if plot_A:
        plt.scatter(pickup_coords_A['X'], pickup_coords_A['Y'], color='blue', label='Pickups')
        plt.scatter(delivery_coords_A['X'], delivery_coords_A['Y'], color='red', label='Deliveries')
        if not plot_B: plt.title("Coordinates district A")
    if plot_B:
        plt.scatter(pickup_coords_B['X'], pickup_coords_B['Y'], color='blue', label='Pickups')
        plt.scatter(delivery_coords_B['X'], delivery_coords_B['Y'], color='red', label='Deliveries')
        if not plot_A: plt.title("Coordinates district B")
    plt.grid(True)
    plt.legend()
    plt.show()

