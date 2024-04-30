import pandas as pd # to read csv files 
import matplotlib.pyplot as plt # for plotting
from sklearn.cluster import KMeans # for clustering 

# Load the Dataset: make sure to change the path to the one in your computer!
df = pd.read_csv('/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A1_1500_1.csv')

print(df.head()) # for taking a look of the first rows of the dataset

# We start visualizing the dataset
plt.scatter(df['X'], df['Y'])
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Pickup and Delivery Locations')
plt.show()

# We will use the KMeans algorithm to cluster the locations into X different neighborhoods
kmeans = KMeans(n_clusters=5) # set the amount of neighborhoods (clusters) to create
df['Neighborhood'] = kmeans.fit_predict(df[['X', 'Y']]) # add the column 'Neighborhood' to the dataset

# Calculate centroids of neighborhoods
centroids = df.groupby('Neighborhood')[['X', 'Y']].mean() 

# Determine city hub location
city_hub_location = centroids.mean()
print("City Hub Location:")
print(city_hub_location)


# For selecting the satellite locations,
satellite_locations = df.groupby('Neighborhood').apply(lambda x: x.iloc[x[['X', 'Y']].sub(city_hub_location).pow(2).sum(1).idxmin() if not x.empty else None])[['X', 'Y']]
print("\nSatellite Locations:")
print(satellite_locations)


# Visualize city hubs and satellites
plt.scatter(df['X'], df['Y'], label='Pickup/Delivery Locations')
plt.scatter(city_hub_location['X'], city_hub_location['Y'], color='yellow', label='City Hub', marker='^', s=100)  # Increase marker size for city hub
plt.scatter(satellite_locations['X'], satellite_locations['Y'], color='red', label='Satellite', marker='s', s=50)  # Increase marker size for satellites

# Set labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('City Hub, Satellites, and Pickup/Delivery Locations')

# Add legend
plt.legend()

# Show plot
plt.show()