import pandas as pd

# Load the CSV file
file_path = "/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A1_1500_1.csv"
data = pd.read_csv(file_path)

# Drop rows where "Day-Ops" is NaN
day_ops_data = data['Day-Ops'].dropna()

# Count the occurrences of each number in the "Day-Ops" column
day_ops_counts = day_ops_data.value_counts().sort_index()

# Convert the counts to a dictionary
day_ops_dict = day_ops_counts.to_dict()

# Print the dictionary
print(day_ops_dict)
