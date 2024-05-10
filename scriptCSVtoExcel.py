import pandas as pd

# Replace these with your file paths
csv_file_path = '/Users/juanluengo/Desktop/Estudios/Universidades/4° Carrera/Quartile 4/Design for transport and logistics/Project/instance_set/Set A/A2_1500_1.csv'
excel_file_path = 'EXCEL_A2_1500_1.xlsx'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Write the DataFrame to an Excel file
df.to_excel(excel_file_path, index=False)

print(f"Successfully converted '{csv_file_path}' to '{excel_file_path}'.")
