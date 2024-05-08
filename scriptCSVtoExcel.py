import pandas as pd

# Replace these with your file paths
csv_file_path = 'your_input_file.csv'
excel_file_path = 'your_output_file.xlsx'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Write the DataFrame to an Excel file
df.to_excel(excel_file_path, index=False)

print(f"Successfully converted '{csv_file_path}' to '{excel_file_path}'.")
