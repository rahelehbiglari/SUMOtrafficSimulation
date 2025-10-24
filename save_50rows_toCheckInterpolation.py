import pandas as pd

# Load the CSV file
file_name = 'Data/training_data/filtered_training_fileLD.csv'

# Read the file into a DataFrame
df = pd.read_csv(file_name)

# Extract the first 50 rows
first_50_rows = df.iloc[1:51]

# Save the first 50 rows to a new CSV file
output_file_name = 'Data/training_data/first_50_filtered_training_fileLD.csv'
first_50_rows.to_csv(output_file_name, index=False)

print(f"First 50 rows saved to {output_file_name}")