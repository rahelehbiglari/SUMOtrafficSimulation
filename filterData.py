import pandas as pd

# Read the CSV file
df = pd.read_csv('Data/training_data/intersection_lane_vehiclesP.csv')

# Filter out  4 first and 4 last states
filtered_df = df[~df['State File'].isin(['_100.00.xml','_200.00.xml''_1300.00.xml', '_1400.00.xml'])]

filtered_df.to_csv('Data/training_data/filtered2last2firstTL_file.csv', index=False)

