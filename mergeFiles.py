import pandas as pd
# df1 = pd.read_csv("D:/intersection_lane_vehiclesPLow.csv", header=None)
# df2 = pd.read_csv("D:/intersection_lane_vehiclesPHigh.csv", header=None, skiprows=1)
df1 = pd.read_csv("D:/intersection_lane_vehiclesPLow.csv", header=None, on_bad_lines='skip')
df2 = pd.read_csv("D:/intersection_lane_vehiclesPHigh.csv", header=None, skiprows=1, on_bad_lines='skip')
# df1 = pd.read_csv("D:\OneDrive - Universiteit Antwerpen/UAntwerpen/experiments/experiments/SameTL_HighDense2025_01_02/intersection_lane_vehiclesPHigh.csv", header=None)
combined_df = pd.concat([df1, df2])
combined_df.to_csv('Data/training_data/intersection_lane_vehiclesP.csv', header=False, index=False)



print(f"Merged file saved as")
