import torch
import pandas as pd
import torch.nn as nn

# Define the deeper model architecture
# class DeepModel(nn.Module):
#     def __init__(self):
#         super(DeepModel, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(8096, 4096),
#             nn.ReLU(),
#             nn.Linear(4096, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, x):
#         return self.network(x)
class DeepModel(nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(8096, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

# Load the saved deep model
# model_path = "deep_model_HighLowInput.pth" 
model_path = "deep_model_HighLowInputDiffTL.pth" 
model = DeepModel()
model.load_state_dict(torch.load(model_path))  # Load trained weights
model.eval()  # Set the model to evaluation mode

# Read the CSV file and extract the first row as input
# csv_file_path = "Data/training_data/first_50_rows_diffTL.csv"
csv_file_path = "Data/training_data/first_50_rows_HD.csv"
# csv_file_path = "Data/training_data/inputDenseP_extrapolation.csv"

# csv_file_path = "Data/training_data/inputDensePExtrapolationDiffTl.csv"
data = pd.read_csv(csv_file_path, header=None) 
example_input = data.iloc[0].tolist()  # Extract the first row and convert it to a list

# Padding or resizing input to match the model's expected input size (8096)
if len(example_input) < 8096:
    example_input += [0] * (8096 - len(example_input))  # Pad with zeros to make the length 8096
elif len(example_input) > 8096:
    example_input = example_input[:8096]  # Truncate the input if it's longer than 8096

# Convert input to tensor
input_tensor = torch.tensor([example_input], dtype=torch.float32)  # Add batch dimension

# Make prediction
with torch.no_grad():
    prediction = model(input_tensor)

print("Prediction:", prediction.item())
