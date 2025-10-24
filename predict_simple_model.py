import torch
import pandas as pd

# Define the simple model structure
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(8096, 1)  # Ensure the input dimension matches the trained model

    def forward(self, x):
        return self.linear(x)

# Load the saved model
# model_path = "simple_model_HighLowInput.pth"
model_path = "simple_model_HighLowInputSameTL.pth"
model = LinearRegression()
model.load_state_dict(torch.load(model_path))  # Load trained weights
model.eval()  # Set the model to evaluation mode



# Read the CSV file and extract the first row as input
# csv_file_path = "Data/training_data/inputDensePExtrapolation.csv" 
csv_file_path = "Data/training_data/inputDenseP.csv" 
# csv_file_path = "Data/training_data/inputDensePExtrapolationDiffTl.csv"

data = pd.read_csv(csv_file_path, header=None)  # Read the CSV file without headers
for j in range(13) :
    example_input = data.iloc[j].tolist()  # Extract the first row and convert it to a list

    # Padding or resizing input to match the model's expected input size (8096)
    # Assuming the rest of the input values are padded with 0s
    if len(example_input) < 8096:
        example_input += [0] * (8096 - len(example_input))  # Pad with zeros to make the length 8096
    elif len(example_input) > 8096:
        example_input = example_input[:8096]  # Truncate the input if it's longer than 8096

    # Convert input to tensor
    input_tensor = torch.tensor([example_input], dtype=torch.float32)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)

    # Output the prediction
    print("Prediction:", prediction.item())
