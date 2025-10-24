## Raheleh ANNSIM paper
Traffic Simulation by SUMO + validity frame of multi-modal systems
This is the running experiment of a paper which has been accepted as a full paper in Annual Modeling and Simulation (ANNSIM2025) conference presented in Madrid, Spain
Towards a Validity Frame of Multi-modal Surrogate Models for Traffic Simulation by Raheleh Biglari, Claudio Gomes, and Joachim Denil.
# TSdt environment
Python 3.11.5
pyTorch
traci for SUMO
sumolib
pandas
numpy
scipy
sklearn

# Genetic optimiser
the next step of this project is adding an optimiser for self adaptation

# Generating Data
DataGenerator_parallel
Generate random trips - 15 for low density and 25 for high density-50 for extrapolation

# interpolation/extrapolation
prepare data: save_50rows_toCheckInterpolation
preditct: predict_deep_model_samples

# remove redundancy
1. Run remove_redundantData - calculate distance - plot heatmap - remove redundant - plot heatmap
2. Run reTrainter - train deep model for 
3. Run predict_deep_model_samples







## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
