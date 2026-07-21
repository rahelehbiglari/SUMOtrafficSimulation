# SUMO Traffic Simulation and Surrogate Modelling

This repository contains the experimental code for the paper:

**R. Biglari, C. Gomes, and J. Denil**,  
“Towards a Validity Frame of Multi-Modal Surrogate Models for Traffic Simulation,”  
accepted as a full paper at the **Annual Modeling and Simulation Conference (ANNSIM 2025)**, Madrid, Spain.

The project investigates the use of surrogate models for traffic simulation and studies how validity frames can support the assessment of multi-modal surrogate models under interpolation, extrapolation, and different traffic-density conditions.

## Project Overview

High-fidelity traffic simulations can be expensive to run repeatedly, especially when they are used for analysis, optimisation, or decision support. This project uses SUMO micro-traffic simulations to generate data and train surrogate models that approximate simulation outputs.

The main goals of the repository are to:

- generate traffic simulation data using SUMO;
- train and evaluate surrogate models;
- compare linear regression models and deep neural networks;
- study interpolation and extrapolation behaviour;
- remove redundant data samples using distance-based filtering;
- retrain specialised models for different traffic-density regimes;
- support the development of validity frames for multi-modal surrogate models.

## Research Context

This repository supports research on validity-aware surrogate modelling for traffic simulation. The experiments are designed to analyse when a surrogate model can be trusted, where its predictions become less reliable, and how data selection or model specialisation can improve prediction performance.

The work was conducted in collaboration with **Prof. Claudio Gomes from Aarhus University** and **Prof. Joachim Denil from the University of Antwerp**.

## My Contribution

I designed and implemented the simulation-data workflow, prepared datasets from SUMO micro-traffic simulations, developed the model training and evaluation scripts, compared linear regression models with deep neural networks, and analysed prediction performance under different data-filtering and model-specialisation strategies.

## Environment

The experiments were developed using:

- Python 3.11.5
- PyTorch
- SUMO
- TraCI
- sumolib
- pandas
- NumPy
- SciPy
- scikit-learn
- Matplotlib

## Data Generation

Traffic data is generated using SUMO simulations.

The script `DataGenerator_parallel.py` generates random trips for different traffic-density scenarios:

- low-density scenario: 15 trips
- high-density scenario: 25 trips
- extrapolation scenario: 50 trips

These generated datasets are used to train and evaluate surrogate models under different operating conditions.

## Interpolation and Extrapolation Experiments

The repository includes scripts for preparing and evaluating data samples used to study interpolation and extrapolation behaviour.

Relevant scripts include:

- `save_50rows_toCheckInterpolation`
- `predict_deep_model_samples`

These scripts help evaluate how well the trained surrogate models generalise to data points inside and outside the original training distribution.

## Redundancy Removal

The repository includes a redundancy-removal workflow based on pairwise distance calculations.

The general workflow is:

1. Run `remove_redundantData.py`
2. Calculate pairwise distances between samples
3. Plot the distance heatmap
4. Remove redundant samples
5. Plot the updated heatmap
6. Retrain the deep learning model
7. Evaluate the model on selected samples

Relevant scripts include:

- `remove_redundantData.py`
- `reTrainter.py`
- `predict_deep_model_samples.py`

This workflow was used to reduce dataset redundancy while maintaining or improving model performance.

## Planned Extension

A planned extension of this project is the integration of a genetic optimiser for self-adaptation. The goal is to use optimisation techniques to support adaptive model selection and decision-making in future versions of the validity-frame workflow.

## Relevance

This repository demonstrates experience in:

- Python-based research software development;
- SUMO-based traffic simulation;
- simulation-data generation;
- surrogate modelling;
- PyTorch-based deep learning;
- GPU-supported model training;
- interpolation and extrapolation analysis;
- distance-based redundancy filtering;
- model evaluation and performance analysis;
- validity-aware approximation for simulation-based systems.

## Citation

```bibtex
@inproceedings{biglari2025validityframe,
  author    = {Biglari, Raheleh and Gomes, Claudio and Denil, Joachim},
  title     = {Towards a Validity Frame of Multi-Modal Surrogate Models for Traffic Simulation},
  booktitle = {Annual Modeling and Simulation Conference},
  year      = {2025},
  address   = {Madrid, Spain}
}
```