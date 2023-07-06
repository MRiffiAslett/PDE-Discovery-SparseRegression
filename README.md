
# PDE-Discovery-SparseRegression

This repository contains code for discovering partial differential equation (PDE) terms using sparse regression techniques.

## Description

The provided script trains a neural network to approximate PDE solutions and utilizes sparse regression with Lasso regularization to identify the underlying PDE terms. It demonstrates the process of generating synthetic data, training a neural network model, computing gradients using central differences, performing sparse regression, and visualizing the discovered PDE terms.

## Dependencies

- numpy==1.19.5
- matplotlib==3.4.3
- scipy==1.7.0
- scikit-learn==0.24.2
- torch==1.9.0

## Usage

1. Clone the repository:

git clone https://github.com/MaxRiffiAslett/PDE-Discovery-SparseRegression.git

## Install the required dependencies:

pip install -r requirements.txt

## Run the script:

python pde_discovery.py

