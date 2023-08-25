# -*- coding: utf-8 -*-
"""
#The distance matrix by Euclidean-Cholesky metrics
"""


import numpy as np
from tqdm import tqdm
import os

# Compute correlation matrix for each window of a time series
def calculate_correlation_matrix(time_series, window_size):
    n = time_series.shape[0]
    num_windows = n // window_size
    correlation_matrices = []
    
    for i in range(num_windows):
        start_index = i * window_size
        end_index = (i + 1) * window_size
        window_data = time_series[start_index:end_index]
        correlation_matrix = np.corrcoef(window_data.T)
        correlation_matrices.append(correlation_matrix)
    
    return correlation_matrices

# Calculate Cholesky normalized matrices from correlation matrices
def calculate_cholesky_normalized_matrices(correlation_matrices):
    cholesky_normalized_matrices = []
    for matrix in correlation_matrices:
        cholesky = np.linalg.cholesky(matrix)
        D_inv = np.diag(1 / np.diag(cholesky))
        cholesky_normalized = D_inv @ cholesky
        cholesky_normalized -= np.eye(cholesky_normalized.shape[0]) # subtract identity matrix
        cholesky_normalized_matrices.append(cholesky_normalized)
    return cholesky_normalized_matrices

# Calculate the distance matrix based on Frobenius norm for the Cholesky normalized matrices
def calculate_distance_matrix(cholesky_normalized_matrices):
    num_matrices = len(cholesky_normalized_matrices)
    distance_matrix = np.zeros((num_matrices, num_matrices))
    
    for i in tqdm(range(num_matrices)):
        for j in range(i+1, num_matrices): # only compute upper triangular part
            distance = np.linalg.norm(cholesky_normalized_matrices[j] - cholesky_normalized_matrices[i], 'fro')
            distance_matrix[i, j] = distance
    
    # copy upper triangular part to lower triangular part
    distance_matrix += distance_matrix.T - np.diag(distance_matrix.diagonal())
    
    return distance_matrix

base_path_input = 'E:/Project/Dataset/'
base_path_output = 'E:/Project/Distance_Matrix/Euclidean_Cholesky'
window_size1 = 10

# Loop over all subdirectories in the input path
for subdirectory in os.listdir(base_path_input):
    subdirectory_path_input = os.path.join(base_path_input, subdirectory)
    
    # Check if it's a directory
    if os.path.isdir(subdirectory_path_input):
        # Prepare corresponding output directory and create if it doesn't exist
        subdirectory_path_output = os.path.join(base_path_output, subdirectory)
        os.makedirs(subdirectory_path_output, exist_ok=True)
        
        # Process each file within the subdirectory
        for i in range(50): 
            input_file = os.path.join(subdirectory_path_input, f'time_series_{i}.csv')
            output_file = os.path.join(subdirectory_path_output, f'distance_matrix_{i}.csv')

            # Load the time series data from the file
            combined_time_series = np.genfromtxt(input_file, delimiter=',')

            # Compute the correlation matrices for the loaded time series
            correlation_matrices = calculate_correlation_matrix(combined_time_series, window_size1)

            # Compute Cholesky normalized matrices
            cholesky_normalized_matrices = calculate_cholesky_normalized_matrices(correlation_matrices)

            # Compute the distance matrix
            distance_matrix = calculate_distance_matrix(cholesky_normalized_matrices)

            # Save the computed distance matrix to a file
            np.savetxt(output_file, distance_matrix, delimiter=',')
            
base_path_input = 'E:/Project/dataset_nnl/'
base_path_output = 'E:/Project/Distance_Matrix/Euclidean_Cholesky_nnl'
window_size1 = 10

# List all subdirectories
for subdirectory in os.listdir(base_path_input):
    subdirectory_path_input = os.path.join(base_path_input, subdirectory)
    if os.path.isdir(subdirectory_path_input):
        # Create a corresponding subdirectory in the output base path
        subdirectory_path_output = os.path.join(base_path_output, subdirectory)
        os.makedirs(subdirectory_path_output, exist_ok=True)
        
        # Process all files in this subdirectory
        for i in range(50): 
            input_file = os.path.join(subdirectory_path_input, f'time_series_{i}.csv')
            output_file = os.path.join(subdirectory_path_output, f'distance_matrix_{i}.csv')

            # Load time series
            combined_time_series = np.genfromtxt(input_file, delimiter=',')

            # Calculate correlation matrices
            correlation_matrices = calculate_correlation_matrix(combined_time_series, window_size1)

            # Calculate Cholesky normalized matrices
            cholesky_normalized_matrices = calculate_cholesky_normalized_matrices(correlation_matrices)

            # Calculate distance matrix
            distance_matrix = calculate_distance_matrix(cholesky_normalized_matrices)

            # Save distance matrix
            np.savetxt(output_file, distance_matrix, delimiter=',')          