# -*- coding: utf-8 -*-
"""
#The distance matrix by Log-Euclidean-Cholesky metrics
"""

import numpy as np
from tqdm import tqdm
from scipy.linalg import logm
import os

# Function to calculate correlation matrices for a time series with a given window size
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

# Function to calculate normalized log matrices for a set of correlation matrices
def calculate_normalized_log_matrices(correlation_matrices):
    normalized_log_matrices = []
    for matrix in correlation_matrices:
        cholesky = np.linalg.cholesky(matrix)
        D_inv = np.diag(1 / np.diag(cholesky))
        normalized_cholesky = D_inv @ cholesky
        normalized_log_matrix = logm(normalized_cholesky)
        normalized_log_matrices.append(normalized_log_matrix)
    return normalized_log_matrices

# Function to compute the distance matrix based on the normalized log matrices
def calculate_distance_matrix(normalized_log_matrices):
    num_matrices = len(normalized_log_matrices)
    distance_matrix = np.zeros((num_matrices, num_matrices))
    
    for i in tqdm(range(num_matrices)):
        for j in range(i+1, num_matrices): # only compute upper triangular part
            distance = np.linalg.norm(normalized_log_matrices[j] - normalized_log_matrices[i], 'fro')
            distance_matrix[i, j] = distance
    
    # copy upper triangular part to lower triangular part
    distance_matrix += distance_matrix.T - np.diag(distance_matrix.diagonal())
    
    return distance_matrix

# Base paths for the input time series datasets and the output distance matrices
base_path_input = 'E:/Project/dataset/'
base_path_output = 'E:/Project/distance_matrix/Log_Euclidean_Cholesky'
window_size1 = 10

# Iterate over each subdirectory in the base input path
for subdirectory in os.listdir(base_path_input):
    subdirectory_path_input = os.path.join(base_path_input, subdirectory)
    
    if os.path.isdir(subdirectory_path_input):
        # Create a corresponding subdirectory in the output path for storing distance matrices
        subdirectory_path_output = os.path.join(base_path_output, subdirectory)
        os.makedirs(subdirectory_path_output, exist_ok=True)
        
        # Process each time series file in the current subdirectory
        for i in range(50): 
            input_file = os.path.join(subdirectory_path_input, f'time_series_{i}.csv')
            output_file = os.path.join(subdirectory_path_output, f'distance_matrix_{i}.csv')

            # Load the time series data from the file
            combined_time_series = np.genfromtxt(input_file, delimiter=',')

            # Compute correlation matrices for the loaded time series
            correlation_matrices = calculate_correlation_matrix(combined_time_series, window_size1)

            # Compute normalized log matrices for the correlation matrices
            normalized_log_matrices = calculate_normalized_log_matrices(correlation_matrices)
            
            # Compute the distance matrix using the normalized log matrices
            distance_matrix = calculate_distance_matrix(normalized_log_matrices)

            # Save the computed distance matrix to a file
            np.savetxt(output_file, distance_matrix, delimiter=',')  