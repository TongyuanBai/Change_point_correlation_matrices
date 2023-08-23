# -*- coding: utf-8 -*-
"""
#The distance matrix by Poly-Hyperbolic-Cholesky metrics
"""

import numpy as np
import math
from tqdm import tqdm
import os

# Compute the correlation matrix for each window of a time series
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

# Calculate the distance matrix based on hyperbolic Cholesky distance
def calculate_distance_matrix(correlation_matrices):
    num_matrices = len(correlation_matrices)
    distance_matrix = np.zeros((num_matrices, num_matrices))
    
    for i in tqdm(range(num_matrices)):
        for j in range(i+1, num_matrices):
            distance = hyperbolic_cholesky_distance(correlation_matrices[i], correlation_matrices[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix

# Compute the Cholesky decomposition for each correlation matrix
def calculate_cholesky_matrices(correlation_matrices):
    cholesky_matrices = [np.linalg.cholesky(matrix) for matrix in correlation_matrices]
    return cholesky_matrices

# Helper function to calculate Q, a component for the hyperbolic distance
def calculate_Q(matrix1, matrix2, t):
    nonzero_counts = [np.count_nonzero(row) for row in matrix1]
    k = nonzero_counts[t]
    Q_t = np.sum(matrix1[t, :k-1].T * matrix2[t, :k-1].T) - matrix1[t, k-1] * matrix2[t, k-1]
    return Q_t

# Calculate hyperbolic distance between two Cholesky matrices
def hyperbolic_cholesky_distance(cholesky1, cholesky2):
    n = cholesky1.shape[0]
    Q_values = []
    for t in range(1, n):
        Q_i = calculate_Q(cholesky1, cholesky2, t)
        distance = math.acos(-Q_i) ** 2
        Q_values.append(distance)
    sum_Q_values = sum(Q_values)
    return sum_Q_values
    
base_path_input = 'E:/Project/dataset/'
base_path_output = 'E:/Project/distance_matrix/Poly_Hyperbolic/'
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
            
            # Compute the Cholesky decomposition for each correlation matrix
            cholesky_matrices = calculate_cholesky_matrices(correlation_matrices)

            # Compute the distance matrix based on hyperbolic Cholesky distance
            distance_matrix = calculate_distance_matrix(cholesky_matrices)

            # Save the computed distance matrix to a file
            np.savetxt(output_file, distance_matrix, delimiter=',')  