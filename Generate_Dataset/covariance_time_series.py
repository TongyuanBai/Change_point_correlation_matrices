# -*- coding: utf-8 -*-
"""
Generating the time series
"""
import os
import numpy as np
from tqdm import tqdm

# Simulate a covariance matrix
# Define the fixed eigenvector
np.random.seed(42)
fixed_eigenvectors = np.random.rand(5, 5)
q_fixed, _ = np.linalg.qr(fixed_eigenvectors)

def simulate_covariance_matrix(n, eigenvalue_range=(0.1, 1)):
    eigenvalues = np.random.uniform(*eigenvalue_range, size=n)
    eigenvalues.sort()
    covariance_matrix = q_fixed[:, :n] @ np.diag(eigenvalues) @ q_fixed[:, :n].T
    return covariance_matrix

# Generate a time series based on covariance matrix
def generate_time_series(cov_matrix, num_rows):
    num_cols = cov_matrix.shape[0]
    time_series = np.random.multivariate_normal(np.zeros(num_cols), cov_matrix, size=num_rows)
    return time_series

# Check if two matrices are different
def matrices_are_different(matrix1, matrix2, tolerance=1e-6):
    frobenius_norm = np.linalg.norm(matrix1 - matrix2, 'fro')
    return frobenius_norm > tolerance

# Main execution
base_path = 'E:/Project/dataset'
dimension = 5
row_sizes = [(1000, 2000), (1500, 1500), (2000, 1000)]
eigenvalue_ranges = [(0.1, 1), (1, 5), (5, 10)]

for i in tqdm(range(50)):
    for case_num, (num_rows1, num_rows2) in enumerate(row_sizes):
        for eigenvalue_range in eigenvalue_ranges:
            
            # Generate covariance matrices
            cov_matrix1 = simulate_covariance_matrix(dimension, eigenvalue_range)
            while True:
                cov_matrix2 = simulate_covariance_matrix(dimension, eigenvalue_range)
                if matrices_are_different(cov_matrix1, cov_matrix2):
                    break
                    
            # Generate time series
            time_series1 = generate_time_series(cov_matrix1, num_rows1)
            time_series2 = generate_time_series(cov_matrix2, num_rows2)
            combined_time_series = np.vstack((time_series1, time_series2))
            
            # Save to CSV
            subfolder_path = os.path.join(base_path, f'case_{case_num}_eigen_{eigenvalue_range[0]}_{eigenvalue_range[1]}')
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            output_file_time_series = os.path.join(subfolder_path, f'time_series_{i}.csv')
            np.savetxt(output_file_time_series, combined_time_series, delimiter=',')