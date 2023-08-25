# -*- coding: utf-8 -*-
"""
Graph method with three metrices in EEG
"""

import os
import pandas as pd

extract_path = 'E:/Project/EEG/'

extracted_files = os.listdir(extract_path)

ori_folder_path = os.path.join(extract_path, 'ori')
ori_files = os.listdir(ori_folder_path)

columns = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
           'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Pz']

if len(ori_files) != len(columns):
    raise ValueError("Mismatch between number of files and provided column names.")

df = pd.DataFrame()

for file_name, column in zip(ori_files, columns):
    file_path = os.path.join(ori_folder_path, file_name)
    df[column] = pd.read_csv(file_path, header=None, squeeze=True)

import numpy as np
from tqdm import tqdm
import os

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

def calculate_cholesky_normalized_matrices(correlation_matrices):
    cholesky_normalized_matrices = []
    for matrix in correlation_matrices:
        cholesky = np.linalg.cholesky(matrix)
        D_inv = np.diag(1 / np.diag(cholesky))
        cholesky_normalized = D_inv @ cholesky
        cholesky_normalized -= np.eye(cholesky_normalized.shape[0]) # subtract identity matrix
        cholesky_normalized_matrices.append(cholesky_normalized)
    return cholesky_normalized_matrices

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

# Calculate correlation matrices
correlation_matrices_euc = calculate_correlation_matrix(df, 100)

# Calculate Cholesky normalized matrices
cholesky_normalized_matrices_euc = calculate_cholesky_normalized_matrices(correlation_matrices_euc)

# Calculate distance matrix
distance_matrix_euc = calculate_distance_matrix(cholesky_normalized_matrices_euc)

np.savetxt('E:/Project/EEG/Euclidean/distance_matrix_euclidean.csv', distance_matrix_euc, delimiter=',')

import numpy as np
from tqdm import tqdm
from scipy.linalg import logm
import os

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

def calculate_normalized_log_matrices(correlation_matrices):
    normalized_log_matrices = []
    for matrix in correlation_matrices:
        cholesky = np.linalg.cholesky(matrix)
        D_inv = np.diag(1 / np.diag(cholesky))
        normalized_cholesky = D_inv @ cholesky
        normalized_log_matrix = logm(normalized_cholesky)
        normalized_log_matrices.append(normalized_log_matrix)
    return normalized_log_matrices

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

# Calculate correlation matrices
correlation_matrices_log_euc = calculate_correlation_matrix(df, 100)

# Calculate Cholesky normalized matrices
cholesky_normalized_matrices_log_euc = calculate_normalized_log_matrices(correlation_matrices_log_euc)

# Calculate distance matrix
distance_matrix_log_euc = calculate_distance_matrix(cholesky_normalized_matrices_log_euc)

np.savetxt('E:/Project/EEG/Log_Euclidean/distance_matrix_log_euclidean.csv', distance_matrix_log_euc, delimiter=',')

###Poly_Hyperbolic
import numpy as np
import math
from tqdm import tqdm
import os

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

def calculate_distance_matrix(correlation_matrices):
    num_matrices = len(correlation_matrices)
    distance_matrix = np.zeros((num_matrices, num_matrices))
    
    for i in tqdm(range(num_matrices)):
        for j in range(i+1, num_matrices):
            distance = hyperbolic_cholesky_distance(correlation_matrices[i], correlation_matrices[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix

def calculate_cholesky_matrices(correlation_matrices):
    cholesky_matrices = [np.linalg.cholesky(matrix) for matrix in correlation_matrices]
    return cholesky_matrices

def calculate_Q(matrix1, matrix2, t):
    nonzero_counts = [np.count_nonzero(row) for row in matrix1]
    k = nonzero_counts[t]
    Q_t = np.sum(matrix1[t, :k-1].T * matrix2[t, :k-1].T) - matrix1[t, k-1] * matrix2[t, k-1]
    return Q_t

def hyperbolic_cholesky_distance(cholesky1, cholesky2):
    n = cholesky1.shape[0]
    Q_values = []
    for t in range(1, n):
        Q_i = calculate_Q(cholesky1, cholesky2, t)
        distance = math.acos(-Q_i) ** 2
        Q_values.append(distance)
    sum_Q_values = sum(Q_values)
    return sum_Q_values

# Calculate correlation matrices
correlation_matrices_poly = calculate_correlation_matrix(df, 100)

cholesky_matrices_poly = calculate_cholesky_matrices(correlation_matrices_poly)

distance_matrix_poly = calculate_distance_matrix(cholesky_matrices_poly)

# Save distance matrix
np.savetxt('E:/Project/EEG/Poly_hyperbolic/distance_matrix_poly_hyperbolic.csv', distance_matrix_poly, delimiter=',')  
