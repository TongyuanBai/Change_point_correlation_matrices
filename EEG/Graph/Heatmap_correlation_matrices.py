# -*- coding: utf-8 -*-
"""
Correlation of two matrices
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

# Calculate correlation matrices
correlation_matrices = calculate_correlation_matrix(df, 100)

matrix_50 = correlation_matrices[99]
matrix_450 = correlation_matrices[399]

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(2, 1, figsize=(10, 16))

sns.heatmap(matrix_50, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=ax[0])
ax[0].set_title("The Heatmap of the 50th Correlation Matrix")

sns.heatmap(matrix_450, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=ax[1])
ax[1].set_title("The Heatmap of the 450th Correlation Matrix")

plt.tight_layout()
plt.show()