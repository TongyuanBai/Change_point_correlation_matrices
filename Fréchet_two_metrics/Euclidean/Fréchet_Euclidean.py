# -*- coding: utf-8 -*-
"""
#Euclidean-Cholesky metrics
"""

import os
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm, solve
from tqdm import tqdm

# Calculate correlation matrix for each window of a time series
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

# Perform diffeomorphism on a matrix
def diffeomorphism(C):
    chol_C = np.linalg.cholesky(C)
    chol_c_processed = solve(np.diag(np.diag(chol_C)), chol_C)
    return chol_c_processed

# Calculate the inverse of the diffeomorphism
def inverse_diffeomorphism(Gamma):
    Gamma_T = Gamma.T
    Gamma_Square = Gamma @ Gamma_T
    Diag_Gamma_Square = np.diag(np.diag(Gamma_Square))
    Diag_Gamma_Square_processed = np.linalg.inv(sqrtm(Diag_Gamma_Square)) @ Gamma_Square @ np.linalg.inv(sqrtm(Diag_Gamma_Square))
    return Diag_Gamma_Square_processed

# Calculate the Fr\'chet mean of a list of correlation matrices
def frechet_mean_correlation_matrices(correlation_matrices):
    diffeomorphisms = [diffeomorphism(corr_mat) for corr_mat in correlation_matrices]
    mean_diffeomorphism = np.mean(diffeomorphisms, axis=0)
    frechet_mean = inverse_diffeomorphism(mean_diffeomorphism)
    return frechet_mean

# Calculate D^2 metric and its variance for a sample
def calculate_dsq_sigma(sam):
    frechet_mean_sam = frechet_mean_correlation_matrices(sam)
    dsq = [np.sum((x - frechet_mean_sam)**2) for x in sam]
    sigma = np.mean(np.array(dsq)**2) - np.mean(dsq)**2
    return dsq, sigma

# Calculate the Fr\'chet test statistic
def frechet_test_statistic(sizes, data, indices, dsq, sigma):
    sam = [data[i] for i in indices]
    n1, n2 = sizes
    n = n1 + n2
    lambda_ = n1 / n
    sam1 = sam[:n1]
    sam2 = sam[n1:]
    m1 = frechet_mean_correlation_matrices(sam1)
    m2 = frechet_mean_correlation_matrices(sam2)
    V1 = np.mean([np.sum((x - m1)**2) for x in sam1])
    V2 = np.mean([np.sum((x - m2)**2) for x in sam2])
    V1n = np.mean([np.sum((x - m2)**2) for x in sam1])
    V2n = np.mean([np.sum((x - m1)**2) for x in sam2])
    add_factor = (np.sqrt(n) * (V1n - V1)) + (np.sqrt(n) * (V2n - V2))
    test_stat = ((lambda_ * (1 - lambda_) * ((n * (V1 - V2)**2) + add_factor**2)) / sigma)
    return test_stat

# Calculate the test statistic for bootstrap samples
def stat_calc_boot(input_list, indices):
    sam = [input_list["data"][i] for i in indices]
    c = input_list["c"]
    n = len(sam)
    dsq, sigma = calculate_dsq_sigma(sam)
    fstat = [frechet_test_statistic((j, n-j), sam, list(range(n)), dsq, sigma) for j in range(int(c*n), n-int(c*n)+1)]
    return {"t": max(fstat), "tau": np.argmax(fstat)}

# Perform the change point test
def cp_test(data, c):
    n = len(data)
    n0 = int(c*n)
    res_stat = stat_calc_boot({"data": data, "c": c}, list(range(n)))
    tau_f = res_stat["tau"] + n0 - 1
    m_f = res_stat["t"]
    boot_sample = [np.random.choice(n, n, replace=True) for _ in range(100)]
    res_boot = [stat_calc_boot({"data": data, "c": c}, boot_sample[i]) for i in range(100)]
    t = [res_boot[i]["t"] for i in range(100)]
    p_boot = 1 - np.argmin(np.abs(m_f - np.sort(t))) / len(t)
    return {"cp_est": tau_f, "pval_boot": p_boot}

# Calculate mean absolute error
def calculate_mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

# Paths for data and results
data_folder_path = "E:/Project/dataset"
results_folder_path = "E:/Project/Result/Frechet_euclidean"

# Define true values and folders
true_values = {"case_0_eigen_0.1_1": 100, 
               "case_0_eigen_1_5": 100, 
               "case_0_eigen_5_10": 100, 
               "case_1_eigen_0.1_1": 150, 
               "case_1_eigen_1_5": 150, 
               "case_1_eigen_5_10": 150, 
               "case_2_eigen_0.1_1": 200, 
               "case_2_eigen_1_5": 200, 
               "case_2_eigen_5_10": 200}

folders = ["case_0_eigen_0.1_1", "case_0_eigen_1_5", "case_0_eigen_5_10", 
           "case_1_eigen_0.1_1", "case_1_eigen_1_5", "case_1_eigen_5_10", 
           "case_2_eigen_0.1_1", "case_2_eigen_1_5", "case_2_eigen_5_10"]

ranges = [(0, 3000), (0, 3000), (0, 3000),
          (0, 3000), (0, 3000), (0, 3000),
          (0, 3000), (0, 3000), (0, 3000)]

# Ensure folders and ranges have the same length
assert len(folders) == len(ranges)

# Iterate through each folder and its corresponding range
for folder, range_ in zip(folders, ranges):
    start, end = range_
    file_list = [os.path.join(data_folder_path, folder, f) for f in os.listdir(os.path.join(data_folder_path, folder)) if f.endswith(".csv")]
    mae_list = []
    cp_list = []

    # Process each file in the current folder
    for file in tqdm(file_list, desc=f"Processing files for {folder}"):
        data = pd.read_csv(file, header=None)
        data = data.iloc[start:end] 
        correlation_matrices = calculate_correlation_matrix(data, 10)
        cp_result = cp_test(correlation_matrices, 0.1)
        tau_f = cp_result["cp_est"]
        cp_list.append(tau_f)
        mae = calculate_mae(true_values[folder], tau_f)
        mae_list.append(mae)

    # Save the results
    results_subfolder_path = os.path.join(results_folder_path, folder)
    if not os.path.exists(results_subfolder_path):
        os.makedirs(results_subfolder_path)
    pd.DataFrame({"MAE": mae_list}).to_csv(os.path.join(results_subfolder_path, "mae_results.csv"), index=False)
    pd.DataFrame({"cp": cp_list}).to_csv(os.path.join(results_subfolder_path, "cp_results.csv"), index=False)