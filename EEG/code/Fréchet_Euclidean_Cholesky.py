# -*- coding: utf-8 -*-
"""
Fréchet method with Euclidean-Cholesky metrics
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
import pandas as pd
from scipy.linalg import sqrtm, solve
from tqdm import tqdm

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

def diffeomorphism(C):
    chol_C = np.linalg.cholesky(C)
    chol_c_processed = solve(np.diag(np.diag(chol_C)), chol_C)
    return chol_c_processed

def inverse_diffeomorphism(Gamma):
    Gamma_T = Gamma.T
    Gamma_Square = Gamma @ Gamma_T
    Diag_Gamma_Square = np.diag(np.diag(Gamma_Square))
    Diag_Gamma_Square_processed = np.linalg.inv(sqrtm(Diag_Gamma_Square)) @ Gamma_Square @ np.linalg.inv(sqrtm(Diag_Gamma_Square))
    return Diag_Gamma_Square_processed

def frechet_mean_correlation_matrices(correlation_matrices):
    diffeomorphisms = [diffeomorphism(corr_mat) for corr_mat in correlation_matrices]
    mean_diffeomorphism = np.mean(diffeomorphisms, axis=0)
    frechet_mean = inverse_diffeomorphism(mean_diffeomorphism)
    return frechet_mean

def calculate_dsq_sigma(sam):
    frechet_mean_sam = frechet_mean_correlation_matrices(sam)
    dsq = [np.sum((x - frechet_mean_sam)**2) for x in sam]
    sigma = np.mean(np.array(dsq)**2) - np.mean(dsq)**2
    return dsq, sigma

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

def stat_calc_boot(input_list, indices):
    sam = [input_list["data"][i] for i in indices]
    c = input_list["c"]
    n = len(sam)
    dsq, sigma = calculate_dsq_sigma(sam)
    fstat = [frechet_test_statistic((j, n-j), sam, list(range(n)), dsq, sigma) for j in range(int(c*n), n-int(c*n)+1)]
    # Save fstat to CSV
    fstat_df = pd.DataFrame(fstat, columns=["fstat"])
    fstat_df.to_csv("E:/Project/EEG/Frechet/fstat_values.csv", index=False)
    return {"t": max(fstat), "tau": np.argmax(fstat)}

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
    return {"cp_est": tau_f, "pval_boot": p_boot}  # Return fstat as well


def calculate_mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

correlation_matrices = calculate_correlation_matrix(df, 100)
cp_result = cp_test(correlation_matrices, 0.1)
tau_f = cp_result["cp_est"]

mae = calculate_mae(340, tau_f)
#cp=385,mae=45