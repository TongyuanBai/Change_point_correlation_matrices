# -*- coding: utf-8 -*-
"""
Fréchet method vs Graph method
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define paths and parameters for data extraction and plotting
extraction_dir = "E:/Project/Result"

# Define the order for the x-axis and other parameters for plotting
x_order = ['Fréchet_Euclidean', '1-MST', '1-NNG', '3-MST', '3-NNG', '5-MST', '5-NNG', '7-MST', '7-NNG', '9-MST', '9-NNG']
sci_palette = sns.color_palette("husl", 5)
new_legend_names = {
    'MAE': 'Fréchet_Euclidean',
    'predict_ori_r': 'Original Edge-count Scan Statistic',
    'predict_weighted_r': 'Weighted Edge-count Scan Statistic',
    'predict_maxtype_r': 'Max-Type Edge-count Scan Statistic',
    'predict_generalized_r': 'Generalized Edge-count Scan Statistic' 
}
metrics = ['predict_ori_r', 'predict_weighted_r', 'predict_maxtype_r', 'predict_generalized_r']
metrics_fre = ['MAE']

# Load data
directories = sorted([d for d in os.listdir(extraction_dir) if os.path.isdir(os.path.join(extraction_dir, d))])
subdirectories = sorted([d for d in os.listdir(os.path.join(extraction_dir, "Euclidean_Cholesky_ori")) if os.path.isdir(os.path.join(extraction_dir, "Euclidean_Cholesky_ori", d))])

# Reading the data and replace 'NNL' with 'NNG' in the 'source' column
dataframes = {}
for subdirectory in subdirectories:
    file_path = os.path.join(extraction_dir, "Euclidean_Cholesky_ori", subdirectory, "error.csv")
    dataframes[subdirectory] = pd.read_csv(file_path)
    dataframes[subdirectory]['source'] = dataframes[subdirectory]['source'].replace('NNL', 'NNG')

subdirectories_fre = sorted([d for d in os.listdir(os.path.join(extraction_dir, "Frechet_euclidean_ori")) if os.path.isdir(os.path.join(extraction_dir, "Frechet_euclidean_ori", d))])
dataframes_fre = {}
for subdirectory_fre in subdirectories_fre:
    file_path_fre = os.path.join(extraction_dir, "Frechet_euclidean_ori", subdirectory_fre, "mae_results.csv")
    dataframes_fre[subdirectory_fre] = pd.read_csv(file_path_fre)

# Define combinations for custom_order and custom_titles
combinations = [
    (['case_0_eigen_0.1_1', 'case_1_eigen_0.1_1', 'case_2_eigen_0.1_1'], ["Case1", "Case4", "Case7"]),
    (['case_0_eigen_1_5', 'case_1_eigen_1_5', 'case_2_eigen_1_5'], ["Case2", "Case5", "Case8"]),
    (['case_0_eigen_5_10', 'case_1_eigen_5_10', 'case_2_eigen_5_10'], ["Case3", "Case6", "Case9"])
]

save_path = "E:/Project/Graph/compare/euclidean/" 

for idx, (custom_order, custom_titles) in enumerate(combinations):
    fig, axes = plt.subplots(3, 1, figsize=(13, 15), sharex=True)
    for index, key in enumerate(custom_order):
        ax = axes[index]
        
        df_euc = dataframes[key]
        df_euc['x_value'] = df_euc['k'].astype(str) + "-" + df_euc['source']
        melted_euc = df_euc.melt(id_vars=['x_value'], value_vars=metrics)
        melted_euc['value'] = np.log(melted_euc['value'])  # Apply natural logarithm
        
        df_fre = dataframes_fre[key]
        df_fre['x_value'] = 'Fréchet_Euclidean'
        melted_fre = df_fre.melt(id_vars=['x_value'], value_vars=metrics_fre)
        melted_fre['value'] = np.log(melted_fre['value'])  # Apply natural logarithm
        
        combined_melted = pd.concat([melted_euc, melted_fre], ignore_index=True)
        sns.boxplot(data=combined_melted, x='x_value', y='value', hue='variable', order=x_order, palette=sci_palette, ax=ax)
        
        ax.set_title(custom_titles[index], fontsize=11)
        # Removed the line setting yscale to log since we applied manual transformation
        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels(x_order, rotation=45, fontsize=11)
        ax.set_yticklabels(ax.get_yticks(), fontsize=11)
        ax.set_xlabel("Methods", fontsize=11)
        ax.set_ylabel("Log Transformed MAE", fontsize=11)  # Update ylabel
        ax.legend([], [], frameon=False)
        ax.tick_params(axis='both', which='both', length=0)

    handles, labels = axes[-1].get_legend_handles_labels()
    ordered_handles = [handles[labels.index(l)] for l in new_legend_names.keys()]
    ordered_labels = list(new_legend_names.values())
    
    fig.legend(ordered_handles, ordered_labels, title='', loc='lower center', prop={'size':11}, ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(pad=1.0)
    
    # Save the figure
    filename = f"combined_plot_euc_{idx+1}.png"
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
    plt.close()

####################
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Paths and parameters
extraction_dir = "E:/Project/Result"
x_order = ['Fréchet_Log_Euclidean', '1-MST', '1-NNG', '3-MST', '3-NNG', '5-MST', '5-NNG', '7-MST', '7-NNG', '9-MST', '9-NNG']
sci_palette = sns.color_palette("husl", 5)
new_legend_names = {
    'MAE': 'Fréchet_Log_Euclidean',
    'predict_ori_r': 'Original Edge-count Scan Statistic',
    'predict_weighted_r': 'Weighted Edge-count Scan Statistic',
    'predict_maxtype_r': 'Max-Type Edge-count Scan Statistic',
    'predict_generalized_r': 'Generalized Edge-count Scan Statistic' 
}
metrics = ['predict_ori_r', 'predict_weighted_r', 'predict_maxtype_r', 'predict_generalized_r']
metrics_fre = ['MAE']

# Load data
directories = sorted([d for d in os.listdir(extraction_dir) if os.path.isdir(os.path.join(extraction_dir, d))])
subdirectories = sorted([d for d in os.listdir(os.path.join(extraction_dir, "Log_Euclidean_Cholesky_ori")) if os.path.isdir(os.path.join(extraction_dir, "Log_Euclidean_Cholesky_ori", d))])
dataframes = {}
for subdirectory in subdirectories:
    file_path = os.path.join(extraction_dir, "Log_Euclidean_Cholesky_ori", subdirectory, "error.csv")
    dataframes[subdirectory] = pd.read_csv(file_path)
    dataframes[subdirectory]['source'] = dataframes[subdirectory]['source'].replace('NNL', 'NNG')

subdirectories_fre = sorted([d for d in os.listdir(os.path.join(extraction_dir, "Frechet_log_euclidean_ori")) if os.path.isdir(os.path.join(extraction_dir, "Frechet_log_euclidean_ori", d))])
dataframes_fre = {}
for subdirectory_fre in subdirectories_fre:
    file_path_fre = os.path.join(extraction_dir, "Frechet_log_euclidean_ori", subdirectory_fre, "mae_results.csv")
    dataframes_fre[subdirectory_fre] = pd.read_csv(file_path_fre)

# Define combinations for custom_order and custom_titles
combinations = [
    (['case_0_eigen_0.1_1', 'case_1_eigen_0.1_1', 'case_2_eigen_0.1_1'], ["Case1", "Case4", "Case7"]),
    (['case_0_eigen_1_5', 'case_1_eigen_1_5', 'case_2_eigen_1_5'], ["Case2", "Case5", "Case8"]),
    (['case_0_eigen_5_10', 'case_1_eigen_5_10', 'case_2_eigen_5_10'], ["Case3", "Case6", "Case9"])
]

save_path = "E:/Project/Graph/compare/log_euclidean/" 

for idx, (custom_order, custom_titles) in enumerate(combinations):
    fig, axes = plt.subplots(3, 1, figsize=(13, 15), sharex=True)
    for index, key in enumerate(custom_order):
        ax = axes[index]
        df_euc = dataframes[key]
        df_euc['x_value'] = df_euc['k'].astype(str) + "-" + df_euc['source']
        melted_euc = df_euc.melt(id_vars=['x_value'], value_vars=metrics)
        melted_euc['value'] = np.log(melted_euc['value'])  # Apply natural logarithm
        
        df_fre = dataframes_fre[key]
        df_fre['x_value'] = 'Fréchet_Log_Euclidean'
        melted_fre = df_fre.melt(id_vars=['x_value'], value_vars=metrics_fre)
        melted_fre['value'] = np.log(melted_fre['value'])  # Apply natural logarithm
        
        combined_melted = pd.concat([melted_euc, melted_fre], ignore_index=True)
        sns.boxplot(data=combined_melted, x='x_value', y='value', hue='variable', order=x_order, palette=sci_palette, ax=ax)
        
        ax.set_title(custom_titles[index], fontsize=11)
        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels(x_order, rotation=45, fontsize=11)
        ax.set_yticklabels(ax.get_yticks(), fontsize=11)
        ax.set_xlabel("Methods", fontsize=11)
        ax.set_ylabel("Log Scale MAE", fontsize=11)
        ax.legend([], [], frameon=False)
        ax.tick_params(axis='both', which='both', length=0)

    handles, labels = axes[-1].get_legend_handles_labels()
    ordered_handles = [handles[labels.index(l)] for l in new_legend_names.keys()]
    ordered_labels = list(new_legend_names.values())
    
    fig.legend(ordered_handles, ordered_labels, title='', loc='lower center', prop={'size':11}, ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(pad=1.0)
    
    # Save the figure
    filename = f"combined_plot_log_euc_{idx+1}.png"
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
    plt.close()
