# -*- coding: utf-8 -*-
"""
Boxplots for graph method
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set extraction directory
extraction_dir = "E:/Project/Result"

directories = sorted([d for d in os.listdir(extraction_dir) if os.path.isdir(os.path.join(extraction_dir, d))])
subdirectories = sorted([d for d in os.listdir(os.path.join(extraction_dir, "Poly_Hyperbolic_ori")) if os.path.isdir(os.path.join(extraction_dir, "Poly_Hyperbolic_ori", d))])

# Extract data: Load 'error.csv' 
dataframes = {}
for subdirectory in subdirectories:
    file_path = os.path.join(extraction_dir, "Poly_Hyperbolic_ori", subdirectory, "error.csv")
    dataframes[subdirectory] = pd.read_csv(file_path)

# Define order for the x-axis and the palette for plotting
x_order = ['1-MST', '1-NNG', '3-MST', '3-NNG', '5-MST', '5-NNG', '7-MST', '7-NNG', '9-MST', '9-NNG']
metrics = ['predict_ori_r', 'predict_weighted_r', 'predict_maxtype_r', 'predict_generalized_r']
new_legend_names = {
    'predict_ori_r': 'Original Edge-count Scan Statistic',
    'predict_weighted_r': 'Weighted Edge-count Scan Statistic',
    'predict_maxtype_r': 'Max-Type Edge-count Scan Statistic',
    'predict_generalized_r': 'Generalized Edge-count Scan Statistic'
}
sci_palette = sns.color_palette("husl", 4)

for key, dataframe in dataframes.items():
    dataframe['source'] = dataframe['source'].replace('NNL', 'NNG')

custom_order = [
    'case_0_eigen_0.1_1', 'case_0_eigen_1_5', 'case_0_eigen_5_10', 
    'case_1_eigen_0.1_1', 'case_1_eigen_1_5', 'case_1_eigen_5_10',
    'case_2_eigen_0.1_1', 'case_2_eigen_1_5', 'case_2_eigen_5_10'
]
custom_titles = ["Case1", "Case2", "Case3", "Case4", "Case5", "Case6", "Case7", "Case8", "Case9"]
# Initialize the figure again
fig, axes = plt.subplots(3, 3, figsize=(15, 13))

for index, key in enumerate(custom_order):
    dataframe = dataframes[key]
    ax = axes.flatten()[index]
    
    # Combine k and source columns for the x-axis
    dataframe['x_value'] = dataframe['k'].astype(str) + "-" + dataframe['source']
    melted_df = dataframe.melt(id_vars=['x_value'], value_vars=metrics)
    
    # Plotting
    sns.boxplot(data=melted_df, x='x_value', y='value', hue='variable', order=x_order, palette=sci_palette, ax=ax)

    # Set titles, labels, and scale
    ax.set_title(custom_titles[index], fontsize=12)
    ax.set_yscale("log")
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order, rotation=45, fontsize=11)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=11)
    ax.set_xlabel("Graphics", fontsize=11)
    ax.set_ylabel("Log Scale MAE", fontsize=11)
    ax.legend([],[], frameon=False)

# Place the legend on bottom center
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles[:4], [new_legend_names[label] for label in labels[:4]], title='', loc='lower center', prop={'size':11}, ncol=4, bbox_to_anchor=(0.5, -0.05))

# Adjust layout
plt.tight_layout()
plt.show()

################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set extraction directory
extraction_dir = "E:/Project/Result"

directories = sorted([d for d in os.listdir(extraction_dir) if os.path.isdir(os.path.join(extraction_dir, d))])
subdirectories = sorted([d for d in os.listdir(os.path.join(extraction_dir, "Euclidean_Cholesky_ori")) if os.path.isdir(os.path.join(extraction_dir, "Euclidean_Cholesky_ori", d))])

# Extract data: Load 'error.csv' 
dataframes = {}
for subdirectory in subdirectories:
    file_path = os.path.join(extraction_dir, "Euclidean_Cholesky_ori", subdirectory, "error.csv")
    dataframes[subdirectory] = pd.read_csv(file_path)

# Define order for the x-axis and the palette for plotting
x_order = ['1-MST', '1-NNG', '3-MST', '3-NNG', '5-MST', '5-NNG', '7-MST', '7-NNG', '9-MST', '9-NNG']
metrics = ['predict_ori_r', 'predict_weighted_r', 'predict_maxtype_r', 'predict_generalized_r']
new_legend_names = {
    'predict_ori_r': 'Original Edge-count Scan Statistic',
    'predict_weighted_r': 'Weighted Edge-count Scan Statistic',
    'predict_maxtype_r': 'Max-Type Edge-count Scan Statistic',
    'predict_generalized_r': 'Generalized Edge-count Scan Statistic'
}
sci_palette = sns.color_palette("husl", 4)

for key, dataframe in dataframes.items():
    dataframe['source'] = dataframe['source'].replace('NNL', 'NNG')

custom_order = [
    'case_0_eigen_0.1_1', 'case_0_eigen_1_5', 'case_0_eigen_5_10', 
    'case_1_eigen_0.1_1', 'case_1_eigen_1_5', 'case_1_eigen_5_10',
    'case_2_eigen_0.1_1', 'case_2_eigen_1_5', 'case_2_eigen_5_10'
]
custom_titles = ["Case1", "Case2", "Case3", "Case4", "Case5", "Case6", "Case7", "Case8", "Case9"]
# Initialize the figure again
fig, axes = plt.subplots(3, 3, figsize=(15, 13))

for index, key in enumerate(custom_order):
    dataframe = dataframes[key]
    ax = axes.flatten()[index]
    
    # Combine k and source columns for the x-axis
    dataframe['x_value'] = dataframe['k'].astype(str) + "-" + dataframe['source']
    melted_df = dataframe.melt(id_vars=['x_value'], value_vars=metrics)
    
    # Plotting
    sns.boxplot(data=melted_df, x='x_value', y='value', hue='variable', order=x_order, palette=sci_palette, ax=ax)

    # Set titles, labels, and scale
    ax.set_title(custom_titles[index], fontsize=12)
    ax.set_yscale("log")
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order, rotation=45, fontsize=11)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=11)
    ax.set_xlabel("Graphics", fontsize=11)
    ax.set_ylabel("Log Scale MAE", fontsize=11)
    ax.legend([],[], frameon=False)

# Place the legend on bottom center
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles[:4], [new_legend_names[label] for label in labels[:4]], title='', loc='lower center', prop={'size':11}, ncol=4, bbox_to_anchor=(0.5, -0.05))

# Adjust layout
plt.tight_layout()
plt.show()

################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set extraction directory
extraction_dir = "E:/Project/Result"

directories = sorted([d for d in os.listdir(extraction_dir) if os.path.isdir(os.path.join(extraction_dir, d))])
subdirectories = sorted([d for d in os.listdir(os.path.join(extraction_dir, "Log_Euclidean_Cholesky_ori")) if os.path.isdir(os.path.join(extraction_dir, "Log_Euclidean_Cholesky_ori", d))])

# Extract data: Load 'error.csv' 
dataframes = {}
for subdirectory in subdirectories:
    file_path = os.path.join(extraction_dir, "Log_Euclidean_Cholesky_ori", subdirectory, "error.csv")
    dataframes[subdirectory] = pd.read_csv(file_path)

# Define order for the x-axis and the palette for plotting
x_order = ['1-MST', '1-NNG', '3-MST', '3-NNG', '5-MST', '5-NNG', '7-MST', '7-NNG', '9-MST', '9-NNG']
metrics = ['predict_ori_r', 'predict_weighted_r', 'predict_maxtype_r', 'predict_generalized_r']
new_legend_names = {
    'predict_ori_r': 'Original Edge-count Scan Statistic',
    'predict_weighted_r': 'Weighted Edge-count Scan Statistic',
    'predict_maxtype_r': 'Max-Type Edge-count Scan Statistic',
    'predict_generalized_r': 'Generalized Edge-count Scan Statistic'
}
sci_palette = sns.color_palette("husl", 4)

for key, dataframe in dataframes.items():
    dataframe['source'] = dataframe['source'].replace('NNL', 'NNG')

custom_order = [
    'case_0_eigen_0.1_1', 'case_0_eigen_1_5', 'case_0_eigen_5_10', 
    'case_1_eigen_0.1_1', 'case_1_eigen_1_5', 'case_1_eigen_5_10',
    'case_2_eigen_0.1_1', 'case_2_eigen_1_5', 'case_2_eigen_5_10'
]
custom_titles = ["Case1", "Case2", "Case3", "Case4", "Case5", "Case6", "Case7", "Case8", "Case9"]
# Initialize the figure again
fig, axes = plt.subplots(3, 3, figsize=(15, 13))

for index, key in enumerate(custom_order):
    dataframe = dataframes[key]
    ax = axes.flatten()[index]
    
    # Combine k and source columns for the x-axis
    dataframe['x_value'] = dataframe['k'].astype(str) + "-" + dataframe['source']
    melted_df = dataframe.melt(id_vars=['x_value'], value_vars=metrics)
    
    # Plotting
    sns.boxplot(data=melted_df, x='x_value', y='value', hue='variable', order=x_order, palette=sci_palette, ax=ax)

    # Set titles, labels, and scale
    ax.set_title(custom_titles[index], fontsize=12)
    ax.set_yscale("log")
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order, rotation=45, fontsize=11)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=11)
    ax.set_xlabel("Graphics", fontsize=11)
    ax.set_ylabel("Log Scale MAE", fontsize=11)
    ax.legend([],[], frameon=False)

# Place the legend on bottom center
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles[:4], [new_legend_names[label] for label in labels[:4]], title='', loc='lower center', prop={'size':11}, ncol=4, bbox_to_anchor=(0.5, -0.05))

# Adjust layout
plt.tight_layout()
plt.show()