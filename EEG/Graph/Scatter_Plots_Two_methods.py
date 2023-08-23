# -*- coding: utf-8 -*-
"""
#Scatter Plots for Graph
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the enhanced sorting function
def enhanced_sort(column):
    prefix_order = ["ori", "weighted", "maxtype", "generalized"]
    k_order = ["k1", "k3", "k5", "k7", "k9"]
    method_order = ["mst", "nnl"]
    
    prefix = [x for x in prefix_order if x in column][0]
    k_val = [x for x in k_order if x in column][0]
    method = [x for x in method_order if x in column][0]
    
    return prefix_order.index(prefix), k_order.index(k_val), method_order.index(method)

# Load the data
data_poly = pd.read_csv("E:/Project/EEG/Poly_hyperbolic/combined_poly_data.csv")
data_euc = pd.read_csv("E:/Project/EEG/Euclidean/combined_euc.csv")
data_log = pd.read_csv("E:/Project/EEG/Log_Euclidean/combined_log_euc_data.csv")

# Create a 3x1 subplot layout
fig, axes = plt.subplots(3, 1, figsize=(10, 24))

# Data, title and axes list for iteration
data_list = [data_poly, data_euc, data_log]
titles = [
    'Different Graphics and Scan Statistics for Detection Change Point in EEG Dataset with Poly-Hyperbolic-Cholesky Metrics',
    'Different Graphics and Scan Statistics for Detection Change Point in EEG Dataset with Euclidean-Cholesky Metrics',
    'Different Graphics and Scan Statistics for Detection Change Point in EEG Dataset with Log-Euclidean-Cholesky Metrics'  
]

legend_labels = []
legend_handles = []

for ax, data, title in zip(axes, data_list, titles):
    enhanced_sorted_columns = sorted(data.columns, key=enhanced_sort)
    colors = plt.cm.jet(np.linspace(0, 1, len(enhanced_sorted_columns)))

    for idx, column in enumerate(enhanced_sorted_columns):
        line, = ax.plot(data[column], marker='o', markersize=3, linestyle='-', linewidth=1, color=colors[idx])
        
        # Only append the handles and labels from the first subplot to avoid duplicates
        if ax == axes[0]:
            legend_handles.append(line)
            legend_labels.append(column)

    ax.set_xlabel('Time Index', fontsize=11)
    ax.set_ylabel('Test Statistic', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=11)

# Create one legend for the whole figure on the right side
fig.legend(legend_handles, legend_labels, loc='center right', bbox_to_anchor=(1.28, 0.2), fontsize=11, ncol=1)

plt.tight_layout()  # Adjusting layout

plt.show()

##############################
#Scatter Plots for Fréchet
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
combined_frechet_data = pd.read_csv('E:/Project/EEG/Frechet/combined_frechet.csv')

# Apply ggplot style for sci-like coloring
plt.style.use('ggplot')

# Plotting the two columns as separate subplots with updated styles
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

# Plot for fstat_values_frechet
axes[0].scatter(range(len(combined_frechet_data)), combined_frechet_data['fstat_values_frechet'], 
                label='fstat_values_frechet', alpha=0.7, marker='o', s=50)
axes[0].set_title('Fréchet Method for Detection Change Point in EEG Dataset with Euclidean-Cholesky Metrics', fontsize=11)
axes[0].set_xlabel('Index', fontsize=11)
axes[0].set_ylabel('Test Statistic', fontsize=11)
axes[0].tick_params(labelsize=11)

# Plot for fstat_values_log_euclidean
axes[1].scatter(range(len(combined_frechet_data)), combined_frechet_data['fstat_values_log_euclidean'], 
                label='fstat_values_log_euclidean', alpha=0.7, marker='o', s=50)
axes[1].set_title('Fréchet Method for Detection Change Point in EEG Dataset with Log-Euclidean-Cholesky Metrics', fontsize=11)
axes[1].set_xlabel('Index', fontsize=11)
axes[1].set_ylabel('Test Statistic', fontsize=11)
axes[1].tick_params(labelsize=11)

plt.tight_layout()
plt.show()
