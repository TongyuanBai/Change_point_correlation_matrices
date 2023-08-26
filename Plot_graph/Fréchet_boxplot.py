# -*- coding: utf-8 -*-
"""
Boxplots for Fréchet method
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to extract data from the given directory
def extract_data(extract_dir):
    # List and sort the contents of the given directory
    folder_content = os.listdir(extract_dir)
    folder_content.sort()

    # Extract MAE values from the CSV files and append to a list
    data_list = []
    for folder in folder_content:
        file_path = os.path.join(extract_dir, folder, "mae_results.csv")
        data = pd.read_csv(file_path)
        data_list.append(data["MAE"].values)
    return data_list

# Function to plot the extracted data
def plot_data_with_borders(data_list, ordered_x_labels, color, legend_label):
    plt.figure(figsize=(6, 4))
    
    log_transformed_data = [np.log(data) for data in data_list]
    
    # Create a boxplot with custom colors and styles
    box_plot = plt.boxplot(log_transformed_data, vert=True, patch_artist=True, 
                           capprops=dict(color='black'),
                           whiskerprops=dict(color='black'),
                           flierprops=dict(color='black', markeredgecolor='black'),
                           medianprops=dict(color='black'),
                           labels=ordered_x_labels)
    for patch in box_plot['boxes']:
        patch.set_facecolor(color)
        
    # Set labels and font sizes
    plt.xticks(rotation=45, fontsize=11)
    plt.ylabel("Log Scale MAE", fontsize=11)
    plt.xlabel("Cases", fontsize=11)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=11)

    # Add a legend to the plot
    plt.legend([box_plot["boxes"][0]], [legend_label], loc='upper right')

    # Display all borders
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    
    plt.show()

# Extract data for both methods
data_list_original = extract_data("E:/Project/Result/Frechet_euclidean_ori")
data_list_new = extract_data("E:/Project/Result/Frechet_log_euclidean_ori")

# Order the extracted data based on the cases
ordered_data_list_original = [data_list_original[i-1] for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
ordered_data_list_new = [data_list_new[i-1] for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
ordered_x_labels = [f"Case{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]]

# Define a common color for the plots
common_color_blue = "#003366"

# Plot the data for both metrices
plot_data_with_borders(ordered_data_list_original, ordered_x_labels, common_color_blue, 'Frechet Euclidean')
plot_data_with_borders(ordered_data_list_new, ordered_x_labels, common_color_blue, 'Frechet Log Euclidean')
