# -*- coding: utf-8 -*-
"""
Merge results
"""

import pandas as pd
import os

def merge_files_from_folder(folder_path: str, output_file_name: str) -> str:
    # List all the files in the directory
    file_list = os.listdir(folder_path)

    # Initialize an empty DataFrame to store the merged data
    merged_data = pd.DataFrame()

    # Loop through each file and add its contents as a new column to the merged_data DataFrame
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        column_name = os.path.splitext(file)[0]  # Using the filename without extension as the column name
        
        # Read the file into a DataFrame, skipping the first row
        data = pd.read_csv(file_path, header=None, skiprows=1, squeeze=True)
        
        # Add the data as a new column to the merged_data DataFrame
        merged_data[column_name] = data

    # Save the merged data to a CSV file
    merged_data.to_csv(output_file_name, index=False)
    
    return output_file_name

# Usage example:
merge_files_from_folder("E:/Project/EEG/log_euc", "E:/Project/EEG/Log_Euclidean/combined_log_euc_data.csv")
merge_files_from_folder("E:/Project/EEG/poly", "E:/Project/EEG/Poly_hyperbolic/combined_poly_data.csv")
merge_files_from_folder("E:/Project/EEG/euc", "E:/Project/EEG/Euclidean/combined_euc.csv")
merge_files_from_folder("E:/Project/EEG/Frechet", "E:/Project/EEG/Frechet/combined_frechet.csv")
