# -*- coding: utf-8 -*-
"""
Generate the dataset for NNG 
"""

import os
import pandas as pd

extraction_path = "E:/Project/"
new_extraction_path = "E:/Project/dataset_nnl/"

def remove_duplicates_from_file(file_path, new_folder_path):
    # Load the data
    data = pd.read_csv(file_path, header=None)
    
    # Drop duplicate rows
    data = data.drop_duplicates()
    
    # Ensure the new directory exists
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    
    # Save the data to the new directory with the same file name
    new_file_path = os.path.join(new_folder_path, os.path.basename(file_path))
    data.to_csv(new_file_path, index=False, header=False)

# Process all the data files
subfolders = os.listdir(os.path.join(extraction_path, 'dataset'))
for subfolder in subfolders:
    subfolder_path = os.path.join(extraction_path, 'dataset', subfolder)
    new_subfolder_path = os.path.join(new_extraction_path, subfolder)
    for data_file in os.listdir(subfolder_path):
        data_file_path = os.path.join(subfolder_path, data_file)
        remove_duplicates_from_file(data_file_path, new_subfolder_path)
