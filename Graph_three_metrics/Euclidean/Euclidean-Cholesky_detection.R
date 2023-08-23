library(gSeg)
library(ade4)
library(tidyverse)
library(dplyr)
library(ggplot2)

# Set the base path for reading data files for MST
base_path_input_mst <- "E:/Project/distance_matrix/Euclidean_Cholesky"
# Set the base path for reading data files for NNL
base_path_input_nnl <- "E:/Project/distance_matrix/Euclidean_Cholesky_nnl"
# Set the base path for writing error files
base_path_output <- "E:/Project/Result/Euclidean_Cholesky_ori"

# Set the number of simulations
num_simulations <- 49

# Define the values of k
k_values <- c(1, 3, 5, 7, 9)

# Read the true value (assuming you have a single true value)
true_values <- list("case_0_eigen_0.1_1" = 100, 
                    "case_0_eigen_1_5" = 100, 
                    "case_0_eigen_5_10" = 100, 
                    "case_1_eigen_0.1_1" = 150, 
                    "case_1_eigen_1_5" = 150, 
                    "case_1_eigen_5_10" = 150, 
                    "case_2_eigen_0.1_1" = 200, 
                    "case_2_eigen_1_5" = 200, 
                    "case_2_eigen_5_10" = 200)

# List all subdirectories (from MST path as they should be similar for both)
subdirectories <- list.dirs(path = base_path_input_mst, recursive = FALSE)

for (subdirectory in subdirectories) {
  # Extract the subdirectory name
  subdirectory_name <- basename(subdirectory)
  true_value <- true_values[[subdirectory_name]]
  # Create a corresponding subdirectory in the output base path
  subdirectory_path_output <- file.path(base_path_output, subdirectory_name)
  dir.create(subdirectory_path_output, showWarnings = FALSE)
  
  results_mst <- data.frame()
  results_nnl <- data.frame()
  error <- data.frame()
  
  # Loop through each simulation
  for (simulation in 0:num_simulations) {
    # Read the distance matrix from file for MST
    file_path_mst <- paste0(base_path_input_mst, "/", subdirectory_name, "/distance_matrix_", simulation, ".csv")
    data_mst <- read.csv(file_path_mst, header = FALSE)
    
    # Read the distance matrix from file for NNL
    file_path_nnl <- paste0(base_path_input_nnl, "/", subdirectory_name, "/distance_matrix_", simulation, ".csv")
    data_nnl <- read.csv(file_path_nnl, header = FALSE)
    
    # Loop through each value of k
    for (k in k_values) {
      # Construct the MST
      distance_matrix_mst <- as.dist(data_mst)
      mst <- mstree(distance_matrix_mst, k)
      
      # Find the change point based on the correlation matrix
      n1 <- length(data_mst)
      r <- gseg1(n1, mst, statistics = "all")
      
      # Get the predicted values from MST
      predict_ori_r_mst <- r$scanZ$ori$tauhat
      predict_weighted_r_mst <- r$scanZ$weighted$tauhat
      predict_maxtype_r_mst <- r$scanZ$max.type$tauhat
      predict_generalized_r_mst <- r$scanZ$generalized$tauhat
      
      # Calculate the MAE for each category from MST
      mae_ori_mst <- mean(abs(true_value - predict_ori_r_mst))
      mae_weighted_mst <- mean(abs(true_value - predict_weighted_r_mst)) 
      mae_maxtype_mst <- mean(abs(true_value - predict_maxtype_r_mst))
      mae_generalized_mst <- mean(abs(true_value - predict_generalized_r_mst))
      
      distance_matrix_nnl <- as.dist(data_nnl)
      nng <- nnl(distance_matrix_nnl, k)
      
      # Find the change point based on the correlation matrix using NNL
      n2 <- length(data_nnl)
      r_nnl <- gseg1(n2, nng, statistics = "all")
      
      # Get the predicted values from NNL
      predict_ori_r_nnl <- r_nnl$scanZ$ori$tauhat
      predict_weighted_r_nnl <- r_nnl$scanZ$weighted$tauhat
      predict_maxtype_r_nnl <- r_nnl$scanZ$max.type$tauhat
      predict_generalized_r_nnl <- r_nnl$scanZ$generalized$tauhat
      
      # Calculate the MAE for each category from NNL
      mae_ori_nnl <- mean(abs(true_value - predict_ori_r_nnl))
      mae_weighted_nnl <- mean(abs(true_value - predict_weighted_r_nnl))
      mae_maxtype_nnl <- mean(abs(true_value - predict_maxtype_r_nnl))
      mae_generalized_nnl <- mean(abs(true_value - predict_generalized_r_nnl))
      
      # Create data frames to store the results for the current simulation and k value
      result_mst <- data.frame(
        simulation = simulation,
        k = k,
        predict_ori_r = mae_ori_mst,
        predict_weighted_r = mae_weighted_mst,
        predict_maxtype_r = mae_maxtype_mst,
        predict_generalized_r = mae_generalized_mst,
        source = "MST"
      )
      
      result_nnl <- data.frame(
        simulation = simulation,
        k = k,
        predict_ori_r = mae_ori_nnl,
        predict_weighted_r = mae_weighted_nnl,
        predict_maxtype_r = mae_maxtype_nnl,
        predict_generalized_r = mae_generalized_nnl,
        source = "NNL"
      )
      
      # Add the results to the overall data frames
      results_mst <- rbind(results_mst, result_mst)
      results_nnl <- rbind(results_nnl, result_nnl)
      error <- rbind(error, result_mst, result_nnl)
      
    }  # End of k loop
    
    error_file_path <- paste0(subdirectory_path_output, "/error.csv")
    write.csv(error, error_file_path, row.names = FALSE)
  }
}