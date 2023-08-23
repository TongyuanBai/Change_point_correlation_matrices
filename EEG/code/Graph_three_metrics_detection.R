library(gSeg)
library(ade4)
library(tidyverse)
library(dplyr)
library(ggplot2)

results_df <- data.frame(
  k = numeric(),
  predict_ori_r_mst = numeric(),
  predict_weighted_r_mst = numeric(),
  predict_maxtype_r_mst = numeric(),
  predict_generalized_r_mst = numeric(),
  predict_ori_r_nnl = numeric(),
  predict_weighted_r_nnl = numeric(),
  predict_maxtype_r_nnl = numeric(),
  predict_generalized_r_nnl = numeric()
)

setwd("E:/Project/EEG/Euclidean")
data_euclidean <- read.csv("distance_matrix_euclidean.csv", header = FALSE)

# Define the values of k
k_values <- c(1, 3, 5, 7, 9)

# Loop through each value of k
for (k in k_values) {
  # Construct the MST
  distance_matrix <- as.dist(data_euclidean)
  mst <- mstree(distance_matrix, k)
  
  # Find the change point based on the correlation matrix
  n <- length(data_euclidean)
  r <- gseg1(n, mst, statistics = "all")
  
  # Get the predicted values from MST
  predict_ori_r_mst <- r$scanZ$ori$tauhat
  predict_weighted_r_mst <- r$scanZ$weighted$tauhat
  predict_maxtype_r_mst <- r$scanZ$max.type$tauhat
  predict_generalized_r_mst <- r$scanZ$generalized$tauhat
  
  # Get the statistics values from MST
  ori_stat_mst <- r$scanZ$ori$Z
  weighted_stat_mst <- r$scanZ$weighted$Zw
  maxtype_stat_mst <- r$scanZ$max.type$M
  generalized_stat_mst <- r$scanZ$generalized$S
  
  # Save each MST Z vector to its respective CSV
  write.csv(data.frame(ori_stat_mst), paste("E:/Project/EEG/Euclidean/ori_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(weighted_stat_mst), paste("E:/Project/EEG/Euclidean/weighted_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(maxtype_stat_mst), paste("E:/Project/EEG/Euclidean/maxtype_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(generalized_stat_mst), paste("E:/Project/EEG/Euclidean/generalized_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  
  # Construct the NNL
  nng <- nnl(distance_matrix, k)
  
  # Find the change point based on the correlation matrix using NNL
  r_nnl <- gseg1(n, nng, statistics = "all")
  
  # Get the predicted values from MST
  predict_ori_r_nnl <- r_nnl$scanZ$ori$tauhat
  predict_weighted_r_nnl <- r_nnl$scanZ$weighted$tauhat
  predict_maxtype_r_nnl <- r_nnl$scanZ$max.type$tauhat
  predict_generalized_r_nnl <- r_nnl$scanZ$generalized$tauhat
  
  # Get the statistics values from NNL
  ori_stat_nnl <- r_nnl$scanZ$ori$Z
  weighted_stat_nnl <- r_nnl$scanZ$weighted$Zw
  maxtype_stat_nnl <- r_nnl$scanZ$max.type$M
  generalized_stat_nnl <- r_nnl$scanZ$generalized$S
  
  # Save each NNL Z vector to its respective CSV
  write.csv(data.frame(ori_stat_nnl), paste("E:/Project/EEG/Euclidean/ori_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(weighted_stat_nnl), paste("E:/Project/EEG/Euclidean/weighted_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(maxtype_stat_nnl), paste("E:/Project/EEG/Euclidean/maxtype_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(generalized_stat_nnl), paste("E:/Project/EEG/Euclidean/generalized_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  
  # Save predictions to results_df
  results_df <- rbind(results_df, data.frame(
    k = k,
    predict_ori_r_mst = predict_ori_r_mst,
    predict_weighted_r_mst = predict_weighted_r_mst,
    predict_maxtype_r_mst = predict_maxtype_r_mst,
    predict_generalized_r_mst = predict_generalized_r_mst,
    predict_ori_r_nnl = predict_ori_r_nnl,
    predict_weighted_r_nnl = predict_weighted_r_nnl,
    predict_maxtype_r_nnl = predict_maxtype_r_nnl,
    predict_generalized_r_nnl = predict_generalized_r_nnl
  ))
}

# Save the overall results
write.csv(results_df, "E:/Project/EEG/Euclidean/all_results_euclidean.csv", row.names = FALSE)

results_df <- data.frame(
  k = numeric(),
  predict_ori_r_mst = numeric(),
  predict_weighted_r_mst = numeric(),
  predict_maxtype_r_mst = numeric(),
  predict_generalized_r_mst = numeric(),
  predict_ori_r_nnl = numeric(),
  predict_weighted_r_nnl = numeric(),
  predict_maxtype_r_nnl = numeric(),
  predict_generalized_r_nnl = numeric()
)

setwd("E:/Project/EEG/Log_Euclidean")
data_log_euclidean <- read.csv("distance_matrix_log_euclidean.csv", header = FALSE)

# Define the values of k
k_values <- c(1, 3, 5, 7, 9)

# Loop through each value of k
for (k in k_values) {
  # Construct the MST
  distance_matrix <- as.dist(data_log_euclidean)
  mst <- mstree(distance_matrix, k)
  
  # Find the change point based on the correlation matrix
  n <- length(data_log_euclidean)
  r <- gseg1(n, mst, statistics = "all")
  
  # Get the predicted values from MST
  predict_ori_r_mst <- r$scanZ$ori$tauhat
  predict_weighted_r_mst <- r$scanZ$weighted$tauhat
  predict_maxtype_r_mst <- r$scanZ$max.type$tauhat
  predict_generalized_r_mst <- r$scanZ$generalized$tauhat
  
  # Get the statistics values from MST
  ori_stat_mst <- r$scanZ$ori$Z
  weighted_stat_mst <- r$scanZ$weighted$Zw
  maxtype_stat_mst <- r$scanZ$max.type$M
  generalized_stat_mst <- r$scanZ$generalized$S
  
  # Save each MST Z vector to its respective CSV
  write.csv(data.frame(ori_stat_mst), paste("E:/Project/EEG/Log_Euclidean/ori_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(weighted_stat_mst), paste("E:/Project/EEG/Log_Euclidean/weighted_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(maxtype_stat_mst), paste("E:/Project/EEG/Log_Euclidean/maxtype_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(generalized_stat_mst), paste("E:/Project/EEG/Log_Euclidean/generalized_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  
  # Construct the NNL
  nng <- nnl(distance_matrix, k)
  
  # Find the change point based on the correlation matrix using NNL
  r_nnl <- gseg1(n, nng, statistics = "all")
  
  # Get the predicted values from MST
  predict_ori_r_nnl <- r_nnl$scanZ$ori$tauhat
  predict_weighted_r_nnl <- r_nnl$scanZ$weighted$tauhat
  predict_maxtype_r_nnl <- r_nnl$scanZ$max.type$tauhat
  predict_generalized_r_nnl <- r_nnl$scanZ$generalized$tauhat
  
  # Get the statistics values from NNL
  ori_stat_nnl <- r_nnl$scanZ$ori$Z
  weighted_stat_nnl <- r_nnl$scanZ$weighted$Zw
  maxtype_stat_nnl <- r_nnl$scanZ$max.type$M
  generalized_stat_nnl <- r_nnl$scanZ$generalized$S
  
  # Save each NNL Z vector to its respective CSV
  write.csv(data.frame(ori_stat_nnl), paste("E:/Project/EEG/Log_Euclidean/ori_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(weighted_stat_nnl), paste("E:/Project/EEG/Log_Euclidean/weighted_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(maxtype_stat_nnl), paste("E:/Project/EEG/Log_Euclidean/maxtype_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(generalized_stat_nnl), paste("E:/Project/EEG/Log_Euclidean/generalized_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  
  # Save predictions to results_df
  results_df <- rbind(results_df, data.frame(
    k = k,
    predict_ori_r_mst = predict_ori_r_mst,
    predict_weighted_r_mst = predict_weighted_r_mst,
    predict_maxtype_r_mst = predict_maxtype_r_mst,
    predict_generalized_r_mst = predict_generalized_r_mst,
    predict_ori_r_nnl = predict_ori_r_nnl,
    predict_weighted_r_nnl = predict_weighted_r_nnl,
    predict_maxtype_r_nnl = predict_maxtype_r_nnl,
    predict_generalized_r_nnl = predict_generalized_r_nnl
  ))
}

# Save the overall results
write.csv(results_df, "E:/Project/EEG/Log_Euclidean/all_results_log_euclidean.csv", row.names = FALSE)

results_df <- data.frame(
  k = numeric(),
  predict_ori_r_mst = numeric(),
  predict_weighted_r_mst = numeric(),
  predict_maxtype_r_mst = numeric(),
  predict_generalized_r_mst = numeric(),
  predict_ori_r_nnl = numeric(),
  predict_weighted_r_nnl = numeric(),
  predict_maxtype_r_nnl = numeric(),
  predict_generalized_r_nnl = numeric()
)

setwd("E:/Project/EEG/Poly_hyperbolic")
data_hyperbolic <- read.csv("distance_matrix_poly_hyperbolic.csv", header = FALSE)

# Define the values of k
k_values <- c(1, 3, 5, 7, 9)

# Loop through each value of k
for (k in k_values) {
  # Construct the MST
  distance_matrix <- as.dist(data_hyperbolic)
  mst <- mstree(distance_matrix, k)
  
  # Find the change point based on the correlation matrix
  n <- length(data_hyperbolic)
  r <- gseg1(n, mst, statistics = "all")
  
  # Get the predicted values from MST
  predict_ori_r_mst <- r$scanZ$ori$tauhat
  predict_weighted_r_mst <- r$scanZ$weighted$tauhat
  predict_maxtype_r_mst <- r$scanZ$max.type$tauhat
  predict_generalized_r_mst <- r$scanZ$generalized$tauhat
  
  # Get the statistics values from MST
  ori_stat_mst <- r$scanZ$ori$Z
  weighted_stat_mst <- r$scanZ$weighted$Zw
  maxtype_stat_mst <- r$scanZ$max.type$M
  generalized_stat_mst <- r$scanZ$generalized$S
  
  # Save each MST Z vector to its respective CSV
  write.csv(data.frame(ori_stat_mst), paste("E:/Project/EEG/Poly_hyperbolic/ori_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(weighted_stat_mst), paste("E:/Project/EEG/Poly_hyperbolic/weighted_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(maxtype_stat_mst), paste("E:/Project/EEG/Poly_hyperbolic/maxtype_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(generalized_stat_mst), paste("E:/Project/EEG/Poly_hyperbolic/generalized_stat_mst_k", k, ".csv", sep=""), row.names = FALSE)
  
  # Construct the NNL
  nng <- nnl(distance_matrix, k)
  
  # Find the change point based on the correlation matrix using NNL
  r_nnl <- gseg1(n, nng, statistics = "all")
  
  # Get the predicted values from MST
  predict_ori_r_nnl <- r_nnl$scanZ$ori$tauhat
  predict_weighted_r_nnl <- r_nnl$scanZ$weighted$tauhat
  predict_maxtype_r_nnl <- r_nnl$scanZ$max.type$tauhat
  predict_generalized_r_nnl <- r_nnl$scanZ$generalized$tauhat
  
  # Get the statistics values from NNL
  ori_stat_nnl <- r_nnl$scanZ$ori$Z
  weighted_stat_nnl <- r_nnl$scanZ$weighted$Zw
  maxtype_stat_nnl <- r_nnl$scanZ$max.type$M
  generalized_stat_nnl <- r_nnl$scanZ$generalized$S
  
  # Save each NNL Z vector to its respective CSV
  write.csv(data.frame(ori_stat_nnl), paste("E:/Project/EEG/Poly_hyperbolic/ori_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(weighted_stat_nnl), paste("E:/Project/EEG/Poly_hyperbolic/weighted_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(maxtype_stat_nnl), paste("E:/Project/EEG/Poly_hyperbolic/maxtype_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  write.csv(data.frame(generalized_stat_nnl), paste("E:/Project/EEG/Poly_hyperbolic/generalized_stat_nnl_k", k, ".csv", sep=""), row.names = FALSE)
  
  # Save predictions to results_df
  results_df <- rbind(results_df, data.frame(
    k = k,
    predict_ori_r_mst = predict_ori_r_mst,
    predict_weighted_r_mst = predict_weighted_r_mst,
    predict_maxtype_r_mst = predict_maxtype_r_mst,
    predict_generalized_r_mst = predict_generalized_r_mst,
    predict_ori_r_nnl = predict_ori_r_nnl,
    predict_weighted_r_nnl = predict_weighted_r_nnl,
    predict_maxtype_r_nnl = predict_maxtype_r_nnl,
    predict_generalized_r_nnl = predict_generalized_r_nnl
  ))
}

# Save the overall results
write.csv(results_df, "E:/Project/EEG/Poly_hyperbolic/all_results_poly_hyperbolic.csv", row.names = FALSE)
