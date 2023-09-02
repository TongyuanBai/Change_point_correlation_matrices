# Change Point for Correlation Matrices
## Introduction 
This project proposed two change point detection methods for correlation matrices with three different metrics. The aim of project is to find the accurate location of change point.
## Software
- Python 3.8
- R Studio 4.2.0

## Documentation
This part I will introduce the function of each code file. This repository includes five folder. 

### Generate_Dataset
This folder includes two code files. The name of file covariance_time_series.py is used to generate the time series data. It generate nine different cases which the combination of location of change point and the eigenvalue are all different. Another file called generate_for_nnl_graph_method.py is to construct NNG graph for Graph-based change point detection method.

### Fréchet_two_metrics
This folder includes two folders, Euclidean and Log_Euclidean. In each subfolder, it include one code file that describe how to apply Fréchet-based method with Euclidean-Cholesky metrics or Log-Euclidean-Cholesky metrics. In both code file, we defined the function of calculating Fréchet moments with two different metrics. As for result, we save the location of change point and the results of MAE values. 

### Graph_three_metrics
This folder includes three folders, Poly_Hyperbolic, Euclidean and Log_Euclidean. Each subfolder includes two code files. For example, the name of fill called Euclidean-Cholesky.py is to calculate the distance matrix among all correlation matrices. Then, the name of Euclidean-Cholesky detection.R is to construce the types of graph and find the location of change point. Finally, we save the results of MAE values of the combinations of value of k and the types of graph.

### Plot_graph
This folder includes three code files. The name of file Fréchet_boxplot.py is used to plot boxplots of the results of MAE values of Fréchet-based method with two different metrics that calculated in the name of folder called Fréchet_two_metrics. The name of file boxplot_graph_three_metrics.py is used to plot boxplots of the results of MAE values of Graph-based method with three different metrics that calculated in the name of folder called Graph_three_metrics. The name of file compare_different_methods.py is used to compare the the results of MAE values of two method with the same metric.

### EEG
This folder includes two folders, included code and Graph. In code folder, the name of file Fréchet_Euclidean_Cholesky.py is to detect the location of change point by Fréchet-based method with Euclidean-Cholesky metrics. The name of file Fréchet_Log_Euclidean_Cholesky.py is to detect the location of change point by Fréchet-based method with Log-Euclidean-Cholesky metrics. The name of file Graph_three_metrics.py is to construct the distance matrix with three metrics. The name of file Graph_three_metrics_detection.R is to detect the location of change point by Graph-based method with three different metrics. In Graph folder, the name of file Heatmap_correlation_matrices.py is to plot the heatmap of two correrlation matrices come from before and after change point. The name of file Merged_Data.py is to process the data and merge all results together which is very convenient to plot the scatter plots in the next step. The name of file Scatter_Plots_Two_methods.py is to plot scatter plots for two methods with different metrics that can know the location of change point from another view. 

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
