# Recommendation Tool to improve Optimization of Computational Resources

This repository contains some of the methods, techniques and algorithms that have been developed as part of a project at the Information Technology department at National University of Singapore(NUS) that aims to optimize and improve allocation of computational resources to jobs and tasks submitted. This is achieved by perform data analysis and visualisation on historical data based on the logs of past jobs submitted by users of the High Performance Computing(HPC) servers, followed by feature selection and engineering as features for the models to be used. Some of the features that have been included but are not limtied to, are:

- Memory Requested
- CPU requested
- Department of expertise

The scheduling system that is used to allocate resources for computational tasks is the PBS system. Using the extracted and transformed data from the logs, we proceeded to apply machine learning techniques to predict the estimated CPU and memory efficiency as well as allocation for the user using their job script. 

-
Methodology
-

The algorithm that we use to predict the CPU and memory utilization of a user program consists of 5 base models:

- RandomForestRegressor
- GradientBoostingRegressor
- SVR using the RBF kernel (Support Vector Machines)
- CatBoostRegressor
- XGBRegressor

from the Scikit-Learn(sklearn), CatBoost(catboost) and XGBoost(xgb) libraries. These 5 models are trained and tuned using Bayesian Optimization to approximate the best hyperparameters, and then stacked together using a layer 2 regression model using the vecstack library. The Scikit-Optimize library(skopt) was used for hyperparameter tuning using Bayesian Optimization.

The following scripts are auxiliary scripts for data extraction to be used with the recommendation tool:

- ai_cloud_etl: Module that provides operations for data extraction, feature engineering and selection
- ai_cloud_model: Module that contains parameters for loading the overal prediction algorithm, as well as model-related supporting operations
- job_script_process: Subroutine script to process the details of a given PBS job script and return the required information on the job
- queue_extract: Routine script to obtain the default parameters(CPU, memory) of each queue in the cluster. Using to handle missing values and allocate defaults, or provide upper/lower bounding should values be too high/low

-
Upcoming Improvements:
- 
*Queue Recommendation*

The script queue_recommendation.py is currently in progress that measures the load across cluster queues, and aims to recommend a less busy queue to submit the job to if applicable. This is aimed at the balancing of loads across clusters to better optimize the utilization of available resources in a server cluster

*Neural Networks*

Neural networks have been known to do well in modelling the relationship between dependent and indepedent variables as well as complex structures in data. We aim to further research on the use of neural networks as both layer 1 and 2 models in the stacking framework that we have proposed as an improvement to our current prediction algorithm.
