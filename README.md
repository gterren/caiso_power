# Joint Day-Ahead Probabilistic Energy Forecast

A robust, scalable library for probabilistic day-ahead forecasting of electricity demand and renewable (solar & wind) generation in CAISO trading hubs and major utilities — leveraging publicly available weather data and machine learning methods. It also facilitates system-level joint uncertainty analysis, multi-resource node-level joint uncertainty analysis, and operating reserve planning.

### Key Features:

* **Joint probabilistic forecasts:** Simultaneously model electricity demand, solar, and wind generation using joint probability distributions.
* **Sparse + Bayesian learning:** Combines four sparse learning methods for feature selection and four Bayesian learning methods for uncertainty quantification.
* **Model evaluation framework:** Rigorously assesses model combinations via proper scoring rules.
* **CAISO-specific zones:** Supports the three CAISO trading hubs (NP15, SP15, ZP26) and the three major utilities (PG&E, SCE, SDG&E), and produces a ~25% forecasting accuracy improvement.
* **Reserve allocation tool:** Uses forecast confidence intervals to optimize operating reserve levels beyond deterministic forecasts.

## Required packages

``conda install -c anaconda numpy pandas scipy scikit-learn`` 

``conda install scikit-learn-intelex``

``conda install -c conda-forge properscoring blosc mpi4py``

``pip install group-lasso``

## Code description



* ``val_shallow_learning.py''
* ``test_multitask_shallow_learning.py''
* ``test_shallow_learning.py''
* ``scoring_utils.py''
* ``model_shallow_learning.py''

### Functionalities

* ``utils.py`` this functions with multiple versions of Gaussian Process Regression (GPR) models.
* ``GP_utils.py`` contains the functions of the different implemented versions of Gaussian Process Regression (GPR) models.It includes:
  + Single-task GP using GPyTorch and Scikit-learn.
  + Multitask GP (MT-GPR) using GPyTorch with support for various kernels and recursive prediction.
  + Cool-MTGP variants, including hierarchical and approximate models.
  + Custom prediction functions for both standard and multitask GP settings.

* ``scoring_utils.py`` contains a set of metrics for evaluating deterministic and probabilistic forecasts. It includes:
  + Classical deterministic error metrics (RMSE, MAE, MBE).
  + Probabilistic scores (CRPS, LogS, Energy Score, Variogram Score) --- Multivariate aggregation and breakdown by tasks/zones.
  + Interval-based scores (Interval Score, Confidence Interval Coverage).
* ``aux_utils.py`` contains the functions necessary to parallelize the code across nodes in a High-Performance Computer (HPC) and the saving functions to compile and store the dataframes with the results.


## Reference

The manuscript is currently undergoing revisions in Nature Communications. The draft is publicly available (https://www.researchsquare.com/article/rs-5891000/v1). We recommend using the following reference:

Terrén-Serrano, Guillermo, Ranjit Deshmukh, and Manel Martínez-Ramón. "Joint Probabilistic Day-Ahead Energy Forecast for Power System Operations." (2025).

