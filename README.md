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
* ``loading_utils.py`` data preprocessing and modeling pipeline. It includes: 
  + **I/O Utilities:**
    - ``_load_data_in_chunks`` Loads compressed .dat files containing weather and system variables.
    - ``_load_spatial_masks`` loads spatial masks (e.g., land use or resource density) for feature filtering or selection.
    - ``_generate_dataset`` saves all preprocessed data into a single pickle file.
    - ``_load_processed_dataset`` loads the saved preprocessed data pickle file for training.
  + **Feature Engineering:**
    - ``_extrapolate_wind`` wind speeds extrapolation at multiple altitudes (60, 100, 120m) from measured values at 10m and 80m.
    - ``_periodic`` encodes periodic time features (month, hour, etc.) using a cosine transform.
  + **Dataset Structuring:**
    - ``_multisource_structure_dataset`` combines selected data streams (e.g., demand, solar, wind, forecasts) into unified 3D tensors for modeling.
  + **Model Input Formatting:**
    - ``_sparse_learning_dataset_format`` flattens 3D tensor data into 2D feature/target arrays for sparse models.
    - ``_dense_learning_dataset`` builds the 3D arrays for kernel models.
  + **Standardization:**
    - ``_dense_learning_stand`` and ``_sparse_learning_stand`` to normalize feature and target variables.
  + **Train/Test Splits:**
    - ``_training_and_testing_dataset`` to split structured data into training and test sets.


* ``GP_utils.py`` contains the functions of the different implemented versions of Gaussian Process Regression (GPR) models. It includes:
  + Single-task GP using ``GPyTorch`` and ``Scikit-learn``.
  + Multitask GP (MT-GPR) using ``GPyTorch`` with support for various kernels and recursive prediction.
  + Cool-MTGP variants, including hierarchical and approximate models.
  + Custom prediction functions for both standard and multitask GP settings.
* ``scoring_utils.py`` contains a set of metrics for evaluating deterministic and probabilistic forecasts. It includes:
  + Classical deterministic error metrics (RMSE, MAE, MBE).
  + Probabilistic scores (CRPS, LogS, Energy Score, Variogram Score) --- Multivariate aggregation and breakdown by tasks/zones.
  + Interval-based scores (Interval Score, Confidence Interval Coverage).
* ``aux_utils.py`` contains the functions necessary to parallelize the code across nodes in a High-Performance Computer (HPC) and the saving functions to compile and store the dataframes with the results.

### SLURM files

Scripts in a bash to submit jobs to a high-performance computing (HPC) cluster that uses the SLURM workload manager.
* ``drive.sh`` loops over hyperparameter configuration and submits the ``.job`` file.
* ``run.job`` submits jobs to the batch queue POD HPC parallelized with ``mpi4py``.
* ``run_braid.job`` submit jobs to the batch queue in Braid2 HPC parallelized with ``mpi4py``.
* ``run_largemem.job``submits jobs to CPUs in the large largemem queue in POD HPC parallelized with ``mpi4py``.

## Reference

The manuscript is currently undergoing revisions in Nature Communications. The draft is publicly available (https://www.researchsquare.com/article/rs-5891000/v1). We recommend using the following reference:

Terrén-Serrano, Guillermo, Ranjit Deshmukh, and Manel Martínez-Ramón. "Joint Probabilistic Day-Ahead Energy Forecast for Power System Operations." (2025).

