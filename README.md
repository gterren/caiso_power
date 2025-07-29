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


To install **Cool MT-GPR** ``git get https://github.com/OGHinde/Cool_MTGP.git`` from the GitHub repository (``https://github.com/OGHinde/Cool_MTGP'') in the code's main folder ``shallow_learning``. We only need the ``lib`` folder.

## Code description

### Main

* ``val_shallow_learning.py`` cross-validates the hyperparameters in the proposed two-stage Machine Learning (ML) pipeline and saves the validation proper scoring rules for model selection.
* ``test_shallow_learning.py`` tests the proposed two-stage ML pipeline for a given set of hyperparameters, generates the predictive distribution from the model chain for the testing set, draws predictive scenarios, evaluates its performance and the performances of the baselines, and saves the testing-only results.
* ``model_shallow_learning.py`` trains the proposed two-stage ML pipeline for a given set of hyperparameters, generates the predictive distribution from the model chain for the testing set, draws predictive scenarios for each testing day, and saves them.

### Functionalities

* ``utils.py`` this functions with multiple versions of Gaussian Process Regression (GPR) models. It includes:
  + **Sparse Learning Models** feature selection function calls based on realized weather features (Lasso, OMP, Elastic Net, and Group Lasso).
  + **Dense Learning Models** recursive Bayesian chains function calls using the features selected by sparse models and the forecasted weather features (BLR, RVM, GPR, and MT-GPR).
  + **Reboust Recursive Predictive distributions** function calls for a multihorizon forecast recursively using previous predictions as inputs (similar to seq2seq models), ensuring robust predictive covariance matrix inversion.
  
* ``loading_utils.py`` data preprocessing and modeling pipeline. It includes: 
  + **Data Loading** to load compressed ``.dat`` files containing weather and system variables, spatial masks (e.g., land use or resource density) for feature filtering or selection.
  + **Feature Engineering** to extrapolate wind speeds at multiple altitudes (60, 100, 120m) from measured values at 10m and 80m and periodic time encoding (month, hour, etc.) using a cosine transform.
  + **Dataset Structuring** function calls to combine selected data streams (e.g., demand, solar, wind, forecasts) into unified 3D tensors for modeling.
  + **Model Input Formatting** function calls to flatten 3D tensor data into 2D feature/target arrays for sparse models, and build the 3D arrays for kernel models.
  + **Standardization** function calls to normalize features and target variables.
  + **Train/Test Splits** function calls to structure data into training and test sets.
  + **I/O Utilities** function calls to save all preprocessed data into a single pickle file and load the saved preprocessed data pickle file for training.
  
* ``GP_utils.py`` contains the functions of the different implemented versions of Gaussian Process Regression (GPR) models. It includes:
  + **Single-task GPR** function calls using ``GPyTorch`` and ``Scikit-learn``.
  + **Multi-task GPR (MT-GPR)** function calls using ``GPyTorch`` with support for various kernels and recursive prediction.
  + **Cool-MTGPR** variants function calls (including hierarchical and approximate models).
  + **Prediction functions** call for both standard and multitask GP settings.
    
* ``scoring_utils.py`` contains a set of metrics for evaluating deterministic and probabilistic forecasts. It includes:
  + **Classical error metrics** functions (RMSE, MAE, MBE).
  + **Proper scoring rules** functions (CRPS, Energy Score, Variogram Score, Interval Score) --- Multivariate aggregation and breakdown by tasks/zones.
  + **Coverage scores** functions (Confidence Interval Coverage).
    
* ``aux_utils.py`` contains the functions necessary to parallelize the code across nodes in a High-Performance Computer (HPC) and the saving functions to compile and store the dataframes with the results.

### SLURM files

Scripts in a bash to submit jobs to a high-performance computing (HPC) cluster that uses the SLURM workload manager.
* ``drive.sh`` loops over hyperparameter configuration and submits ``.job`` files.
* ``run.job`` submits jobs to POD's batch queue, HPC parallelized with ``mpi4py``.
* ``run_braid.job``submits jobs to Braid2's batch queue, parallelized with ``mpi4py``.
* ``run_largemem.job``submits jobs to nodes in POD's large largemem queue, parallelized with ``mpi4py``.

## Reference

The manuscript is currently undergoing revisions in Nature Communications. The draft is publicly available (https://www.researchsquare.com/article/rs-5891000/v1). We recommend using the following reference:

Terrén-Serrano, Guillermo, Ranjit Deshmukh, and Manel Martínez-Ramón. "Joint Probabilistic Day-Ahead Energy Forecast for Power System Operations." (2025).

