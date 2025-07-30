import sys, time

import numpy as np
import pandas as pd

from utils import *
from scoring_utils import *
from loading_utils import *
from aux_utils import *

# MPI job variables
i_job, N_jobs, comm = _get_node_info()

# Degine path to data
path_to_prc = r"/home/gterren/caiso_power/data/processed/"
path_to_aux = r"/home/gterren/caiso_power/data/auxiliary/"
path_to_rst = r"/home/gterren/caiso_power/results/"

# Get Grid Dimensions
N = 104
M = 88

dl_methods_ = ['BLR', 'RVM', 'GPR', 'MTGPR', 'NGR']
sl_methods_ = ['lasso', 'OMP', 'elastic_net', 'group_lasso', 'dense']
resources_  = ['load', 'solar', 'wind']

# Energy timeseries list
resources_ = ['load', 'solar', 'wind']
# Sparse Learning (SL) methods list
sl_methods_ = ['lasso', 'OMP', 'elastic_net', 'group_lasso', 'dense']
# Dense Learning (DL) methods list
dl_methods_ = ['BLR', 'RVM', 'GPR', 'MTGPR', 'NGR']
# Load/Solar/Wind timeseries assign to each node
# Nodes: [[MWD, PGE, SCE, SDGE, VEA], [NP, SP, ZP], [NP, SP]]
# [[Nothern California], [Southern California], [Central California]]
i_assets_ = [[1, 2, 3], [0, 1, 2], [0, 1]]

# Assets in resources: {load:5, solar:3, wind:2}
R = int(sys.argv[1])
i_resources_ = [R]

# Spatial masks:
# i_mask: All, US Land, All CAISO Resources density, Resource density
# tau: threshold
i_mask = 3
tau    = 0.

# Autoregressive series from today
# (1: included, 0: excluded)
AR = 1
# Cyclo-stationary from lagged series
# (1: included, 0: excluded)
CS = 1
# Time dammy variables
# (1: included, 0: excluded)
TM = 1
# Recursive forecast in covariates
# (1: included, 0: excluded)
RC = 1
# Autoregressive and Cyclo-stationary series lag
N_lags = 6

# Sparse Learning (SL) index
SL = int(sys.argv[2])
# Dense Learning (DL) index
DL = int(sys.argv[3])
# SL method
sl_method = sl_methods_[SL]
# DL method
dl_method = dl_methods_[DL]
print(sl_method, dl_method)

# Generate I/O files names for a given experiment
resource =  '_'.join([resources_[i_resource] for i_resource in i_resources_])
dataset  =  '_'.join(['{}-{}'.format(resources_[i_resource], '-'.join(map(str, i_assets_[i_resource])))
                      for i_resource in i_resources_]) + '_M{}_periodic.pkl'.format(i_mask)
print(resource, dataset)

# Sparse learning model standardization (SL model: 0 standaridation, 1: nostandaridation)
# (see manuscript section Data Preprocessing)
x_sl_stnd, y_sl_stnd = [[[1, 1],[0, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[1, 1],[0, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[1, 1],[1, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]][DL],
                        [0, 0]][SL]
# Dense learning model standardization (DL model: 0 standaridation, 1: nostandaridation)
# (see manuscript section Data Preprocessing)
x_dl_stnd, y_dl_stnd = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]][DL]

# SL and DL hyperparameters combination to validate
# (see manuscript Section hyperparameters)
alphas_ = [5e-4, 1e-4, 5e-3, 1e-3, 1e-2, 5e-2, 1e-1]
nus_    = [1e-4, 1e-3, 1e-2, 1e-1, 1.]
betas_  = [10, 20, 40, 80, 160, 320, 640]
omegas_ = [0.01, 0.25, 0.5, 0.75]
if R == 0: gammas_ = [0.1, 1., 10.]
if R == 1: gammas_ = [1., 10., 100.]
if R == 2: gammas_ = [0.01, 0.1, 1.]
etas_     = [0.25, 0.5, 0.75, 1.]
lambdas_  = [1., 10., 100., 1000.]
xis_      = [0, 4, 5, 6, 7] # ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']
kappas_1_ = [25, 50, 100, 200]
kappas_2_ = [0.1, 0.05]
kappas_3_ = [.4, .5, .6]
kappas_4_ = [.1, .2, .3, .4, .5]
kappas_   = [kappas_1_, kappas_2_, kappas_3_, kappas_4_]

# Get all combination of possible hyperparameters
# Split them in batchs and get experiment index corresponding
# to this job
exp_, N_thetas = _get_cv_param(alphas_, nus_, betas_, omegas_, gammas_, etas_, lambdas_, xis_, kappas_, SL, DL)
i_exps_        = _random_experiments_index_batch_job(exp_, i_job, N_jobs)
print(i_exps_, len(i_exps_), len(exp_), N_thetas, i_job, N_jobs)

# Loading spatial masks
# (see SI Section Spatial Masks)
M_ = _load_spatial_masks(i_resources_, path_to_aux)

# Loading preprocessed (filtered) dataset
# (see manuscript Section Processing and Filtering)
# (see SI Section Data Processing)
X_sl_, Y_sl_, g_sl_, X_dl_, Y_dl_, g_dl_, Z_, ZZ_, Y_ac_, Y_fc_ = _load_processed_dataset(dataset, path_to_prc)

# Split SL dataset data in training and testing
# (see manuscript Section Validation, Training, and Testing)
X_sl_tr_, X_sl_ts_ = _training_and_testing_dataset(X_sl_)
Y_sl_tr_, Y_sl_ts_ = _training_and_testing_dataset(Y_sl_)
# Clean unused variables from RAM memory
del X_sl_, Y_sl_

# Split DL dataset in training and testing
# (see manuscript Section Validation, Training, and Testing)
X_dl_tr_, X_dl_ts_ = _training_and_testing_dataset(X_dl_)
Y_dl_tr_, Y_dl_ts_ = _training_and_testing_dataset(Y_dl_)
# Clean unused variables from RAM memory
del X_dl_, Y_dl_

# Naive and CAISO forecasts as baselines
# (see manuscript Section AI-Based Probabilistic Models
# Enhance the Performance of a Day-Ahead Energy Forecast)
Y_per_fc_, Y_ca_fc_, Y_clm_fc_ = _naive_forecasts(Y_ac_, Y_fc_, N_lags)
# Clean unused variables from RAM memory
del Y_ac_, Y_fc_

# Split baseline data in training and testing
# (see manuscript Section Validation, Training, and Testing)
Y_per_fc_tr_, Y_per_fc_ts_ = _training_and_testing_dataset(Y_per_fc_)
Y_ca_fc_tr_, Y_ca_fc_ts_   = _training_and_testing_dataset(Y_ca_fc_)
Y_clm_fc_tr_, Y_clm_fc_ts_ = _training_and_testing_dataset(Y_clm_fc_)
# Clean unused variables from RAM memory
del Y_per_fc_, Y_ca_fc_, Y_clm_fc_

# Evaluate baseline forecast with deterministic error metrics
# (see SI section Scoring Rules)
E_per_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_per_fc_ts_)
E_ca_ts_  = _baseline_det_metrics(Y_dl_ts_, Y_ca_fc_ts_)
E_clm_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_clm_fc_ts_)
print(E_per_ts_)
print(E_ca_ts_)
print(E_clm_ts_)

# Generate sparse learning training and testing dataset in the correct format
# (see manuscript Section Feature Vectors for Sparse Learning)
X_sl_tr_, Y_sl_tr_ = _sparse_learning_dataset_format(X_sl_tr_, Y_sl_tr_)
X_sl_ts_, Y_sl_ts_ = _sparse_learning_dataset_format(X_sl_ts_, Y_sl_ts_)

# Training models for all hyperparameters in the experiment set
R_dl_ts_ = []
for i_exp in i_exps_:

    # Get hyperparameters combination in the experiment
    theta_  = exp_[i_exp]
    thetas_ = [theta_, theta_, theta_]

    # Experiment key
    key = r'{}-{}_{}{}-{}{}_{}'.format(sl_method, dl_method, x_sl_stnd, y_sl_stnd, x_dl_stnd, y_dl_stnd, theta_)
    print(i_job, i_exp, x_sl_stnd, y_sl_stnd, x_dl_stnd, y_dl_stnd, thetas_)

    # Get Smoothing and Calibration indicator
    smoothing_ts_, calibration_ts_ = _load_dict(path_to_mdl,
                                                file_name = '{}-{}-{}_e{}.pkl'.format(resource, sl_method, dl_method, i_exp))

    # Standardize SL dataset
    # (see manuscript section Data Preprocessing)
    X_sl_tr_stnd_, Y_sl_tr_stnd_, X_sl_ts_stnd_, sl_scaler_ = _sparse_learning_stand(X_sl_tr_,
                                                                                     Y_sl_tr_,
                                                                                     X_sl_ts_,
                                                                                     x_sl_stnd,
                                                                                     y_sl_stnd)

    # Training SL model
    # (see manuscript Section Sparse learning)
    t_sl_tr              = time.time()
    W_hat_, Y_sl_ts_hat_ = _fit_sparse_learning(X_sl_tr_stnd_,
                                                X_sl_ts_stnd_,
                                                Y_sl_tr_stnd_,
                                                Y_sl_ts_,
                                                g_sl_,
                                                thetas_,
                                                sl_scaler_,
                                                SL,
                                                y_sl_stnd)
    # Training time
    t_sl_tr = time.time() - t_sl_tr

    # Standardize DL dataset
    # (see manuscript Section Data Preprocessing)
    X_dl_tr_stnd_, Y_dl_tr_stnd_, X_dl_ts_stnd_, dl_scaler_ = _dense_learning_stand(X_dl_tr_,
                                                                                    Y_dl_tr_,
                                                                                    X_dl_ts_,
                                                                                    x_dl_stnd,
                                                                                    y_dl_stnd)

    # Traning DL recursively with a model chain
    # (see manuscript Section Bayesian Learning)
    t_dl_tr = time.time()
    if DL != 3:
        # see manuscript Sections Bayesian Linear Regression (BLR),
        # Relevance Vector Machine (RVM), and Gaussian Process for Regression (GPR)
        # see manuscript Sections Model Chain
        models_ = _fit_dense_learning(X_dl_tr_stnd_,
                                      Y_dl_tr_stnd_,
                                      Y_dl_tr_,
                                      W_hat_,
                                      g_dl_,
                                      thetas_,
                                      RC,
                                      DL,
                                      key)
    else:
        # see manuscript Section Multi-Task Gaussian Process for Regression (MTGPR)
        # see manuscript Sections Model Chain
        models_ = _fit_multitask_dense_learning(X_dl_tr_stnd_,
                                                Y_dl_tr_stnd_,
                                                Y_dl_tr_,
                                                W_hat_,
                                                g_dl_,
                                                thetas_,
                                                RC,
                                                DL,
                                                key)
    # Training time
    t_dl_tr = time.time() - t_dl_tr

    # Independent/joint predictive distributions
    t_ts = time.time()
    if DL != 3:
        M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_ = _pred_prob_dist(models_,
                                                                    dl_scaler_,
                                                                    X_dl_ts_stnd_,
                                                                    Y_dl_ts_,
                                                                    W_hat_,
                                                                    g_dl_,
                                                                    RC,
                                                                    DL,
                                                                    y_dl_stnd)
    else:
        M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_ = _multitask_pred_prob_dist(models_,
                                                                              dl_scaler_,
                                                                              X_dl_ts_stnd_,
                                                                              Y_dl_ts_,
                                                                              W_hat_,
                                                                              g_dl_,
                                                                              RC,
                                                                              DL,
                                                                              y_dl_stnd)
    # Testing time
    t_ts = time.time() - t_ts

    # Calibrate predictive distribution
    # (see manuscript Section Predictive Density Calibratio)
    S2_dl_ts_hat_, C_dl_ts_hat_ = _calibrate_predictive_covariance(M_dl_ts_hat_,
                                                                   C_dl_ts_hat_,
                                                                   calibration_ts_)

    # Independent/joint predictive scenarios
    # (see manuscript Figure 7a)
    t_prob_ts = time.time()
    if DL != 3:
        Y_dl_ts_hat_ = _joint_prob_prediction(models_,
                                              dl_scaler_,
                                              X_dl_ts_stnd_,
                                              Y_dl_ts_,
                                              W_hat_,
                                              calibration_ts_,
                                              g_dl_,
                                              RC,
                                              DL,
                                              y_dl_stnd,
                                              N_scenarios = 100)
    else:
        Y_dl_ts_hat_ = _multitask_joint_prediction(models_,
                                                   dl_scaler_,
                                                   X_dl_ts_stnd_,
                                                   Y_dl_ts_,
                                                   W_hat_,
                                                   calibration_ts_,
                                                   g_dl_,
                                                   RC,
                                                   DL,
                                                   y_dl_stnd,
                                                   N_scenarios = 100)
    # Scenario drawing time
    t_prob_ts = time.time() - t_prob_ts

    # Smooth scenarios shape
    # (see manuscript Section Scenario Smoothing)
    Y_dl_ts_hat_ = _calibrate_scenarios_temporal_structure(Y_dl_ts_hat_,
                                                           sigmas_ = smoothing_ts_)

    # Evaluate ML forecast with deterministic error metrics
    # (see SI section Scoring Rules)
    E_dl_ts_ = _baseline_det_metrics(Y_dl_ts_, M_dl_ts_hat_)

    # Evaluate univariate proper scoring rules
    # (see SI section Scoring Rules)
    P_dl_ts_ = _prob_metrics(Y_dl_ts_, M_dl_ts_hat_, S2_dl_ts_hat_, Y_dl_ts_hat_)

    # Evaluate ML forecast with multivaritae proper scoring rules
    # (see SI section Scoring Rules)
    if R == 1:
        # When evaluating solar generation forcast remove night hours
        Y_dl_ts_p_      = Y_dl_ts_[..., 4:-4]
        M_dl_ts_hat_p_  = M_dl_ts_hat_[..., 4:-4]
        C_dl_ts_hat_p_  = C_dl_ts_hat_[..., 4:-4]
        S2_dl_ts_hat_p_ = S2_dl_ts_hat_[..., 4:-4]
        Y_dl_ts_hat_p_  = Y_dl_ts_hat_[..., 4:-4, :]
        MV_dl_ts_       = _multivariate_prob_metrics(Y_dl_ts_p_,
                                                     M_dl_ts_hat_p_,
                                                     C_dl_ts_hat_p_,
                                                     S2_dl_ts_hat_p_,
                                                     Y_dl_ts_hat_p_)
    else:
        MV_dl_ts_ = _multivariate_prob_metrics(Y_dl_ts_,
                                               M_dl_ts_hat_,
                                               C_dl_ts_hat_,
                                               S2_dl_ts_hat_,
                                               Y_dl_ts_hat_)

    # Calculate number of features selected by the SL method
    N_feaures_ = [(W_hat_[:, tsk] != 0.).sum() for tsk in range(W_hat_.shape[1])]
    N_feaures  = int(np.sum(N_feaures_))
    #print(N_feaures_,N_feaures )

    # Format experiments meta data in a dataframe
    meta_ = pd.DataFrame([i_exp, SL, x_sl_stnd, y_sl_stnd, DL, x_dl_stnd, y_dl_stnd, theta_, N_feaures_, N_feaures, t_sl_tr, t_dl_tr, t_ts, t_prob_ts],
                         index = ['experiment',
                                  'sparse_method',
                                  'x_sl_std',
                                  'y_sl_std',
                                  'dense_method',
                                  'x_dl_std',
                                  'y_dl_std',
                                  'parameters',
                                  'all_dimensions',
                                  'dimensions',
                                  'sparse_training_time',
                                  'dense_training_time',
                                  'testing_time',
                                  'prob_testing_time']).T

    R_dl_ts_.append(pd.concat([meta_,
                               _flatten_DataFrame(P_dl_ts_),
                               _flatten_DataFrame(E_dl_ts_),
                               _flatten_DataFrame(MV_dl_ts_)], axis = 1))

# Combine the results from all experiment batches
# running in parallel using mpi4py
_combine_parallel_results(comm,
                          pd.concat(R_dl_ts_, axis = 0),
                          i_job,
                          N_jobs,
                          path_to_rst,
                          file_name = 'test-{}-{}-{}.csv'.format(resource, sl_method, dl_method))
