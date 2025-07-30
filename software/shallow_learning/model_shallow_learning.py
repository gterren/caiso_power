import sys, time, ast

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from scipy.stats import norm, multivariate_normal

from utils import *
from scoring_utils import *
from aux_utils import *
from loading_utils import *

# Degine path to data
path_to_prc = r"/home/gterren/caiso_power/data/processed/"
path_to_raw = r"/home/gterren/caiso_power/data/dataset-2023/"
path_to_aux = r"/home/gterren/caiso_power/data/auxiliary/"
path_to_rst = r"/home/gterren/caiso_power/results/"
path_to_mdl = r"/home/gterren/caiso_power/models/journal_w_sigma/"

# Get Grid Dimensions
N = 104
M = 88

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

# Define identification experiment key
# HPC paralellization parameters
i_resource = int(sys.argv[1])
i_exp      = int(sys.argv[2])
i_job      = 0
N_jobs     = 1
i_batch    = 0
N_batches  = 1

# Get combination of hyperparameters in the results validation .csv
# (see manuscript Section Hyperparameters)
filename = "prob_model_selection_{}.csv".format(resources_[i_resource])
exps_    = pd.read_csv(path_to_mdl + filename)
idx_exp_ = exps_.index.values
i_exps_  = _experiments_index_batch_job(idx_exp_, i_batch, N_batches, i_job, N_jobs)
# Get experiment hyperparameters for model testing
i_exp   = i_exps_[i_exp]
theta_  = ast.literal_eval(exps_.loc[i_exp, "parameters"])
thetas_ = [theta_, theta_, theta_]
print(theta_)
# Model section criterion used in validation (ES, VS or IS)
score = exps_.loc[i_exp, "score"]
# Sparse Learning (SL) method
sl_method = exps_.loc[i_exp, "sparse_method"]
# Dense Learning (DL) method
dl_method = exps_.loc[i_exp, "dense_method"]
# SL index
SL = sl_methods_.index(sl_method)
# DL index
DL = dl_methods_.index(dl_method)
print(sl_method, dl_method, SL, DL)

# Generate I/O files names for a given experiment
resources    = exps_.loc[i_exp, "resource"].rsplit('_')
i_resources_ = [resources_.index(resource) for resource in resources]
resource     = '_'.join([resources_[i_resource] for i_resource in i_resources_])
dataset      = '_'.join(['{}-{}'.format(resources_[i_resource], '-'.join(map(str, i_assets_[i_resource]))) for i_resource in i_resources_]) + '_M{}.pkl'.format(i_mask)

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

# Experiment key
key = r'{}-{}_{}{}-{}{}_{}'.format(sl_method, dl_method, x_sl_stnd, y_sl_stnd, x_dl_stnd, y_dl_stnd, theta_)

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
#print(X_sl_tr_.shape, Y_sl_tr_.shape, X_sl_ts_.shape, Y_sl_ts_.shape)

# Clean unused variables from RAM memory
del X_sl_, Y_sl_

# Split DL dataset in training and testing
# (see manuscript Section Validation, Training, and Testing)
X_dl_tr_, X_dl_ts_ = _training_and_testing_dataset(X_dl_)
Y_dl_tr_, Y_dl_ts_ = _training_and_testing_dataset(Y_dl_)
#print(X_dl_tr_.shape, Y_dl_tr_.shape, X_dl_ts_.shape, Y_dl_ts_.shape)

# Clean unused variables from RAM memory
del X_dl_, Y_dl_

meta_ts_ = pd.DataFrame(ZZ_[0, -Y_dl_ts_.shape[0]:, [0, 1, 2, 3]].T, columns = ['year', 'month', 'day', 'yearday'])

# Clean unused variables from RAM memory
del Z_, ZZ_

# Naive and CAISO forecasts as baselines
# (see manuscript Section AI-Based Probabilistic Models Enhance the Performance of a Day-Ahead Energy Forecast)
Y_per_fc_, Y_ca_fc_, Y_clm_fc_ = _naive_forecasts(Y_ac_, Y_fc_, N_lags)

# Clean unused variables from RAM memory
del Y_ac_, Y_fc_

# Split baseline data in training and testing
# (see manuscript Section Validation, Training, and Testing)
Y_per_fc_tr_, Y_per_fc_ts_ = _training_and_testing_dataset(Y_per_fc_)
Y_ca_fc_tr_, Y_ca_fc_ts_   = _training_and_testing_dataset(Y_ca_fc_)
Y_clm_fc_tr_, Y_clm_fc_ts_ = _training_and_testing_dataset(Y_clm_fc_)
#print(Y_per_fc_tr_.shape, Y_per_fc_ts_.shape)
#print(Y_ca_fc_tr_.shape, Y_ca_fc_ts_.shape)
#print(Y_clm_fc_tr_.shape, Y_clm_fc_ts_.shape)

# Clean unused variables from RAM memory
del Y_per_fc_, Y_ca_fc_, Y_clm_fc_

# Generate sparse learning training and testing dataset in the correct format
# (see manuscript Section Feature Vectors for Sparse Learning)
X_sl_tr_, Y_sl_tr_ = _sparse_learning_dataset_format(X_sl_tr_, Y_sl_tr_)
X_sl_ts_, Y_sl_ts_ = _sparse_learning_dataset_format(X_sl_ts_, Y_sl_ts_)
#print(X_sl_tr_test_.shape, Y_sl_tr_test_.shape, X_sl_ts_test_.shape, Y_sl_ts_test_.shape)

# Standardize SL dataset
# (see manuscript section Data Preprocessing)
X_sl_tr_stnd_, Y_sl_tr_stnd_, X_sl_ts_stnd_, sl_scaler_ = _sparse_learning_stand(X_sl_tr_, Y_sl_tr_, X_sl_ts_, x_sl_stnd, y_sl_stnd)
#print(X_sl_tr_stnd_.shape, Y_sl_tr_stnd_.shape, X_sl_ts_stnd_.shape)

# Training SL model
# (see manuscript Section Sparse learning)
t_sl_tr              = time.time()
W_hat_, Y_sl_ts_hat_ = _fit_sparse_learning(X_sl_tr_stnd_, X_sl_ts_stnd_, Y_sl_tr_stnd_, Y_sl_ts_, g_sl_, thetas_, sl_scaler_, SL, y_sl_stnd)
# SL traning time
t_sl_tr = time.time() - t_sl_tr
#print(W_hat_.shape, Y_sl_ts_hat_.shape)

# Standardize DL dataset
# (see manuscript Section Data Preprocessing)
X_dl_tr_stnd_, Y_dl_tr_stnd_, X_dl_ts_stnd_, dl_scaler_ = _dense_learning_stand(X_dl_tr_, Y_dl_tr_, X_dl_ts_, x_dl_stnd, y_dl_stnd)
#print(X_dl_tr_stnd_.shape, Y_dl_tr_stnd_.shape, X_dl_ts_stnd_.shape)

# Traning DL recursively with a model chain
# (see manuscript Section Bayesian Learning)
t_dl_tr = time.time()
if DL != 3:
    # see manuscript Sections Bayesian Linear Regression (BLR),
    # Relevance Vector Machine (RVM), and Gaussian Process for Regression (GPR)
    # see manuscript Sections Model Chain
    models_ = _fit_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_, W_hat_, g_dl_, thetas_, RC, DL, key)
else:
    # see manuscript Section Multi-Task Gaussian Process for Regression (MTGPR)
    # see manuscript Sections Model Chain
    models_ = _fit_multitask_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_, W_hat_, g_dl_, thetas_, RC, DL, key)
# DL training time
t_dl_tr = time.time() - t_dl_tr

# Independent/joint predictive distributions
t_ts = time.time()
if DL != 3:
    # (see manuscript Sections Bayesian Linear Regression (BLR),
    # Relevance Vector Machine (RVM), and Gaussian Process for Regression (GPR))
    # (see manuscript Sections Model Chain)
    M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_ = _pred_prob_dist(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd)
else:
    # (see manuscript Section Multi-Task Gaussian Process for Regression (MTGPR))
    # (see manuscript Section Model Chain)
    M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_ = _multitask_pred_prob_dist(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd)
# DL testing time 
t_ts = time.time() - t_ts
#print(M_dl_ts_hat_.shape, S2_dl_ts_hat_.shape, C_dl_ts_hat_.shape)

# Independent/joint predictive scenarios
# (see manuscript Figure 7a)
t_prob_ts = time.time()
if DL != 3:
    # Draw independet predictive scenarios from the predictive distribution of a BLR, RVM and GPR
    Y_dl_ts_hat_ = _joint_prob_prediction(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd, N_samples = 100)
else:
    # Draw joint predictive scenarios from the predictive distribution of a MTGPR
    Y_dl_ts_hat_ = _multitask_joint_prediction(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd, N_samples = 100)
# Drawing scenarios time
t_prob_ts = time.time() - t_prob_ts

# Save timing results
time_ = pd.DataFrame([t_sl_tr, t_dl_tr, t_ts, t_prob_ts], columns = ['time'],
                                                          index   = ['sparse_training',
                                                                     'dense_training',
                                                                     'testing',
                                                                     'prob_testing'])
print(time_)

# Evaluate baseline forecast with deterministic error metrics
# (see SI section Scoring Rules)
# Aggregated across samples
E_per_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_per_fc_ts_)
E_ca_ts_  = _baseline_det_metrics(Y_dl_ts_, Y_ca_fc_ts_)
E_clm_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_clm_fc_ts_)
# For each sample
E_per_ts_all_ = _baseline_det_metrics_dist(Y_dl_ts_, Y_per_fc_ts_, 'persistence')
E_ca_ts_all_  = _baseline_det_metrics_dist(Y_dl_ts_, Y_ca_fc_ts_, 'caiso')
E_clm_ts_all_ = _baseline_det_metrics_dist(Y_dl_ts_, Y_clm_fc_ts_, 'climatology')

# Save model and outputs
_model = {}
_model['time']           = time_
_model['mask']           = M_[i_mask]
_model['weights']        = W_hat_
_model['feature_labels'] = g_sl_
_model['targets']        = Y_dl_ts_
_model['targets_meta']   = meta_ts_

# Evaluate ML forecast with deterministic error metrics
# and multivaritae proper scoring rules
# (see SI section Scoring Rules)
# Aggregated across samples
_model['bayesian_scoring']      = _multivariate_prob_metrics(Y_dl_ts_, M_dl_ts_hat_, C_dl_ts_hat_, S2_dl_ts_hat_, Y_dl_ts_hat_)
_model['deterministic_scoring'] = _baseline_det_metrics(Y_dl_ts_, M_dl_ts_hat_)
_model['baseline_scoring']      = pd.concat([E_per_ts_, E_ca_ts_, E_clm_ts_], axis = 0).reset_index(drop = True)
# For each sample
_model['bayesian_scoring_all']      = _multivariate_prob_metrics_dist(Y_dl_ts_, M_dl_ts_hat_, C_dl_ts_hat_, S2_dl_ts_hat_, Y_dl_ts_hat_)
_model['deterministic_scoring_all'] = _baseline_det_metrics_dist(Y_dl_ts_, M_dl_ts_hat_, 'ml')
_model['baseline_scoring_all']      = pd.concat([E_per_ts_all_, E_ca_ts_all_, E_clm_ts_all_], axis = 0).reset_index(drop = True)

# Save ML predictive distribution and scenarios
_model['mean']       = M_dl_ts_hat_
_model['covariance'] = C_dl_ts_hat_
_model['variance']   = S2_dl_ts_hat_
_model['samples']    = Y_dl_ts_hat_

# Save baseline forecasts
_model['climatology'] = Y_clm_fc_ts_
_model['caiso']       = Y_ca_fc_ts_
_model['persitence']  = Y_per_fc_ts_

_save_dict(_model, path_to_mdl, file_name = '{}-{}-{}-{}.pkl'.format(resource, sl_method, dl_method, score))

