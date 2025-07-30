import sys, time, ast, pickle

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
path_to_mdl = r"/home/gterren/caiso_power/models/journal_paper_w_sigma_n_lambda-Cal-4/"
path_to_prm = r"/home/gterren/caiso_power/models/journal_paper_w_lambda/"

# Get Grid Dimensions
N = 104
M = 88

resources_ = ['load', 'solar', 'wind']
sources_   = ['NP15', 'SP15', 'ZP26']

sl_methods_ = ['lasso', 'OMP', 'elastic_net', 'group_lasso', 'dense']
dl_methods_ = ['BLR', 'RVM', 'GPR', 'MTGPR', 'NGR']

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

# Get combination of hyperparameters in the results validation .csv
# (see manuscript Section Hyperparameters)
filename = "prob_model_selection_multisource.csv"
exps_    = pd.read_csv(path_to_mdl + filename)
idx_exp_ = exps_.index.values

# Get experiment hyperparameters for model testing
i_exp   = idx_exp_[i_exp]
thetas_ = ast.literal_eval(exps_.loc[i_exp, "parameters"])

# Energy features in the experiment
source      = exps_.loc[i_exp, "resource"]
i_resources = sources_.index(source)
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

# Assets in resources: {load:5, solar:3, wind:2}
i_resources_ = [[0, 1, 2], [0, 1, 2], [0, 1]][i_resources]
# Resources: [{MWD, PGE, SCE, SDGE, VEA}, {NP, SP, ZP}, {NP, SP}]
i_assets_ = [[[1], [0], [0]], [[2], [1], [1]], [[3], [2]]][i_resources]
# Generate columns names for error metrics
headers_ = [['NP15_load', 'NP15_solar', 'NP15_wind'],
            ['SP15_load', 'SP15_solar', 'SP15_wind'],
            ['ZP26_load', 'ZP26_solar'],
            ['NP15_load', 'SP15_load', 'ZP26_load', 'NP15_solar', 'SP15_solar', 'ZP26_solar', 'NP15_wind', 'SP15_wind']][i_resources]
print(headers_)

# Return original experiment index in the hyperparamters
# combinations list used in the validation
i_exp = int(exps_.loc[i_exp, "experiment"])
print(i_exp, source, score, sl_method, dl_method, thetas_, i_resources, SL, DL)

# Generate I/O files names for a given experiment
resource = '_'.join([resources_[i_resource] for i_resource in i_resources_])
dataset  = '_'.join(['{}-{}'.format(resources_[i_resource], '-'.join(map(str, i_assets_[i_resource])))
                     for i_resource in i_resources_]) + '_M{}.pkl'.format(i_mask)
print(resource, dataset)

# Loading probability distribution calibration parameters
# and predictive scenarios smoothing parameter
# Calibration/Smoothing active or not
calibration = True
smoothing   = True

smoothing_ts_, calibration_ts_ = _load_dict(path_to_prm,
                                            file_name = '{}-{}-{}_e{}.pkl'.format(source, sl_method, dl_method, i_exp))

smoothing_ts_   = np.mean(smoothing_ts_, axis = 0).tolist()
calibration_ts_ = np.mean(calibration_ts_, axis = 0)
print(smoothing_ts_, calibration_ts_.shape)

# Dense learning model standardization (DL model: 0 standaridation, 1: nostandaridation)
# (see manuscript section Data Preprocessing)
x_sl_stnd, y_sl_stnd = [[[1, 1], [0, 1], [1, 1], [1, 1], [1, 1]][DL],
                        [[1, 1], [0, 1], [1, 1], [1, 1], [1, 1]][DL],
                        [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]][DL],
                        [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]][DL],
                        [0, 0]][SL]
# Dense learning model standardization (DL model: 0 standaridation, 1: nostandaridation)
# (see manuscript section Data Preprocessing)
x_dl_stnd, y_dl_stnd = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]][DL]

# Experiment key
key = r'{}-{}_{}{}-{}{}_{}'.format(sl_method, dl_method, x_sl_stnd, y_sl_stnd, x_dl_stnd, y_dl_stnd, thetas_)
print(key)

# Loading spatial masks
# (see SI Section Spatial Masks)
M_ = _load_spatial_masks(i_resources_, path_to_aux)

# Loading preprocessed (filtered) dataset
# (see manuscript Section Processing and Filtering)
# (see SI Section Data Processing)
X_sl_, Y_sl_, g_sl_, X_dl_, Y_dl_, g_dl_, Z_, ZZ_, Y_ac_, Y_fc_ = _load_processed_dataset(dataset, path_to_prc)
# print(X_sl_.shape, Y_sl_.shape, g_sl_.shape)
# print(X_dl_.shape, Y_dl_.shape, g_dl_.shape)
# print(Z_.shape, ZZ_.shape, Y_ac_.shape, Y_fc_.shape)

# Split SL dataset data in training and testing
# (see manuscript Section Validation, Training, and Testing)
X_sl_tr_, X_sl_ts_ = _training_and_testing_dataset(X_sl_)
Y_sl_tr_, Y_sl_ts_ = _training_and_testing_dataset(Y_sl_)
print(X_sl_tr_.shape, Y_sl_tr_.shape, X_sl_ts_.shape, Y_sl_ts_.shape)

# Clean unused variables from RAM memory
del X_sl_, Y_sl_

# Split DL dataset in training and testing
# (see manuscript Section Validation, Training, and Testing)
X_dl_tr_, X_dl_ts_ = _training_and_testing_dataset(X_dl_)
Y_dl_tr_, Y_dl_ts_ = _training_and_testing_dataset(Y_dl_)
print(X_dl_tr_.shape, Y_dl_tr_.shape, X_dl_ts_.shape, Y_dl_ts_.shape)

# Clean unused variables from RAM memory
del X_dl_, Y_dl_

meta_ts_ = pd.DataFrame(ZZ_[0, -Y_dl_ts_.shape[0]:, [0, 1, 2, 3]].T, columns = ['year',
                                                                                'month',
                                                                                'day',
                                                                                'yearday'])

# Clean unused variables from RAM memory
del Z_, ZZ_

# Naive and CAISO forecasts as baselines
# (see manuscript Section AI-Based Probabilistic Models
# Enhance the Performance of a Day-Ahead Energy Forecast)
Y_per_fc_, Y_ca_fc_, Y_clm_fc_ = _naive_forecasts(Y_ac_, Y_fc_, N_lags)
#print(Y_per_fc_.shape, Y_ca_fc_.shape, Y_clm_fc_.shape)

# Clean unused variables from RAM memory
del Y_ac_, Y_fc_

# Split baseline data in training and testing
# (see manuscript Section Validation, Training, and Testing)
Y_per_fc_tr_, Y_per_fc_ts_ = _training_and_testing_dataset(Y_per_fc_)
Y_ca_fc_tr_, Y_ca_fc_ts_   = _training_and_testing_dataset(Y_ca_fc_)
Y_clm_fc_tr_, Y_clm_fc_ts_ = _training_and_testing_dataset(Y_clm_fc_)
# print(Y_per_fc_tr_.shape, Y_per_fc_ts_.shape)
# print(Y_ca_fc_tr_.shape, Y_ca_fc_ts_.shape)
# print(Y_clm_fc_tr_.shape, Y_clm_fc_ts_.shape)

# Clean unused variables from RAM memory
del Y_per_fc_, Y_ca_fc_, Y_clm_fc_

# Generate sparse learning training and testing dataset in the correct format
# (see manuscript Section Feature Vectors for Sparse Learning)
X_sl_tr_, Y_sl_tr_ = _sparse_learning_dataset_format(X_sl_tr_, Y_sl_tr_)
X_sl_ts_, Y_sl_ts_ = _sparse_learning_dataset_format(X_sl_ts_, Y_sl_ts_)
#print(X_sl_tr_test_.shape, Y_sl_tr_test_.shape, X_sl_ts_test_.shape, Y_sl_ts_test_.shape)

# Standardize SL dataset
# (see manuscript section Data Preprocessing)
X_sl_tr_stnd_, Y_sl_tr_stnd_, X_sl_ts_stnd_, sl_scaler_ = _sparse_learning_stand(X_sl_tr_,
                                                                                 Y_sl_tr_,
                                                                                 X_sl_ts_,
                                                                                 x_sl_stnd,
                                                                                 y_sl_stnd)
#print(X_sl_tr_stnd_.shape, Y_sl_tr_stnd_.shape, X_sl_ts_stnd_.shape)

# Training SL model
# (see manuscript Section Sparse learning)
t_sl_tr = time.time()
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
#print(W_hat_.shape, Y_sl_ts_hat_.shape)

# Standardize DL dataset
# (see manuscript Section Data Preprocessing)
X_dl_tr_stnd_, Y_dl_tr_stnd_, X_dl_ts_stnd_, dl_scaler_ = _dense_learning_stand(X_dl_tr_,
                                                                                Y_dl_tr_,
                                                                                X_dl_ts_,
                                                                                x_dl_stnd,
                                                                                y_dl_stnd)
#print(X_dl_tr_stnd_.shape, Y_dl_tr_stnd_.shape, X_dl_ts_stnd_.shape)

# Traning DL recursively with a model chain
# (see manuscript Section Bayesian Learning)
t_dl_tr = time.time()
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

# Joint predictive distributions
t_ts = time.time()
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
#print(M_dl_ts_hat_.shape, S2_dl_ts_hat_.shape, C_dl_ts_hat_.shape)

# Calibrate predictive distribution
# (see manuscript Section Predictive Density Calibratio)
if calibration:
    S2_dl_ts_hat_, C_dl_ts_hat_ = _calibrate_predictive_covariance(M_dl_ts_hat_,
                                                                   C_dl_ts_hat_,
                                                                   calibration_ts_)

# Joint predictive scenarios
# (see manuscript Figure 7a)
t_prob_ts = time.time()
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
                                           N_scenarios = 100,
                                           calibration = calibration)

# Smooth scenarios shape
# (see manuscript Section Scenario Smoothing)
if smoothing:
    Y_dl_ts_hat_ = _calibrate_scenarios_temporal_structure(Y_dl_ts_hat_,
                                                           sigmas_ = smoothing_ts_)

# Scenario drawing time                                                                                                                     calibration = calibration)
t_prob_ts = time.time() - t_prob_ts

# Save timing results
time_ = pd.DataFrame([t_sl_tr, t_dl_tr, t_ts, t_prob_ts], columns = ['time'],
                                                          index   = ['sparse_training',
                                                                     'dense_training',
                                                                     'testing',
                                                                     'prob_testing'])

# Evaluate baseline forecast with deterministic error metrics
# (see SI section Scoring Rules)
# Aggregated across samples
E_per_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_per_fc_ts_, headers_)
E_ca_ts_  = _baseline_det_metrics(Y_dl_ts_, Y_ca_fc_ts_, headers_)
E_clm_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_clm_fc_ts_, headers_)
# For each sample
E_per_ts_all_ = _baseline_det_metrics_dist(Y_dl_ts_,
                                           Y_per_fc_ts_,
                                           headers_,
                                           'persistence')

E_ca_ts_all_ = _baseline_det_metrics_dist(Y_dl_ts_,
                                          Y_ca_fc_ts_,
                                          headers_,
                                          'caiso')

E_clm_ts_all_ = _baseline_det_metrics_dist(Y_dl_ts_,
                                           Y_clm_fc_ts_,
                                           headers_,
                                           'climatology')

# Save model and outputs
_model                   = {}
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
_model['bayesian_scoring']  _multivariate_prob_metrics(Y_dl_ts_,
                                                       M_dl_ts_hat_,
                                                       C_dl_ts_hat_,
                                                       S2_dl_ts_hat_,
                                                       Y_dl_ts_hat_)

_model['deterministic_scoring'] = _baseline_det_metrics(Y_dl_ts_,
                                                        M_dl_ts_hat_,
                                                        headers_)

_model['baseline_scoring'] = pd.concat([E_per_ts_,
                                        E_ca_ts_,
                                        E_clm_ts_], axis = 0).reset_index(drop = True)
# For each sample
_model['bayesian_scoring_all'] = _multisource_prob_metrics_dist(Y_dl_ts_,
                                                                M_dl_ts_hat_,
                                                                C_dl_ts_hat_,
                                                                S2_dl_ts_hat_,
                                                                Y_dl_ts_hat_)

_model['deterministic_scoring_all'] = _baseline_det_metrics_dist(Y_dl_ts_,
                                                                 M_dl_ts_hat_,
                                                                 headers_,
                                                                 'ml')

_model['baseline_scoring_all'] = pd.concat([E_per_ts_all_,
                                            E_ca_ts_all_,
                                            E_clm_ts_all_], axis = 0).reset_index(drop = True)

# Save MT-GPR predictive distribution and scenarios
_model['mean']       = M_dl_ts_hat_
_model['covariance'] = C_dl_ts_hat_
_model['variance']   = S2_dl_ts_hat_
_model['samples']    = Y_dl_ts_hat_

# Save baseline forecasts
_model['climatology'] = Y_clm_fc_ts_
_model['caiso']       = Y_ca_fc_ts_
_model['persitence']  = Y_per_fc_ts_

_save_dict(_model,
           path_to_mdl,
           file_name = '{}-{}-{}-{}.pkl'.format(source, sl_method, dl_method, score))

