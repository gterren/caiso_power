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
path_to_pds = r"/home/gterren/caiso_power/data/dataset-2023/"
path_to_aux = r"/home/gterren/caiso_power/data/auxiliary/"
path_to_rst = r"/home/gterren/caiso_power/results/"
path_to_img = r"/home/gterren/caiso_power/images/"
path_to_mdl = r"/home/gterren/caiso_power/models/"

# Notes:
# SL = 0: X = {0, 1}, Y = {0}
# SL = 1: X = {1}, Y = {0}
# SL = 2: X = {0, 1}, Y = {0, 1}
# SL = 3: X = {0, 1}, Y = {0}
# DL = 0: X = {0, 1}, Y = {0}

# Get Grid Dimensions
N = 104
M = 88

dl_methods_ = ['BLR', 'RVM', 'GPR', 'MTGPR', 'NGR']
sl_methods_ = ['lasso', 'OMP', 'elastic_net', 'group_lasso', 'dense']
resources_  = ['load', 'solar', 'wind']

# Resources: [{MWD, PGE, SCE, SDGE, VEA}, {NP, SP, ZP}, {NP, SP}]
i_assets_ = [[1, 2, 3], [0, 1, 2], [0, 1]]

# Spatial masks {All, US Land, All CAISO Resources density, Resource density}
i_mask = 3
tau    = 0.
N_lags = 6

# Autoregressive series from today:
AR = 1
# Cyclo-stationary from lagged series
CS = 1
# Time dammy variables
TM = 1
# Recursive forecast in covariates
RC = 1

key = ''

SL           = 1
DL           = 2
i_resources_ = [1]
sl_method    = sl_methods_[SL]
dl_method    = dl_methods_[DL]
resource     =  '_'.join([resources_[i_resource] for i_resource in i_resources_])
print(sl_method, dl_method, resource)

theta_1_ = (10,7)
theta_2_ = (10,7)
theta_3_ = (10,7)
thetas_  = [theta_1_, theta_2_, theta_3_]
print(thetas_)

# Sparse learning model standardization
x_sl_stnd, y_sl_stnd = [[[1, 1],[0, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[1, 1],[0, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[1, 1],[1, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]][DL],
                         [0, 0]][SL]
# Dense learning model standardization
x_dl_stnd, y_dl_stnd = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]][DL]

# Loading spatial masks
M_ = _load_spatial_masks(i_resources_, path_to_aux)

# Load proposed data
data_ = _load_data_in_chunks([2023], path_to_pds)
#print(len(data_))

# Define data structure for a given experiment
Y_ac_, Y_fc_, X_ac_, X_fc_, Z_, ZZ_, g_sl_, g_dl_, assets_ = _multisource_structure_dataset(data_, i_resources_, i_assets_, M_[i_mask], tau)
#print(Y_ac_.shape, Y_fc_.shape, X_ac_.shape, X_fc_.shape, Z_.shape, g_sl_.shape, g_dl_.shape)
del data_

# Generate sparse learning dataset
X_sl_, Y_sl_, g_sl_ = _dense_learning_dataset(X_ac_, Y_ac_, Z_, g_sl_, N_lags, AR = 0,
                                                                               CS = 0,
                                                                               TM = 1)
#print(X_sl_.shape, Y_sl_.shape, g_sl_.shape)

# Split data in training and testing
X_sl_tr_, X_sl_ts_ = _training_and_testing_dataset(X_sl_)
Y_sl_tr_, Y_sl_ts_ = _training_and_testing_dataset(Y_sl_)
#print(X_sl_tr_.shape, Y_sl_tr_.shape, X_sl_ts_.shape, Y_sl_ts_.shape)
del X_ac_, X_sl_, Y_sl_

# Generate dense learning dataset
X_dl_, Y_dl_, g_dl_ = _dense_learning_dataset(X_fc_, Y_ac_, Z_, g_dl_, N_lags, AR, CS, TM)
#print(X_dl_.shape, Y_dl_.shape, g_dl_.shape)

# Split data in training and testing
X_dl_tr_, X_dl_ts_ = _training_and_testing_dataset(X_dl_)
Y_dl_tr_, Y_dl_ts_ = _training_and_testing_dataset(Y_dl_)
#print(X_dl_tr_.shape, Y_dl_tr_.shape, X_dl_ts_.shape, Y_dl_ts_.shape)

meta_ts_ = pd.DataFrame(ZZ_[0, -Y_dl_ts_.shape[0]:, [0, 1, 2, 3]].T, columns = ['year', 'month', 'day', 'yearday'])

del X_fc_, X_dl_, Y_dl_, Z_, ZZ_
# Naive and CAISO forecasts as baselines
Y_per_fc_, Y_ca_fc_, Y_clm_fc_ = _naive_forecasts(Y_ac_, Y_fc_, N_lags)
#print(Y_per_fc_.shape, Y_ca_fc_.shape, Y_clm_fc_.shape)
del Y_ac_, Y_fc_

# Split data in training and testing
Y_per_fc_tr_, Y_per_fc_ts_ = _training_and_testing_dataset(Y_per_fc_)
Y_ca_fc_tr_, Y_ca_fc_ts_   = _training_and_testing_dataset(Y_ca_fc_)
Y_clm_fc_tr_, Y_clm_fc_ts_ = _training_and_testing_dataset(Y_clm_fc_)
#print(Y_per_fc_tr_.shape, Y_per_fc_ts_.shape)
#print(Y_ca_fc_tr_.shape, Y_ca_fc_ts_.shape)
#print(Y_clm_fc_tr_.shape, Y_clm_fc_ts_.shape)
del Y_per_fc_, Y_ca_fc_, Y_clm_fc_

# Generate sparse learning training and testing dataset in the correct format
X_sl_tr_, Y_sl_tr_ = _sparse_learning_dataset_format(X_sl_tr_, Y_sl_tr_)
X_sl_ts_, Y_sl_ts_ = _sparse_learning_dataset_format(X_sl_ts_, Y_sl_ts_)
#print(X_sl_tr_test_.shape, Y_sl_tr_test_.shape, X_sl_ts_test_.shape, Y_sl_ts_test_.shape)

sl_tr_time = time.time()
print(x_sl_stnd, y_sl_stnd, thetas_)

# Standardize sparse learning dataset
X_sl_tr_stnd_, Y_sl_tr_stnd_, X_sl_ts_stnd_, sl_scaler_ = _sparse_learning_stand(X_sl_tr_, Y_sl_tr_, X_sl_ts_, x_sl_stnd, y_sl_stnd)
#print(X_sl_tr_stnd_.shape, Y_sl_tr_stnd_.shape, X_sl_ts_stnd_.shape)

# Fit sparse learning model
W_hat_, Y_sl_ts_hat_ = _fit_sparse_learning(X_sl_tr_stnd_, X_sl_ts_stnd_, Y_sl_tr_stnd_, Y_sl_ts_, g_sl_, thetas_, sl_scaler_, SL, y_sl_stnd)
#print(W_hat_.shape, Y_sl_ts_hat_.shape)

sl_tr_time = time.time() - sl_tr_time

# Standardize dense learning dataset
dl_tr_time = time.time()

X_dl_tr_stnd_, Y_dl_tr_stnd_, X_dl_ts_stnd_, dl_scaler_ = _dense_learning_stand(X_dl_tr_, Y_dl_tr_, X_dl_ts_, x_dl_stnd, y_dl_stnd)
#print(X_dl_tr_stnd_.shape, Y_dl_tr_stnd_.shape, X_dl_ts_stnd_.shape)

# Fit sense learning - Bayesian model chain
models_ = _fit_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_, W_hat_, g_dl_, thetas_, RC, DL)

dl_tr_time = time.time() - dl_tr_time

# Independent prediction with conficence intervals
dl_ts_time = time.time()

M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_ = _pred_prob_dist(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd)
#print(M_dl_ts_hat_.shape, S2_dl_ts_hat_.shape, C_dl_ts_hat_.shape)

dl_ts_time = time.time() - dl_ts_time

# Joint probabilistic predictions
pr_ts_time = time.time()

Y_dl_ts_hat_ = _joint_prob_prediction(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd, N_samples = 100)
#print(Y_dl_ts_.shape, Y_dl_ts_hat_.shape)

pr_ts_time = time.time() - pr_ts_time

time_ = pd.DataFrame([sl_tr_time, dl_tr_time, dl_ts_time, pr_ts_time], columns = ['time'],
                                                                       index   = ['sparse_training',
                                                                                  'dense_training',
                                                                                  'testing',
                                                                                  'prob_testing'])
print(time_)

E_per_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_per_fc_ts_)
E_ca_ts_  = _baseline_det_metrics(Y_dl_ts_, Y_ca_fc_ts_)
E_clm_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_clm_fc_ts_)

E_per_ts_all_ = _baseline_det_metrics_dist(Y_dl_ts_, Y_per_fc_ts_, 'persistence')
E_ca_ts_all_  = _baseline_det_metrics_dist(Y_dl_ts_, Y_ca_fc_ts_, 'caiso')
E_clm_ts_all_ = _baseline_det_metrics_dist(Y_dl_ts_, Y_clm_fc_ts_, 'climatology')

# Save model and outputs
_model = {}
_model['time']                  = time_
_model['mask']                  = M_[i_mask]
_model['weights']               = W_hat_
_model['feature_labels']        = g_sl_

_model['testing_targets']       = Y_dl_ts_
_model['testing_targets_meta']  = meta_ts_

_model['bayesian_scoring']      = _multivariate_prob_metrics(Y_dl_ts_, M_dl_ts_hat_, C_dl_ts_hat_, Y_dl_ts_hat_)
_model['deterministic_scoring'] = _baseline_det_metrics(Y_dl_ts_, M_dl_ts_hat_)
_model['baseline_scoring_all']  = pd.concat([E_per_ts_, E_ca_ts_, E_clm_ts_], axis = 0).reset_index(drop = True)

_model['bayesian_scoring_all']      = _multivariate_prob_metrics_dist(Y_dl_ts_, M_dl_ts_hat_, C_dl_ts_hat_, Y_dl_ts_hat_)
_model['deterministic_scoring_all'] = _baseline_det_metrics_dist(Y_dl_ts_, M_dl_ts_hat_, 'ml')
_model['baseline_scoring_all']      = pd.concat([E_per_ts_all_, E_ca_ts_all_, E_clm_ts_all_], axis = 0).reset_index(drop = True)

_model['mean']                  = M_dl_ts_hat_
_model['covarice']              = C_dl_ts_hat_
_model['standard_deviation']    = S2_dl_ts_hat_
_model['samples']               = Y_dl_ts_hat_

_save_dict(_model, path_to_mdl, file_name = '{}-{}-{}-{}.pkl'.format(resource, sl_method, dl_method, key))

