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
path_to_mdl = r"/home/gterren/caiso_power/models/journal_paper_w_sigma/"

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

i_resource = int(sys.argv[1])
i_exp      = int(sys.argv[2])

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

# Define identification experiment key
i_job     = 0
N_jobs    = 1
i_batch   = 0
N_batches = 1

filename = "prob_model_selection_{}.csv".format(resources_[i_resource])
exps_    = pd.read_csv(path_to_mdl + filename)
idx_exp_ = exps_.index.values
i_exp    = idx_exp_[i_exp]

score     = exps_.loc[i_exp, "score"]
sl_method = exps_.loc[i_exp, "sparse_method"]
dl_method = exps_.loc[i_exp, "dense_method"]
SL        = sl_methods_.index(sl_method)
DL        = dl_methods_.index(dl_method)
print(sl_method, dl_method, SL, DL)

# Generate input and output file names
resources    = exps_.loc[i_exp, "resource"].rsplit('_')
i_resources_ = [resources_.index(resource) for resource in resources]
resource     = '_'.join([resources_[i_resource] for i_resource in i_resources_])
dataset      = '_'.join(['{}-{}'.format(resources_[i_resource], '-'.join(map(str, i_assets_[i_resource]))) for i_resource in i_resources_]) + '_M{}.pkl'.format(i_mask)

theta_   = ast.literal_eval(exps_.loc[i_exp, "parameters"])
sigmas_  = ast.literal_eval(exps_.loc[i_exp, "sigmas"])
lambdas_ = ast.literal_eval(exps_.loc[i_exp, "lambdas"])
thetas_  = [theta_, theta_, theta_]
print(thetas_)
print(sigmas_)
print(lambdas_)

# Sparse learning model standardization
x_sl_stnd, y_sl_stnd = [[[1, 1],[0, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[1, 1],[0, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[1, 1],[1, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]][DL],
                        [0, 0]][SL]
# Dense learning model standardization
x_dl_stnd, y_dl_stnd = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]][DL]

key = r'{}-{}_{}{}-{}{}_{}'.format(sl_method, dl_method, x_sl_stnd, y_sl_stnd, x_dl_stnd, y_dl_stnd, theta_)

# Loading spatial masks
M_ = _load_spatial_masks(i_resources_, path_to_aux)

# Loading preprocessed (filtered) dataset
X_sl_, Y_sl_, g_sl_, X_dl_, Y_dl_, g_dl_, Z_, ZZ_, Y_ac_, Y_fc_ = _load_processed_dataset(dataset, path_to_prc)

# Split data in training and testing
X_sl_tr_, X_sl_ts_ = _training_and_testing_dataset(X_sl_)
Y_sl_tr_, Y_sl_ts_ = _training_and_testing_dataset(Y_sl_)
#print(X_sl_tr_.shape, Y_sl_tr_.shape, X_sl_ts_.shape, Y_sl_ts_.shape)
del X_sl_, Y_sl_

# Split data in training and testing
X_dl_tr_, X_dl_ts_ = _training_and_testing_dataset(X_dl_)
Y_dl_tr_, Y_dl_ts_ = _training_and_testing_dataset(Y_dl_)
#print(X_dl_tr_.shape, Y_dl_tr_.shape, X_dl_ts_.shape, Y_dl_ts_.shape)
del X_dl_, Y_dl_

meta_ts_ = pd.DataFrame(ZZ_[0, -Y_dl_ts_.shape[0]:, [0, 1, 2, 3]].T, columns = ['year', 'month', 'day', 'yearday'])
del Z_, ZZ_

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

# Standardize sparse learning dataset
X_sl_tr_stnd_, Y_sl_tr_stnd_, X_sl_ts_stnd_, sl_scaler_ = _sparse_learning_stand(X_sl_tr_, Y_sl_tr_, X_sl_ts_, x_sl_stnd, y_sl_stnd)
#print(X_sl_tr_stnd_.shape, Y_sl_tr_stnd_.shape, X_sl_ts_stnd_.shape)

# Fit sparse learning model
t_sl_tr              = time.time()
W_hat_, Y_sl_ts_hat_ = _fit_sparse_learning(X_sl_tr_stnd_, X_sl_ts_stnd_, Y_sl_tr_stnd_, Y_sl_ts_, g_sl_, thetas_, sl_scaler_, SL, y_sl_stnd)
t_sl_tr              = time.time() - t_sl_tr
#print(W_hat_.shape, Y_sl_ts_hat_.shape)

X_dl_tr_stnd_, Y_dl_tr_stnd_, X_dl_ts_stnd_, dl_scaler_ = _dense_learning_stand(X_dl_tr_, Y_dl_tr_, X_dl_ts_, x_dl_stnd, y_dl_stnd)
#print(X_dl_tr_stnd_.shape, Y_dl_tr_stnd_.shape, X_dl_ts_stnd_.shape)

 # Fit sense learning - Bayesian model chain
t_dl_tr = time.time()
if DL != 3:
    models_ = _fit_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_, W_hat_, g_dl_, thetas_, RC, DL, key)
else:
    models_ = _fit_multitask_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_, W_hat_, g_dl_, thetas_, RC, DL, key)
t_dl_tr = time.time() - t_dl_tr

# Independent prediction with conficence intervals
t_ts = time.time()
if DL != 3:
    M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_ = _pred_prob_dist(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd)
else:
    M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_ = _multitask_pred_prob_dist(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd)
t_ts = time.time() - t_ts

# Calibrate predictive covariance matrix
S2_dl_ts_hat_, C_dl_ts_bar_ = _calibrate_predictive_covariance(C_dl_ts_hat_, lambdas_)

# Joint probabilistic predictions
t_prob_ts = time.time()
if DL != 3:
    Y_dl_ts_hat_ = _joint_prob_prediction(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, lambdas_, g_dl_, RC, DL, y_dl_stnd, N_samples = 100)
else:
    Y_dl_ts_hat_ = _multitask_joint_prediction(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, lambdas_, g_dl_, RC, DL, y_dl_stnd, N_samples = 100)

# Update scenarios shape
Y_dl_ts_hat_ = _calibrate_scenarios_temporal_structure(Y_dl_ts_hat_, sigmas_)

t_prob_ts = time.time() - t_prob_ts

time_ = pd.DataFrame([t_sl_tr, t_dl_tr, t_ts, t_prob_ts], columns = ['time'],
                                                          index   = ['sparse_training',
                                                                     'dense_training',
                                                                     'testing',
                                                                     'prob_testing'])

E_per_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_per_fc_ts_)
E_ca_ts_  = _baseline_det_metrics(Y_dl_ts_, Y_ca_fc_ts_)
E_clm_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_clm_fc_ts_)

E_per_ts_all_ = _baseline_det_metrics_dist(Y_dl_ts_, Y_per_fc_ts_, 'persistence')
E_ca_ts_all_  = _baseline_det_metrics_dist(Y_dl_ts_, Y_ca_fc_ts_, 'caiso')
E_clm_ts_all_ = _baseline_det_metrics_dist(Y_dl_ts_, Y_clm_fc_ts_, 'climatology')

# Save model and outputs
_model                              = {}
_model['time']                      = time_
_model['mask']                      = M_[i_mask]
_model['weights']                   = W_hat_
_model['feature_labels']            = g_sl_
_model['targets']                   = Y_dl_ts_
_model['targets_meta']              = meta_ts_
_model['bayesian_scoring']          = _multivariate_prob_metrics(Y_dl_ts_, M_dl_ts_hat_, C_dl_ts_hat_, S2_dl_ts_hat_, Y_dl_ts_hat_)
_model['deterministic_scoring']     = _baseline_det_metrics(Y_dl_ts_, M_dl_ts_hat_)
_model['baseline_scoring']          = pd.concat([E_per_ts_, E_ca_ts_, E_clm_ts_], axis = 0).reset_index(drop = True)
_model['bayesian_scoring_all']      = _multivariate_prob_metrics_dist(Y_dl_ts_, M_dl_ts_hat_, C_dl_ts_hat_, S2_dl_ts_hat_, Y_dl_ts_hat_)
_model['deterministic_scoring_all'] = _baseline_det_metrics_dist(Y_dl_ts_, M_dl_ts_hat_, 'ml')
_model['baseline_scoring_all']      = pd.concat([E_per_ts_all_, E_ca_ts_all_, E_clm_ts_all_], axis = 0).reset_index(drop = True)
_model['mean']                      = M_dl_ts_hat_
_model['covariance']                = C_dl_ts_hat_
_model['variance']                  = S2_dl_ts_hat_
_model['samples']                   = Y_dl_ts_hat_
_model['climatology']               = Y_clm_fc_ts_
_model['caiso']                     = Y_ca_fc_ts_
_model['persitence']                = Y_per_fc_ts_

_save_dict(_model, path_to_mdl, file_name = '{}-{}-{}-{}.pkl'.format(resource, sl_method, dl_method, score))
