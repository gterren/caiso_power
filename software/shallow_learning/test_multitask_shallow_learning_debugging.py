import sys, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from itertools import product

from utils import *

# Degine path to data
path_to_pds = r"/home/gterren/caiso_power/data/dataset-2023/"
path_to_aux = r"/home/gterren/caiso_power/data/auxiliary/"
path_to_rst = r"/home/gterren/caiso_power/results/"
path_to_img = r"/home/gterren/caiso_power/images/"

# Notes:
# SL = 0: X = {0, 1}, Y = {0}
# SL = 1: X = {1}, Y = {0}
# SL = 2: X = {0, 1}, Y = {0, 1}
# SL = 3: X = {0, 1}, Y = {0}
# DL = 0: X = {0, 1}, Y = {0}

# Get Grid Dimensions
N = 104
M = 88
# Assets in resources: {load:5, solar:3, wind:2}
i_resources_ = [1]
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
# Sparse learning model standardization
x_sl_stnd = int(sys.argv[1])
y_sl_stnd = int(sys.argv[2])
# Dense learning model standardization
x_dl_stnd = int(sys.argv[3])
y_dl_stnd = int(sys.argv[4])
# Sparse learning model index
SL = int(sys.argv[5])
DL = int(sys.argv[6])
MTGP = 'Bonilla'
# Define identification experiment key
key = '{}{}_{}{}{}{}_{}{}_{}{}_{}{}'.format(N_lags, i_mask, AR, CS, TM, RC, x_sl_stnd, y_sl_stnd, x_dl_stnd, y_dl_stnd, SL, DL)
print(key)

i_theta = int(sys.argv[7])
# Load the index of US land in the NOAA operational forecast
US_land_ = pd.read_pickle(path_to_aux + r"USland_0.125_(-125,-112)_(32,43).pkl")
# Load the index of Demand, Solar, and Wind land in the NOAA operational forecast
D_den_, S_den_, W_den_ = pd.read_pickle(path_to_aux + r"density_grid_0.125_(-125,-112)_(32,43).pkl")
#print(US_land_.shape, D_den_.shape, S_den_.shape, W_den_.shape)
# Define spatial feature masks
F_ = np.zeros(US_land_.shape)
for i_resource in i_resources_:
    F_ += [D_den_, S_den_, W_den_][i_resource]
#M_ = [np.ones(US_land_.shape), US_land_, D_den_ + S_den_ + W_den_, [D_den_, S_den_, W_den_][i_resource]]
M_ = [np.ones(US_land_.shape), US_land_, D_den_ + S_den_ + W_den_, F_]


# Load proposed data
data_ = _load_data_in_chunks([2022, 2023], path_to_pds)
#print(len(data_))

# Define data structure for a given experiment
#Y_ac_, Y_fc_, X_ac_, X_fc_, Z_, g_sl_, g_dl_ = _structure_dataset(data_, i_resource, M_[i_mask], tau)
Y_ac_, Y_fc_, X_ac_, X_fc_, Z_, g_sl_, g_dl_, assets_ = _multisource_structure_dataset(data_, i_resources_, i_assets_, M_[i_mask], tau)
#print(Y_ac_.shape, Y_fc_.shape, X_ac_.shape, X_fc_.shape, Z_.shape, g_sl_.shape, g_dl_.shape)
del data_

# Generate sparse learning dataset
X_sl_, Y_sl_, g_sl_ = _dense_learning_dataset(X_ac_, Y_ac_, Z_, g_sl_, N_lags, AR = 0, CS = 0, TM = 1)
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
del X_fc_, X_dl_, Y_dl_, Z_

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

# Compute baselines det. error metrics
E_per_ts_ = _det_metrics(Y_dl_ts_, Y_per_fc_ts_)
E_ca_ts_  = _det_metrics(Y_dl_ts_, Y_ca_fc_ts_)
E_clm_ts_ = _det_metrics(Y_dl_ts_, Y_clm_fc_ts_)
#print(E_per_ts_.shape, E_ca_ts_.shape, E_clm_ts_.shape)

# Sparse and dense learning hyper-parameters
if x_sl_stnd == 0: alphas_ = [0.001, 0.01, 0.1, 1., 10., 100.]
else:              alphas_ = [0.000001, 0.00001, 0.001, 0.01, 0.1]
betas_   = [10, 20, 40, 80, 160, 320]  #betas_   = np.linspace(10, X_sl_tr_.shape[1]/2., N_hyps, dtype = int)
omegas_  = [0.01, 0.25, 0.5, 0.75]
gammas_  = [0.1, 1., 10.]
etas_    = [0.25, 0.5, 0.75, 1.]
lambdas_ = [1., 10., 100., 1000.]
xis_     = [0, 1, 4, 5, 6] # ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']
# Get combination of possible parameters
thetas_, N_thetas = _get_cv_param(alphas_, betas_, omegas_, gammas_, etas_, lambdas_, xis_, SL, DL)

# Find optimal parameters
i_theta_ = [i_theta, i_theta, i_theta]
t_init   = time.time()
# Generate sparse learning training and testing dataset in the correct format
X_sl_tr_test_, Y_sl_tr_test_ = _sparse_learning_dataset(X_sl_tr_, Y_sl_tr_)
X_sl_ts_test_, Y_sl_ts_test_ = _sparse_learning_dataset(X_sl_ts_, Y_sl_ts_)
#print(X_sl_tr_test_.shape, Y_sl_tr_test_.shape, X_sl_ts_test_.shape, Y_sl_ts_test_.shape)

# Standardize sparse learning dataset
X_sl_tr_stnd_, Y_sl_tr_stnd_, X_sl_ts_stnd_, sl_scaler_ = _sparse_learning_stand(X_sl_tr_test_, Y_sl_tr_test_, X_sl_ts_test_, x_sl_stnd, y_sl_stnd)
#print(X_sl_tr_stnd_.shape, Y_sl_tr_stnd_.shape, X_sl_ts_stnd_.shape)

# Fit sparse learning model
W_hat_ = _fit_sparse_learning(X_sl_tr_stnd_, X_sl_ts_stnd_, Y_sl_tr_stnd_, Y_sl_ts_test_, g_sl_, thetas_, i_theta_, SL, y_sl_stnd)

# Standardize dense learning dataset
X_dl_tr_stnd_, Y_dl_tr_stnd_, X_dl_ts_stnd_, dl_scaler_ = _dense_learning_stand(X_dl_tr_, Y_dl_tr_, X_dl_ts_, x_dl_stnd, y_dl_stnd)
#print(X_dl_val_tr_.shape, Y_dl_val_tr_.shape, X_dl_val_ts_.shape)

# Fit multitask dense learning - Bayesian model chain
models_ = _fit_multitask_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_, W_hat_, g_dl_, thetas_, i_theta, RC, DL)

# Make multitask probabilistic prediction
Y_dl_ts_hat_, M_dl_ts_hat_, S2_dl_ts_hat_ = _multitask_prob_predict(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_,
                                                                    thetas_, i_theta, RC, DL, y_dl_stnd, N_samples = 100)

# Make multitask joint probabilistic prediction
Y_dl_ts_joint_hat_ = _multitask_joint_prob_predict(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_,
                                                   thetas_, i_theta_, RC, DL, y_dl_stnd, N_samples = 100)

t_end = time.time() - t_init

# # Evaluate dense learning deterministic error metrics
# E_dl_ = _det_metrics(Y_dl_ts_, M_dl_ts_hat_)
# # Evaluate dense learning Bayesian metrics
# B_dl_ = _mv_LPP_score(Y_dl_ts_, M_dl_ts_hat_, S2_dl_ts_hat_)
# # Evaluate dense learning probabilistic metrics
# P_dl_  = _mv_CRPS_score(Y_dl_ts_, Y_dl_ts_hat_)
# #P_dl_joint_ = _mv_CRPS_score(Y_dl_ts_, Y_dl_ts_joint_hat_)
# # Save parameter combination in .csv file
# _save_test_in_csv_file(E_dl_[:, 0, :], key, t_end, i_theta_, [thetas_[i_theta] for i_theta in i_theta_], assets_, path_to_rst, 'MTRMSETest.csv')
# _save_test_in_csv_file(E_dl_[:, 1, :], key, t_end, i_theta_, [thetas_[i_theta] for i_theta in i_theta_], assets_, path_to_rst, 'MTMAETest.csv')
# _save_test_in_csv_file(E_dl_[:, 2, :], key, t_end, i_theta_, [thetas_[i_theta] for i_theta in i_theta_], assets_, path_to_rst, 'MTMBETest.csv')
# _save_test_in_csv_file(B_dl_[:, np.newaxis].T, key, t_end, i_theta_, [thetas_[i_theta] for i_theta in i_theta_], [''. join(str(e) for e in assets_)], path_to_rst, 'MTBayTest.csv')
# _save_test_in_csv_file(P_dl_, key, t_end, i_theta_, [thetas_[i_theta] for i_theta in i_theta_], assets_, path_to_rst, 'MTProbTest.csv')
# _save_test_in_csv_file(P_dl_joint_, key, t_end, i_theta_, [thetas_[i_theta] for i_theta in i_theta_], assets_, path_to_rst, 'MTJointProbTest.csv')

# # Average error metrics
# e_dl_ = np.mean(E_dl_, axis = -1)
# b_dl_ = np.mean(B_dl_, axis = -1)
# p_dl_ = np.mean(P_dl_, axis = -1) #np.concatenate((np.mean(P_dl_, axis = -1)[:, np.newaxis], np.mean(P_dl_joint_, axis = -1)[:, np.newaxis]), axis = 1).T
# print(e_dl_)
# print(b_dl_)
# print(p_dl_)

# def _variogram_score(Y_, Y_hat_, p = .5):
#
#     N_observation, N_tasks, N_horizons, N_samples = Y_hat_.shape
#     score_ = np.zeros((N_horizons, N_observation))
#
#     for n in range(N_horizons):
#         for i in range(N_observation):
#
#             frac = 0
#             for j in range(N_tasks):
#                 for k in range(N_tasks):
#
#                     frac1 = np.absolute(Y_[i, j, n] - Y_[i, k, n])**p
#                     frac2 = 0
#                     for m in range(N_samples):
#                         frac2 += np.absolute(Y_hat_[i, j, n, m] - Y_hat_[i, k, n, m])**p
#                     frac2 /= N_samples
#
#                     frac += (frac1 - frac2)**2
#
#             score_[n, i] = frac
#     return score_
#
# def _energy_score(Y_, Y_hat_):
#
#     N_observation, N_tasks, N_horizons, N_samples = Y_hat_.shape
#     score_ = np.zeros((N_horizons, N_observation))
#
#     for m in range(N_horizons):
#         for i in range(N_observation):
#
#             frac1 = 0
#             for j in range(N_samples):
#                 frac1 += (Y_hat_[i, :, m, j] - Y_[i, :, m]).T @ (Y_hat_[i, :, m, j] - Y_[i, :, m])
#             frac1 /= N_samples
#
#             frac2 = 0
#             for j in range(N_samples):
#                 for k in range(N_samples):
#                     frac2 += (Y_hat_[i, :, m, j] - Y_hat_[i, :, m, k]).T @ (Y_hat_[i, :, m, j] - Y_hat_[i, :, m, k])
#
#             frac2 /= 2*(N_samples**2)
#         score_[m, i] = (frac1 - frac2)
#     return score_


# ES_dl_ = _energy_score(Y_dl_ts_, Y_dl_ts_hat_)
# VS_dl_ = _variogram_score(Y_dl_ts_, Y_dl_ts_hat_, p = .5)
#
# print(np.mean(ES_dl_, axis = 1))
# print(np.mean(ES_dl_, axis = 1).mean())
#
# print(np.mean(VS_dl_, axis = 1))
# print(np.mean(VS_dl_, axis = 1).mean())
#
# ES_dl_ = _energy_score(Y_dl_ts_, Y_dl_ts_joint_hat_)
# VS_dl_ = _variogram_score(Y_dl_ts_, Y_dl_ts_joint_hat_, p = .5)
#
# print(np.mean(ES_dl_, axis = 1))
# print(np.mean(ES_dl_, axis = 1).mean())
#
# print(np.mean(VS_dl_, axis = 1))
# print(np.mean(VS_dl_, axis = 1).mean())

# # Save parameter combination in .csv file
# _save_test_in_csv_file(e_dl_, key, t_end, i_theta_, [thetas_[i_theta] for i_theta in i_theta_], assets_, path_to_rst, 'avgMTDetTest.csv')
# _save_test_in_csv_file(b_dl_[np.newaxis, np.newaxis], key, t_end, i_theta_, [thetas_[i_theta] for i_theta in i_theta_], [''. join(str(e) for e in assets_)], path_to_rst, 'avgMTBayTest.csv')
# _save_test_in_csv_file(p_dl_, key, t_end, i_theta_, [thetas_[i_theta] for i_theta in i_theta_], assets_, path_to_rst, 'avgMTProbTest.csv')

# Save predictions in a pickle file
#_save_pred_in_pkl_file([Y_dl_ts_, M_dl_ts_hat_, S2_dl_ts_hat_, Y_dl_ts_hat_, Y_dl_ts_joint_hat_], key, [''. join(str(e) for e in assets_)], path_to_rst, '_pred_noise.pkl')
