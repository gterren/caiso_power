import sys, time

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from utils import *
from scoring_utils import *
from aux_utils import *
from loading_utils import *

# MPI job variables
i_job, N_jobs, comm = _get_node_info()

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

dl_methods_ = ['GP', 'RVM', 'KGP', 'MTKGP']
sl_methods_ = ['lasso', 'OMP', 'elastic_net', 'group_lasso']
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
# Sparse learning model index
SL = int(sys.argv[1])
DL = int(sys.argv[2])
sl_method = sl_methods_[SL]
dl_method = dl_methods_[DL]
print(sl_method, dl_method)
#MTGP = 'Bonilla'
i_batch   = int(sys.argv[3])
N_batches = 1
N_kfolds  = 5
# Sparse learning model standardization
x_sl_stnd, y_sl_stnd = [[0, 1], [0, 1], [1, 1], [1, 1]][SL]
#x_sl_stnd, y_sl_stnd = [[0, 1], [0, 1], [0, 1], [0, 1]][SL]
# Dense learning model standardization
x_dl_stnd, y_dl_stnd = [[1, 1], [1, 1], [1, 1], [1, 1]][DL]
# Sparse and dense learning hyper-parameters
alphas_  = [0.00001, 0.001, 0.01, 0.1, 1., 10.]
betas_   = [10, 20, 40, 80, 160, 320]
omegas_  = [0.01, 0.25, 0.5, 0.75]
gammas_  = [0.1, 1., 10.]
etas_    = [0.25, 0.5, 0.75, 1.]
lambdas_ = [1., 10., 100., 1000.]
xis_     = [0, 1, 4, 5, 6] # ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']
# Get combination of possible parameters
exp_, N_thetas = _get_cv_param(alphas_, betas_, omegas_, gammas_, etas_, lambdas_, xis_, SL, DL)
i_exps_        = _experiments_index_batch_job(exp_, i_batch, N_batches, i_job, N_jobs)
print(i_exps_, len(i_exps_))


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
data_ = _load_data_in_chunks([2019, 2020, 2021, 2022, 2023], path_to_pds)
#print(len(data_))

# Define data structure for a given experiment
Y_ac_, Y_fc_, X_ac_, X_fc_, Z_, g_sl_, g_dl_, assets_ = _multisource_structure_dataset(data_, i_resources_, i_assets_, M_[i_mask], tau)
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
#print(Y_dl_ts_.shape, Y_per_fc_ts_.shape, Y_ca_fc_ts_.shape, Y_clm_fc_ts_.shape)

E_per_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_per_fc_ts_)
E_ca_ts_  = _baseline_det_metrics(Y_dl_ts_, Y_ca_fc_ts_)
E_clm_ts_ = _baseline_det_metrics(Y_dl_ts_, Y_clm_fc_ts_)

print(E_per_ts_)
print(E_ca_ts_)
print(E_clm_ts_)

# Generate sparse learning training and testing dataset in the correct format
X_sl_tr_, Y_sl_tr_ = _sparse_learning_dataset_format(X_sl_tr_, Y_sl_tr_)
X_sl_ts_, Y_sl_ts_ = _sparse_learning_dataset_format(X_sl_ts_, Y_sl_ts_)
#print(X_sl_tr_test_.shape, Y_sl_tr_test_.shape, X_sl_ts_test_.shape, Y_sl_ts_test_.shape)

# Find optimal parameters
#R_sl_val_ts_ = []
R_dl_val_ts_ = []
for i_exp in i_exps_:
    # Initialize constants
    theta_  = exp_[i_exp]
    thetas_ = [theta_, theta_, theta_]
    t_init  = time.time()
    i_fold  = 0
    print(i_job, i_exp, x_sl_stnd, y_sl_stnd,  x_dl_stnd, y_dl_stnd, thetas_)
    # Initialize variables
    #E_sl_val_ts_  = np.zeros((N_kfolds, 3, 3))
    E_dl_val_ts_  = np.zeros((N_kfolds, 3, 3))
    P_dl_val_ts_  = np.zeros((N_kfolds, 3, 3))
    MV_dl_val_ts_ = np.zeros((N_kfolds, 4))
    # Loop over validation k-folds
    for idx_tr_, idx_ts_ in KFold(n_splits     = N_kfolds,
                                  random_state = None,
                                  shuffle      = False).split(X_dl_tr_):

        # Split spare learning validation partition in training and testing set
        X_sl_val_tr_, X_sl_val_ts_ = X_sl_tr_[idx_tr_, ...], X_sl_tr_[idx_ts_, ...]
        Y_sl_val_tr_, Y_sl_val_ts_ = Y_sl_tr_[idx_tr_, ...], Y_sl_tr_[idx_ts_, ...]
        #print(X_sl_val_tr_.shape, Y_sl_val_tr_.shape, X_sl_val_ts_.shape, Y_sl_val_ts_.shape)

        # Standardize sparse learning dataset
        X_sl_val_tr_stnd_, Y_sl_val_tr_stnd_, X_sl_val_ts_stnd_, sl_scaler_ = _sparse_learning_stand(X_sl_val_tr_, Y_sl_val_tr_, X_sl_val_ts_, x_sl_stnd, y_sl_stnd)
        #print(X_sl_tr_stnd_.shape, Y_sl_tr_stnd_.shape, X_sl_ts_stnd_.shape)

        # Fit sparse learning model
        W_hat_, Y_sl_val_ts_hat_ = _fit_sparse_learning(X_sl_val_tr_stnd_, X_sl_val_ts_stnd_, Y_sl_val_tr_stnd_, Y_sl_val_ts_, g_sl_, thetas_, sl_scaler_, SL, y_sl_stnd)
        #print(W_hat_.shape, Y_sl_ts_hat_.shape)

        #E_sl_val_ts_[i_fold, ...] = _sparse_det_metrics(Y_sl_val_ts_, Y_sl_val_ts_hat_).to_numpy()

        # Split dense learning validation partition in training and testing set
        X_dl_val_tr_, X_dl_val_ts_ = X_dl_tr_[idx_tr_, :], X_dl_tr_[idx_ts_, :]
        Y_dl_val_tr_, Y_dl_val_ts_ = Y_dl_tr_[idx_tr_, :], Y_dl_tr_[idx_ts_, :]
        #print(i_fold, X_dl_val_tr_.shape, Y_dl_val_tr_.shape, X_dl_val_ts_.shape, Y_dl_val_ts_.shape)

        # Standardize dense learning dataset
        t_init = time.time()

        X_dl_val_tr_stnd_, Y_dl_val_tr_stnd_, X_dl_val_ts_stnd_, dl_scaler_ = _dense_learning_stand(X_dl_val_tr_, Y_dl_val_tr_, X_dl_val_ts_, x_dl_stnd, y_dl_stnd)
        #print(X_dl_tr_stnd_.shape, Y_dl_tr_stnd_.shape, X_dl_ts_stnd_.shape)

        # Fit sense learning - Bayesian model chain
        models_ = _fit_dense_learning(X_dl_val_tr_stnd_, Y_dl_val_tr_stnd_, Y_dl_val_tr_, W_hat_, g_dl_, thetas_, RC, DL)

        # Independent prediction with conficence intervals
        M_dl_val_ts_hat_, S2_dl_val_ts_hat_, C_dl_val_ts_hat_ = _pred_prob_dist(models_, dl_scaler_, X_dl_val_ts_stnd_, Y_dl_val_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd)
        #print(M_dl_ts_hat_.shape, S2_dl_ts_hat_.shape, C_dl_ts_hat_.shape)
        # Joint probabilistic predictions
        Y_dl_val_ts_hat_ = _joint_prob_prediction(models_, dl_scaler_, X_dl_val_ts_stnd_, Y_dl_val_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd, N_samples = 250)
        #print(Y_dl_ts_.shape, Y_dl_ts_hat_.shape)

        E_dl_val_ts_[i_fold, ...] = _baseline_det_metrics(Y_dl_val_ts_, M_dl_val_ts_hat_).to_numpy()
        P_dl_val_ts_[i_fold, ...] = _prob_metrics(Y_dl_val_ts_, M_dl_val_ts_hat_, S2_dl_val_ts_hat_, Y_dl_val_ts_hat_).to_numpy()
        MV_dl_val_ts_[i_fold, :]  = _multivariate_prob_metrics(Y_dl_val_ts_, M_dl_val_ts_hat_, C_dl_val_ts_hat_, Y_dl_val_ts_hat_).to_numpy()

        i_fold += 1

    #E_sl_val_ts_  = pd.DataFrame(np.mean(E_sl_val_ts_, axis = 0), columns = ['RMSE', 'MAE', 'MBE'], index = ['NP15', 'SP15', 'ZP26'])
    E_dl_val_ts_  = pd.DataFrame(np.mean(E_dl_val_ts_, axis = 0), columns = ['RMSE', 'MAE', 'MBE'], index = ['NP15', 'SP15', 'ZP26'])
    P_dl_val_ts_  = pd.DataFrame(np.mean(P_dl_val_ts_, axis = 0), columns = ['IIS', 'CRPS', 'sample_ISS'], index = ['NP15', 'SP15', 'ZP26'])
    MV_dl_val_ts_ = pd.DataFrame(np.mean(MV_dl_val_ts_, axis = 0), columns = [''], index = ['IS', 'ES', 'VS', 'sample_IS']).T

    #meta_ = pd.DataFrame([i_exp, SL, x_sl_stnd, y_sl_stnd, theta_, time.time() - t_init], index = ['experiment', 'sparse_method', 'x_std', 'y_std', 'parameters', 'time']).T
    meta_ = pd.DataFrame([i_exp, SL, x_sl_stnd, y_sl_stnd, DL, x_dl_stnd, y_dl_stnd, theta_, time.time() - t_init],
                         index = ['experiment', 'sparse_method', 'sparse_x_std', 'sparse_y_std', 'dense_method', 'dense_x_std', 'dense_y_std', 'parameters', 'time']).T

    #R_sl_val_ts_.append(pd.concat([meta_, _flatten_DataFrame(E_sl_val_ts_)], axis = 1))
    R_dl_val_ts_.append(pd.concat([meta_, _flatten_DataFrame(P_dl_val_ts_), _flatten_DataFrame(E_dl_val_ts_), _flatten_DataFrame(MV_dl_val_ts_)], axis = 1))

#_combine_parallel_results(comm, pd.concat(R_sl_val_ts_, axis = 0), i_job, N_jobs, path_to_rst, file_name = 'val_sparse_learning.csv')
_combine_parallel_results(comm, pd.concat(R_dl_val_ts_, axis = 0), i_job, N_jobs, path_to_rst, file_name = 'val_{}.csv'.format(dl_method))
