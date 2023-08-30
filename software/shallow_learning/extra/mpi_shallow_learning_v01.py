import sys, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, ElasticNet, OrthogonalMatchingPursuit
from sklearn.linear_model import Ridge, ARDRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product

from utils import *

from group_lasso import GroupLasso
GroupLasso.LOG_LOSSES = True

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

# Parallelize experiment combinations
def _split_parameters_sets_into_jobs(i_job, N_jobs, N_thetas):
    return [np.linspace(0, N_thetas - 1, N_thetas, dtype = int)[i_job::N_jobs] for i_job in range(N_jobs)][i_job]

# Get Grid Dimensions
N = 104
M = 88

# Assets in resources: {load:5, solar:3, wind:2}
i_resource = 1
# Resources: [{MWD, PGE, SCE, SDGE, VEA}, {NP, SP, ZP}, {NP, SP}]
#i_asset = 1

# Cross-validation configuration
N_kfolds = 3
# Spatial masks {All, US Land, All CAISO Resources density, Resource density}
i_mask = 3
tau    = 0.
N_lags = 6
# Autoregressive series from today:
AR = int(sys.argv[1])
# Cyclo-stationary from lagged series
CS = int(sys.argv[2])
# Time dammy variables
TM = int(sys.argv[3])
# Recursive forecast in covariates
RC = int(sys.argv[4])
# Sparse learning model standardization
x_sl_stnd = int(sys.argv[5])
y_sl_stnd = int(sys.argv[6])
# Dense learning model standardization
x_dl_stnd = int(sys.argv[7])
y_dl_stnd = int(sys.argv[8])
# Sparse learning model index
SL = int(sys.argv[9])
# Dense learning model index
DL = int(sys.argv[10])
# Define identification experiment key
key = '{}{}_{}{}{}{}_{}{}_{}{}_{}{}_gpytorch'.format(N_lags, i_mask, AR, CS, TM, RC, x_sl_stnd, y_sl_stnd, x_dl_stnd, y_dl_stnd, SL, DL)
print(key)

# MPI job variables
i_job, N_jobs, comm = _get_node_info()
# Load the index of US land in the NOAA operational forecast
US_land_ = pd.read_pickle(path_to_aux + r"USland_0.125_(-125,-112)_(32,43).pkl")
# Load the index of Demand, Solar, and Wind land in the NOAA operational forecast
D_den_, S_den_, W_den_ = pd.read_pickle(path_to_aux + r"density_grid_0.125_(-125,-112)_(32,43).pkl")
print(US_land_.shape, D_den_.shape, S_den_.shape, W_den_.shape)
# Define spatial feature masks
M_ = [np.ones(US_land_.shape), US_land_, D_den_ + S_den_ + W_den_, [D_den_, S_den_, W_den_][i_resource]]

# Load proposed data
data_ = _load_data_in_chunks([2023], path_to_pds)
#print(len(data_))

# Define data structure for a given experiment
Y_ac_, Y_fc_, X_ac_, X_fc_, Z_, g_sl_, g_dl_ = _structure_dataset(data_, i_resource, M_[i_mask], tau)
#print(Y_ac_.shape, Y_fc_.shape, X_ac_.shape, X_fc_.shape, Z_.shape, g_sl_.shape, g_dl_.shape)
#Y_ac_ = Y_ac_[..., i_asset][..., np.newaxis]
#Y_fc_ = Y_fc_[..., i_asset][..., np.newaxis]
#print(Y_ac_.shape, Y_fc_.shape)
del data_

# Generate spare learning dataset
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
gammas_  = [1., 10., 100., 500., 1000.]
etas_    = [0.25, 0.5, 0.75, 1.]
lambdas_ = [0.1, 1., 5., 10., 50., 100., 1000., 10000.]
xis_     = [0, 1, 4, 5, 6] # ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']

# Get combination of possible parameters
thetas_, N_thetas = _get_cv_param(alphas_, betas_, omegas_, gammas_, etas_, lambdas_, xis_, SL, DL)

# Initialize parameters error metric variables
E_sl_theta_ = np.zeros((N_thetas, Y_dl_tr_.shape[1], 3))
E_dl_theta_ = np.zeros((N_thetas, Y_dl_tr_.shape[1], 3))
B_dl_theta_ = np.zeros((N_thetas, Y_dl_tr_.shape[1], 2))

# Parallelize over possible parameters combinations
for i_theta in _split_parameters_sets_into_jobs(i_job, N_jobs, N_thetas):

    # Initialize cross-validation error metric variables
    E_sl_val_ = np.zeros((N_kfolds, Y_dl_tr_.shape[1], 3))
    E_dl_val_ = np.zeros((N_kfolds, Y_dl_tr_.shape[1], 3))
    B_dl_val_ = np.zeros((N_kfolds, Y_dl_tr_.shape[1], 2))

    # Initialize iteration counter and validation score matrices
    i_fold   = 0
    t_sl_end = 0
    t_dl_end = 0
    # Loop over validation k-folds
    for idx_tr_, idx_ts_ in KFold(n_splits     = N_kfolds,
                                  random_state = None,
                                  shuffle      = False).split(X_dl_tr_):

        # Split spare learning validation partition in training and testing set
        X_sl_val_tr_, X_sl_val_ts_ = X_sl_tr_[idx_tr_, ...], X_sl_tr_[idx_ts_, ...]
        Y_sl_val_tr_, Y_sl_val_ts_ = Y_sl_tr_[idx_tr_, ...], Y_sl_tr_[idx_ts_, ...]
        #print(X_sl_val_tr_.shape, Y_sl_val_tr_.shape, X_sl_val_ts_.shape, Y_sl_val_ts_.shape)

        # Generate spase learning training and testing dataset
        X_sl_val_tr_, Y_sl_val_tr_ = _sparse_learning_dataset(X_sl_val_tr_, Y_sl_val_tr_)
        X_sl_val_ts_, Y_sl_val_ts_ = _sparse_learning_dataset(X_sl_val_ts_, Y_sl_val_ts_)
        #print(X_sl_val_tr_.shape, y_sl_val_tr_.shape, X_sl_val_ts_.shape, y_sl_val_ts_.shape)

        # Standardize spase learning dataset
        X_sl_val_tr_stnd_, Y_sl_val_tr_stnd_, X_sl_val_ts_stnd_, sl_scaler_ = _spare_learning_stand(X_sl_val_tr_, Y_sl_val_tr_, X_sl_val_ts_, x_sl_stnd, y_sl_stnd)
        #print(X_sl_val_tr_.shape, y_sl_val_tr_.shape, X_sl_val_ts_.shape)

        # Initialization prediction mean and weights
        W_hat_ = np.zeros((X_sl_val_ts_stnd_[:, g_sl_ != g_sl_[-1]].shape[1], Y_sl_val_ts_.shape[1]))

        if SL != 4:
            Y_sl_val_ts_hat_ = np.zeros(Y_sl_val_ts_.shape)
            t_sl_init        = time.time()
            # Train independent multi-task models for each hour
            for tsk in range(Y_sl_val_ts_hat_.shape[1]):
            # Lasso (linear regression with l_1 norm applied to the coefficients.)
                print(tsk, X_sl_val_tr_stnd_.shape, Y_sl_val_tr_stnd_.shape, Y_sl_val_ts_hat_.shape)
                if SL == 0: _SL = Lasso(alpha    = thetas_[i_theta][0],
                                        max_iter = 2000,
                                        tol      = 0.001).fit(X_sl_val_tr_stnd_, Y_sl_val_tr_stnd_[:, tsk])

                # Orthogonal Matching Persuit (linear regression with l_0 norm applied to the coefficients.)
                if SL == 1: _SL = OrthogonalMatchingPursuit(n_nonzero_coefs = thetas_[i_theta][0],
                                                            normalize       = False).fit(X_sl_val_tr_stnd_, Y_sl_val_tr_stnd_[:, tsk])

                # Elastic net (linear regression with l_1 and l_2 norm apply to coefficients)
                if SL == 2: _SL = ElasticNet(alpha    = thetas_[i_theta][0],
                                             l1_ratio = thetas_[i_theta][1],
                                             max_iter = 2000,
                                             tol      = 0.001).fit(X_sl_val_tr_stnd_, Y_sl_val_tr_stnd_[:, tsk])

                # Group lasso (linear regression with l_1 norm apply to coefficients and regularization applied coefficients by group)
                if SL == 3: _SL = GroupLasso(groups          = g_sl_,
                                             l1_reg          = thetas_[i_theta][1]*thetas_[i_theta][0],
                                             group_reg       = (1 - thetas_[i_theta][1])*thetas_[i_theta][0],
                                             n_iter          = 1000,
                                             scale_reg       = "inverse_group_size",
                                             supress_warning = True,
                                             tol             = 0.001).fit(X_sl_val_tr_stnd_, Y_sl_val_tr_stnd_[:, tsk])

                # Spare learning single-task prediction and optimal model coefficients
                Y_sl_val_ts_hat_[:, tsk], W_hat_[:, tsk] = _sparse_learning_predict(_SL, X_sl_val_ts_stnd_, g_sl_)
                #print(Y_sl_ts_hat_.shape, W_hat_.shape)

            t_sl_end += (time.time() - t_sl_init)/N_kfolds

            # Undo standardization from sparse learning prediction
            if y_sl_stnd == 1: Y_sl_val_ts_hat_ = sl_scaler_[1].inverse_transform(Y_sl_val_ts_hat_)
            #print(Y_sl_val_ts_.shape, Y_sl_val_ts_hat_.shape)

            # Evaluate sparse learning validation deterministic error metrics
            E_sl_val_[i_fold, ...] = _sparse_det_metrics(Y_sl_val_ts_, Y_sl_val_ts_hat_)

        # Split dense learning validation partition in training and testing set
        X_dl_val_tr_, X_dl_val_ts_ = X_dl_tr_[idx_tr_, :], X_dl_tr_[idx_ts_, :]
        Y_dl_val_tr_, Y_dl_val_ts_ = Y_dl_tr_[idx_tr_, :], Y_dl_tr_[idx_ts_, :]
        #print(i_fold, X_dl_val_tr_.shape, Y_dl_val_tr_.shape, X_dl_val_ts_.shape, Y_dl_val_ts_.shape)

        # Standardize dense learning dataset
        X_dl_val_tr_, Y_dl_val_tr_, X_dl_val_ts_, dl_scaler_ = _dense_learning_stand(X_dl_val_tr_, Y_dl_val_tr_, X_dl_val_ts_, x_dl_stnd, y_dl_stnd)
        #print(X_dl_val_tr_.shape, Y_dl_val_tr_.shape, X_dl_val_ts_.shape)

        # Initialization multi-task predictive mean and standard deviation
        Y_dl_val_tr_hat_ = np.zeros(Y_dl_val_tr_.shape)
        Y_dl_val_ts_hat_ = np.zeros(Y_dl_val_ts_.shape)
        S_dl_val_tr_hat_ = np.zeros(Y_dl_val_tr_.shape)
        S_dl_val_ts_hat_ = np.zeros(Y_dl_val_ts_.shape)
        S_dl_val_noise_  = np.zeros((Y_dl_val_ts_hat_.shape[1], Y_dl_val_ts_hat_.shape[2]))

        t_dl_init = time.time()

        # Train independent multi-task models for each hour
        for tsk in range(Y_dl_val_ts_hat_.shape[1]):
            # Train an expert models for each hour
            for hrzn in range(Y_dl_val_ts_hat_.shape[2]):

                # Define training and testing recursive dataset
                X_dl_val_tr_rc_, Y_dl_val_tr_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_val_tr_, Y_dl_val_tr_, Y_dl_val_tr_hat_, g_dl_, W_hat_, RC, hrzn, tsk)
                X_dl_val_ts_rc_, Y_dl_val_ts_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_val_ts_, Y_dl_val_ts_, Y_dl_val_ts_hat_, g_dl_, W_hat_, RC, hrzn, tsk)
                print(key, i_fold, thetas_[i_theta], hrzn, tsk, X_dl_val_tr_rc_.shape, X_dl_val_ts_rc_.shape, Y_dl_val_tr_rc_.shape, Y_dl_val_ts_rc_.shape, g_dl_rc_.shape)

                # Bayesian Linear Regression with prior on the parameters
                if DL == 0: _DL = BayesianRidge(n_iter = 1000,
                                                tol    = 0.001).fit(X_dl_val_tr_rc_, Y_dl_val_tr_rc_[:, tsk])
                # Gaussian Process
                if DL == 2: _DL = GaussianProcess(X_dl_val_tr_rc_, Y_dl_val_tr_rc_[:, tsk], g_dl_rc_, xi   = thetas_[i_theta][-1],
                                                                                                      RC   = RC,
                                                                                                      hrzn = hrzn)
                # Bayesian linear regression with ARD mechanism
                if DL == 1: _DL = ARDRegression(threshold_lambda = thetas_[i_theta][-1],
                                                n_iter           = 1000,
                                                tol              = 0.001).fit(X_dl_val_tr_rc_, Y_dl_val_tr_rc_[:, tsk])
                # Dense learning single-task prediction
                Y_dl_val_tr_hat_[..., tsk, hrzn], S_dl_val_tr_hat_[..., tsk, hrzn], S_dl_val_noise_[tsk, hrzn] = _dense_learning_predict(_DL, X_dl_val_tr_rc_, DL)
                Y_dl_val_ts_hat_[..., tsk, hrzn], S_dl_val_ts_hat_[..., tsk, hrzn], S_dl_val_noise_[tsk, hrzn] = _dense_learning_predict(_DL, X_dl_val_ts_rc_, DL)

            # Undo standardization from dense learning multi-task prediction
            if y_dl_stnd == 1: Y_dl_val_tr_hat_[..., hrzn] = dl_scaler_[1][hrzn].inverse_transform(Y_dl_val_tr_hat_[..., hrzn])
            if y_dl_stnd == 1: Y_dl_val_ts_hat_[..., hrzn] = dl_scaler_[1][hrzn].inverse_transform(Y_dl_val_ts_hat_[..., hrzn])
            if y_dl_stnd == 1: S_dl_val_tr_hat_[..., hrzn] = np.sqrt(dl_scaler_[1][hrzn].var_)*S_dl_val_tr_hat_[..., hrzn]
            if y_dl_stnd == 1: S_dl_val_ts_hat_[..., hrzn] = np.sqrt(dl_scaler_[1][hrzn].var_)*S_dl_val_ts_hat_[..., hrzn]
            if y_dl_stnd == 1: S_dl_val_noise_[..., hrzn]  = np.sqrt(dl_scaler_[1][hrzn].var_)*S_dl_val_noise_[..., hrzn]

        t_dl_end += (time.time() - t_dl_init)/N_kfolds

        # Evaluate dense learning validation deterministic error metrics
        E_dl_det_              = _det_metrics(Y_dl_val_ts_, Y_dl_val_ts_hat_)
        E_dl_val_[i_fold, ...] = np.mean(E_dl_det_, axis = -1)
        # Evaluate dense learning validation probabilistic metrics
        S_dl_val_noise_        = np.repeat(S_dl_val_noise_[np.newaxis, ...], S_dl_val_ts_hat_.shape[0], axis = 0)
        E_dl_prob_             = _prob_metrics(Y_dl_val_ts_, Y_dl_val_ts_hat_, S_dl_val_ts_hat_)
        B_dl_val_[i_fold, ...] = np.mean(E_dl_prob_, axis = -1)
        # Go to the next iteration
        i_fold += 1

    # Average cross-validation errors
    e_dl_val_ = np.mean(E_dl_val_, axis = 0)
    b_dl_val_ = np.mean(B_dl_val_, axis = 0)
    #print(e_sl_val_.shape, e_dl_val_.shape, b_dl_val_.shape)

    # Save parameter combination in .csv file
    #_save_val_in_csv_file(e_dl_val_, [key, t_dl_end, i_theta, thetas_[i_theta]], i_resource, path_to_rst, 'DLDetVal.csv')
    #_save_val_in_csv_file(b_dl_val_, [key, t_dl_end, i_theta, thetas_[i_theta]], i_resource, path_to_rst, 'DLProbVal.csv')

    # Save averaged cross-validation errors for a given parameters set
    E_dl_theta_[i_theta, ...] = e_dl_val_
    B_dl_theta_[i_theta, ...] = b_dl_val_

    if SL != 4:
        E_sl_theta_[i_theta, ...] = np.mean(E_sl_val_, axis = 0)
        #_save_val_in_csv_file(E_sl_theta_[i_theta, ...], [key, t_sl_end, i_theta, thetas_[i_theta]], i_resource, path_to_rst, 'SLDetVal.csv')

print(E_dl_theta_.shape)
# Find optimal parameters
i_metric = 0
print(E_dl_theta_[..., i_metric])

i_theta_ = np.argmin(E_dl_theta_[..., i_metric], axis = 0)
t_init   = time.time()
print(np.min(E_dl_theta_[..., i_metric], axis = 0))
print(i_theta_)

for tsk in range(3):
    print(E_dl_theta_[..., i_metric][i_theta_[tsk], ...])

exit()

# Generate spase learning training and testing dataset in the correct format
X_sl_tr_test_, Y_sl_tr_test_ = _sparse_learning_dataset(X_sl_tr_, Y_sl_tr_)
X_sl_ts_test_, Y_sl_ts_test_ = _sparse_learning_dataset(X_sl_ts_, Y_sl_ts_)
#print(X_sl_tr_test_.shape, Y_sl_tr_test_.shape, X_sl_ts_test_.shape, Y_sl_ts_test_.shape)

# Standardize spase learning dataset
X_sl_tr_stnd_, Y_sl_tr_stnd_, X_sl_ts_stnd_, sl_scaler_ = _spare_learning_stand(X_sl_tr_test_, Y_sl_tr_test_, X_sl_ts_test_, x_sl_stnd, y_sl_stnd)
#print(X_sl_tr_stnd_.shape, Y_sl_tr_stnd_.shape, X_sl_ts_stnd_.shape)

# Initialization prediction mean and weights
W_hat_ = np.zeros((X_sl_ts_stnd_[:, g_sl_ != g_sl_[-1]].shape[1], Y_sl_ts_test_.shape[1]))
if SL != 4:
    Y_sl_ts_hat_test_ = np.zeros(Y_sl_ts_test_.shape)
    # Train independent multi-task models for each hour
    for tsk in range(Y_sl_ts_test_.shape[1]):
        print(tsk, X_sl_tr_stnd_.shape, Y_sl_tr_stnd_.shape, Y_sl_ts_hat_test_.shape)

        # Lasso (linear regression with l_1 norm applied to the coefficients.)
        if SL == 0: _SL = Lasso(alpha    = thetas_[i_theta_[tsk]][0],
                                max_iter = 2000,
                                tol      = 0.001).fit(X_sl_tr_stnd_, Y_sl_tr_stnd_[:, tsk])

        # Orthogonal Matching Persuit (linear regression with l_0 norm applied to the coefficients.)
        if SL == 1: _SL = OrthogonalMatchingPursuit(n_nonzero_coefs = thetas_[i_theta_[tsk]][0],
                                                    normalize       = False).fit(X_sl_tr_stnd_, Y_sl_tr_stnd_[:, tsk])

        # Elastic net (linear regression with l_1 and l_2 norm apply to coefficients)
        if SL == 2: _SL = ElasticNet(alpha    = thetas_[i_theta_[tsk]][0],
                                     l1_ratio = thetas_[i_theta_[tsk]][1],
                                     max_iter = 2000,
                                     tol      = 0.001).fit(X_sl_tr_stnd_, Y_sl_tr_stnd_[:, tsk])

        # Group lasso (linear regression with l_1 norm apply to coefficients and regularization applied coefficients by group)
        if SL == 3: _SL = GroupLasso(groups          = g_sl_,
                                     l1_reg          = thetas_[i_theta_[tsk]][1]*thetas_[i_theta_[tsk]][0],
                                     group_reg       = (1 - thetas_[i_theta_[tsk]][1])*thetas_[i_theta_[tsk]][0],
                                     n_iter          = 1000,
                                     scale_reg       = "inverse_group_size",
                                     supress_warning = True,
                                     tol             = 0.001).fit(X_sl_val_tr_stnd_, Y_sl_val_tr_stnd_[:, tsk])

        # Spare learning single-task prediction and optimal model coefficients
        Y_sl_ts_hat_test_[:, tsk], W_hat_[:, tsk] = _sparse_learning_predict(_SL, X_sl_ts_stnd_, g_sl_)
        #print(Y_sl_ts_hat_.shape, W_hat_.shape)

    # Undo standardization from sparse learning prediction
    if y_sl_stnd == 1: Y_sl_ts_hat_ = sl_scaler_[1].inverse_transform(Y_sl_ts_hat_test_)

# Standardize dense learning dataset
X_dl_tr_stnd_, Y_dl_tr_stnd_, X_dl_ts_stnd_, dl_scaler_ = _dense_learning_stand(X_dl_tr_, Y_dl_tr_, X_dl_ts_, x_dl_stnd, y_dl_stnd)
#print(X_dl_val_tr_.shape, Y_dl_val_tr_.shape, X_dl_val_ts_.shape)

# Initialization multi-task predictive mean and standard deviation
Y_dl_tr_hat_ = np.zeros(Y_dl_tr_.shape)
Y_dl_ts_hat_ = np.zeros(Y_dl_ts_.shape)
S_dl_tr_hat_ = np.zeros(Y_dl_tr_.shape)
S_dl_ts_hat_ = np.zeros(Y_dl_ts_.shape)
S_dl_noise_  = np.zeros((Y_dl_ts_.shape[1], Y_dl_ts_.shape[2]))
# Train independent multi-task models for each hour
for tsk in range(Y_dl_ts_hat_.shape[1]):
    # Train an expert models for each hour
    for hrzn in range(Y_dl_ts_hat_.shape[2]):
        # Define training and testing recursive dataset
        X_dl_tr_rc_, Y_dl_tr_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_hat_, g_dl_, W_hat_, RC, hrzn, tsk)
        X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_hat_, g_dl_, W_hat_, RC, hrzn, tsk)
        print(key, thetas_[i_theta_[tsk]], hrzn, tsk, X_dl_tr_rc_.shape, X_dl_ts_rc_.shape, Y_dl_tr_rc_.shape, Y_dl_ts_rc_.shape, g_dl_rc_.shape)
        # Bayesian Linear Regression with prior on the parameters
        if DL == 0: _DL = BayesianRidge(n_iter = 2000,
                                        tol    = 0.001).fit(X_dl_tr_rc_, Y_dl_tr_rc_[:, tsk])
        # Bayesian linear regression with ARD mechanism
        if DL == 1: _DL = ARDRegression(threshold_lambda = thetas_[i_theta_[tsk]][-1],
                                        n_iter           = 2000,
                                        tol              = 0.0001).fit(X_dl_tr_rc_, Y_dl_tr_rc_[:, tsk])
        # Gaussian Process
        if DL == 2: _DL = GaussianProcess(X_dl_tr_rc_, Y_dl_tr_rc_[:, tsk], g_dl_rc_, xi   = thetas_[i_theta_[tsk]][-1],
                                                                                      RC   = RC,
                                                                                      hrzn = hrzn)

        # Dense learning single-task prediction
        Y_dl_tr_hat_[..., tsk, hrzn], S_dl_tr_hat_[..., tsk, hrzn], S_dl_noise_[tsk, hrzn]  = _dense_learning_predict(_DL, X_dl_tr_rc_, DL)
        Y_dl_ts_hat_[..., tsk, hrzn], S_dl_ts_hat_[..., tsk, hrzn], S_dl_noise_[tsk, hrzn]  = _dense_learning_predict(_DL, X_dl_ts_rc_, DL)

    # Undo standardization from dense learning multi-task prediction
    if y_dl_stnd == 1: Y_dl_tr_hat_[..., hrzn] = dl_scaler_[1][hrzn].inverse_transform(Y_dl_tr_hat_[..., hrzn])
    if y_dl_stnd == 1: Y_dl_ts_hat_[..., hrzn] = dl_scaler_[1][hrzn].inverse_transform(Y_dl_ts_hat_[..., hrzn])
    if y_dl_stnd == 1: S_dl_tr_hat_[..., hrzn] = np.sqrt(dl_scaler_[1][hrzn].var_)*S_dl_tr_hat_[..., hrzn]
    if y_dl_stnd == 1: S_dl_ts_hat_[..., hrzn] = np.sqrt(dl_scaler_[1][hrzn].var_)*S_dl_ts_hat_[..., hrzn]
    if y_dl_stnd == 1: S_dl_noise_[..., hrzn]  = np.sqrt(dl_scaler_[1][hrzn].var_)*S_dl_noise_[..., hrzn]

t_end = time.time() - t_init

# Evaluate dense learning validation deterministic error metrics
E_dl_ = _det_metrics(Y_dl_ts_, Y_dl_ts_hat_)
E_dl_ = np.mean(E_dl_, axis = -1)
# Evaluate dense learning validation probabilistic metrics
S_dl_noise_ = np.repeat(S_dl_noise_[np.newaxis, ...], S_dl_ts_hat_.shape[0], axis = 0)
B_dl_       = _prob_metrics(Y_dl_ts_, Y_dl_ts_hat_, S_dl_ts_hat_ + S_dl_noise_)
B_dl_       = np.mean(B_dl_, axis = -1)
#print(E_sl_.shape, E_dl_.shape, B_dl_.shape)
# Save parameter combination in .csv file
#_save_test_in_csv_file(E_dl_, key, t_end, i_theta_.tolist(), [thetas_[i_theta] for i_theta in i_theta_], i_resource, path_to_rst, 'DLDetTest.csv')
#_save_test_in_csv_file(B_dl_, key, t_end, i_theta_.tolist(), [thetas_[i_theta] for i_theta in i_theta_], i_resource, path_to_rst, 'DLProbTest.csv')
