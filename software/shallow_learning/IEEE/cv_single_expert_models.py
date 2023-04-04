import pickle, glob, os, blosc, csv, sys

import numpy as np
import pandas as pd

from time import sleep
from datetime import datetime, date, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from sklearn.linear_model import Lasso, ElasticNet, OrthogonalMatchingPursuit, Lars, LassoLarsCV
from sklearn.linear_model import Ridge, LassoLars, ARDRegression, BayesianRidge, LassoLarsIC
from sklearn.svm import LinearSVR

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.gaussian_process.kernels import RationalQuadratic, ExpSineSquared, DotProduct

from utils import *

path_to_pds = r"/home/gterren/caiso_power/data/datasets/"
path_to_aux = r"/home/gterren/caiso_power/data/auxiliary/"
path_to_rst = r"/home/gterren/caiso_power/results/"

# Get Grid Dimensions
N = 104
M = 88

lag      = 4
r_cv     = .75
N_kfolds = 7

i_resource = 1
i_asset    = 0

fc_model         = int(sys.argv[1]) # 5 Different options
pixels_selection = int(sys.argv[2]) # 5 Different options
sparse_param     = int(sys.argv[3]) # 7 Different options
fc_param         = int(sys.argv[4]) # 7 Different options

sparse_model  = int(sys.argv[5]) # 3 Different options
covarites_set = int(sys.argv[6]) # 3 Different options
chain_model   = int(sys.argv[7]) # 2 Different options

# Training sparse linear model for feature selection
def _sparse_coefficients_model(X_tr_, X_ts_, y_tr_, y_ts_, param, model, N_params = 7):
    # Define Standardizataion functions
    _X_scaler = StandardScaler().fit(X_tr_)
    _y_scaler = StandardScaler().fit(y_tr_)
    #print(_X_scaler.mean_.shape, _y_scaler.mean_.shape)
    # Standardize dataset
    X_tr_p_ = _X_scaler.transform(X_tr_)
    X_ts_p_ = _X_scaler.transform(X_ts_)
    y_tr_p_ = _y_scaler.transform(y_tr_)[:, 0]
    y_ts_p_ = _y_scaler.transform(y_ts_)[:, 0]
    #print(X_tr_p_.shape, X_ts_p_.shape, y_tr_p_.shape, y_ts_p_.shape)
    # Apply Lasso optimized via LARS algorithm
    if model == 0:
        _SLR = LassoLars(alpha     = np.logspace(-4, 2, N_params)[param],
                         max_iter  = 1000,
                         normalize = False).fit(X_tr_, y_tr_p_)
    # Apply Lasso (linear regression with l_1 norm applied to the coefficients.)
    if model == 1:
        _SLR = Lasso(alpha     = np.logspace(-4, 2, N_params)[param],
                     max_iter  = 2000).fit(X_tr_, y_tr_p_)
    # Orthogonal Matching Puersuit (linear regression with l_0 norm applied to the coefficents.)
    if model == 2:
        _SLR = OrthogonalMatchingPursuit(n_nonzero_coefs = np.linspace(10, int(_X_scaler.mean_.shape[0]/2.), N_params, dtype = int)[param],
                                         normalize       = False).fit(X_tr_, y_tr_p_)

    if model == 3:
        _SLR = LassoLarsIC(criterion = 'aic',
                           max_iter  = 1000,
                           normalize = False).fit(X_tr_, y_tr_p_)

    if model == 4:
        _SLR = LassoLarsIC(criterion = 'bic',
                           max_iter  = 1000,
                           normalize = False).fit(X_tr_, y_tr_p_)
    return _SLR, _y_scaler.inverse_transform(_SLR.predict(X_ts_)[:, np.newaxis])[:, 0]

# Trainign expert models
def _expert_models(X_tr_, X_ts_, Y_tr_, Y_ts_, param, chain_model, model, N_params = 7):
    # Define Storage variable lists
    Y_tr_hat_chain_ = []
    Y_ts_hat_chain_ = []
    # Define prediction storage variables
    Y_ts_hat_ = np.zeros((Y_ts_.shape))
    # Loop over different forecasting horizons
    for i in range(N_hours):
        print(i)
        # Define Training and testing set
        X_tr_p_ = X_tr_[..., i]
        X_ts_p_ = X_ts_[..., i]
        y_tr_   = Y_tr_[..., i]
        y_ts_   = Y_ts_[..., i]
        #print(X_tr_p_.shape, X_ts_p_.shape, y_tr_.shape, y_ts_.shape)
        # Define Standardizataion functions
        _X_scaler = StandardScaler().fit(X_tr_p_)
        _y_scaler = StandardScaler().fit(y_tr_[:, np.newaxis])
        #print(_X_scaler.mean_.shape, _y_scaler.mean_.shape)
        # Standardize dataset
        X_tr_p_ = _X_scaler.transform(X_tr_p_)
        X_ts_p_ = _X_scaler.transform(X_ts_p_)
        y_tr_p_ = _y_scaler.transform(y_tr_[:, np.newaxis])[:, 0]
        y_ts_p_ = _y_scaler.transform(y_ts_[:, np.newaxis])[:, 0]

        if (chain_model == 1) & (len(Y_tr_hat_chain_) > 0):
            # print(X_tr_p_.shape, np.array(Y_tr_hat_chain_).shape)
            # print(X_ts_p_.shape, np.array(Y_ts_hat_chain_).shape)

            X_tr_p_chain_ = np.concatenate((X_tr_p_, np.array(Y_tr_hat_chain_).T), axis = 1)
            X_ts_p_chain_ = np.concatenate((X_ts_p_, np.array(Y_ts_hat_chain_).T), axis = 1)
            X_tr_chain_   = np.concatenate((X_tr_[..., i], np.array(Y_tr_hat_chain_).T), axis = 1)
            X_ts_chain_   = np.concatenate((X_ts_[..., i], np.array(Y_ts_hat_chain_).T), axis = 1)
        else:
            X_tr_p_chain_ = X_tr_p_.copy()
            X_ts_p_chain_ = X_ts_p_.copy()
            X_tr_chain_   = X_tr_[..., i].copy()
            X_ts_chain_   = X_ts_[..., i].copy()
        #print(X_tr_chain_.shape, X_tr_p_chain_.shape, X_ts_chain_.shape, X_ts_p_chain_.shape)
        # Lasso LARS
        if model == 0:
            _LL = LassoLars(alpha     = np.logspace(-4, 2, N_params)[param],
                            max_iter  = 1000,
                            normalize = False).fit(X_tr_chain_, y_tr_p_)
            # Predictor chain for training
            Y_tr_hat_chain_.append(_LL.predict(X_tr_chain_)[:, np.newaxis][:, 0])
            # Predictor chain for testing
            Y_ts_hat_chain_.append(_LL.predict(X_ts_chain_)[:, np.newaxis][:, 0])

        # Linear ridge regression
        if model == 1:
            _RR = Ridge(alpha = np.logspace(-5, 5, N_params)[param]).fit(X_tr_p_chain_, y_tr_p_)
            # Predictor chain for training
            Y_tr_hat_chain_.append(_RR.predict(X_tr_p_chain_))
            # Predictor chain for testing
            Y_ts_hat_chain_.append(_RR.predict(X_ts_p_chain_))

        # Bayesian linear regression with ARD mechanims
        if model == 2:
            _ARD = ARDRegression(threshold_lambda = np.logspace(1, 5, N_params)[param],
                                 n_iter           = 600,
                                 tol              = 0.001).fit(X_tr_p_chain_, y_tr_p_)
            # Predictor chain for training
            Y_tr_hat_chain_.append(_ARD.predict(X_tr_p_chain_))
            # Predictor chain for testing
            Y_ts_hat_chain_.append(_ARD.predict(X_ts_p_chain_))

        # Bayesian Linear Regression
        if model == 3:
            _BLR = BayesianRidge(n_iter = 600,
                                 tol    = 0.001).fit(X_tr_p_chain_, y_tr_p_)
            # Predictor chain for training
            Y_tr_hat_chain_.append(_BLR.predict(X_tr_p_chain_))
            # Predictor chain for testing
            Y_ts_hat_chain_.append(_BLR.predict(X_ts_p_chain_))

        if model == 4:

            # Linear kernel
            _kernel_lin  = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5)) * DotProduct(sigma_0        = 1.,
                                                                                                                 sigma_0_bounds = (1e-5, 1e5))
            # Order 2 Polynomial kernel
            _kernel_poly = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5)) * DotProduct(sigma_0        = 1.,
                                                                                                                 sigma_0_bounds = (1e-5, 1e5))**2
            # Radial basis funtions kernel
            _kernel_RBF  = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5)) * RBF(length_scale        = 1.,
                                                                                                          length_scale_bounds = (1e-5, 1e5)) + ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5))
            # Rational Quadratic kernel
            _kernel_RQ   = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5)) * RationalQuadratic(length_scale        = 1.,
                                                                                                                        alpha               = 0.1,
                                                                                                                        length_scale_bounds = (1e-5, 1e5),
                                                                                                                        alpha_bounds        = (1e-5, 1e5)) + ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5))
            # Matern Kernel with nu hyperparameter set to 0.5
            _kernel_M05  = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5)) * Matern(length_scale        = 1.0,
                                                                                                             length_scale_bounds = (1e-5, 1e5),
                                                                                                             nu                  = 0.5) + ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5))
            # Matern Kernel with nu hyperparameter set to 1.5
            _kernel_M15  = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5)) * Matern(length_scale        = 1.0,
                                                                                                             length_scale_bounds = (1e-5, 1e5),
                                                                                                             nu                  = 1.5) + ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5))
            # Matern Kernel with nu hyperparameter set to 2.5
            _kernel_M25  = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5)) * Matern(length_scale        = 1.0,
                                                                                                             length_scale_bounds = (1e-5, 1e5),
                                                                                                             nu                  = 2.5) + ConstantKernel(constant_value = 1., constant_value_bounds = (1e-5, 1e5))
            # Compile list of possible kernels
            kernels_ = [_kernel_lin, _kernel_poly, _kernel_RBF, _kernel_RQ, _kernel_M05, _kernel_M15, _kernel_M25]
            # Trainign Gaussian process for regression
            _GPR = GaussianProcessRegressor(kernel               = kernels_[param],
                                            alpha                = 1e-10,
                                            n_restarts_optimizer = 7).fit(X_tr_p_chain_, y_tr_p_)
            # Predictor chain for training
            Y_tr_hat_chain_.append(_GPR.predict(X_tr_p_chain_, return_std = False))
            # Predictor chain for testing
            Y_ts_hat_chain_.append(_GPR.predict(X_ts_p_chain_, return_std = False))

        # Gather predictions
        Y_ts_hat_[:, i] = _y_scaler.inverse_transform(Y_ts_hat_chain_[-1][:, np.newaxis])[:, 0]

    return Y_ts_hat_

# Save experiment results in csv file
def _save_results(E_, e_, key, name):
    # Save in the next row of a .csv file
    def __save_in_csv_file(data_, file_name):
        csv.writer(open(file_name, 'a')).writerow(data_)

    __save_in_csv_file(data_     = [str(x) for x in [key] + [e_[0]] + E_[:, 0].tolist()],
                       file_name = path_to_rst + r"{}_RMSE_{}-{}.csv".format(name, i_resource, i_asset))

    __save_in_csv_file(data_     = [str(x) for x in [key] + [e_[1]] + E_[:, 1].tolist()],
                       file_name = path_to_rst + r"{}_MAE_{}-{}.csv".format(name, i_resource, i_asset))

    __save_in_csv_file(data_     = [str(x) for x in [key] + [e_[2]] + E_[:, 2].tolist()],
                       file_name = path_to_rst + r"{}_MAPE_{}-{}.csv".format(name, i_resource, i_asset))

    __save_in_csv_file(data_     = [str(x) for x in [key] + [e_[3]] + E_[:, 3].tolist()],
                       file_name = path_to_rst + r"{}_MBE_{}-{}.csv".format(name, i_resource, i_asset))

    __save_in_csv_file(data_     = [str(x) for x in [key] + [e_[4]] + E_[:, 4].tolist()],
                       file_name = path_to_rst + r"{}_R2_{}-{}.csv".format(name, i_resource, i_asset))

# Load the index of US land in the NOAA operational forecast
US_land_ = pd.read_pickle(path_to_aux + r"USland_0.125_(-125,-112)_(32,43).pkl")
# Load the index of Demand, Solar, and Wind land in the NOAA operational forecast
D_den_, S_den_, W_den_ = pd.read_pickle(path_to_aux + r"density_grid_0.125_(-125,-112)_(32,43).pkl")
print(US_land_.shape, D_den_.shape, S_den_.shape, W_den_.shape)

if pixels_selection == 0: idx_sel_ = S_den_ > 0.
if pixels_selection == 1: idx_sel_ = (S_den_ + D_den_) > 0.
if pixels_selection == 2: idx_sel_ = US_land_
if pixels_selection == 3: idx_sel_ = np.ones(US_land_.shape)

# Load propossed data
data_ = _load_data_in_chunks([2019, 2020, 2021, 2022], path_to_pds)
print(len(data_))

# Define data structure for a given experiment
V_, W_, X_, Y_, Z_ = _structure_dataset(data_, i_resource, i_asset, v_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
                                                                    w_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
                                                                    x_idx_ = [[], [1, 2, -1], []],
                                                                    y_idx_ = [[], [2, 3, -1], []],
                                                                    z_idx_ = [[0], [0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 6, 7]],
                                                                    D_idx_ = idx_sel_)
del data_
print(V_.shape, W_.shape, X_.shape, Y_.shape, Z_.shape)

# Filtering out daylight hours and correct DST shifting
V_, W_, X_, Y_, Z_ = V_[5:-3, ...], W_[5:-3, ...], X_[5:-3, ...], Y_[5:-3, ...], Z_[5:-3, ...]
#V_, W_, X_, Y_, Z_ = _DST_aligning_solar_data(V_, W_, X_, Y_, Z_)
print(V_.shape, W_.shape, X_.shape, Y_.shape, Z_.shape)

# Define experiment key...
key = '{}-{}_{}-{}-{}_{}-{}-e0'.format(sparse_model, fc_model, pixels_selection, covarites_set, chain_model, sparse_param, fc_param)
print(key)

ACF_ = np.concatenate([_ACF(V_[i, :], 1) for i in range(V_.shape[0])], axis = 1)[0, :].T

# Compute the forecast baseilse
Y_per_hat_p_ = _persistence(V_)[:, lag:].T
Y_cli_hat_p_ = _climatology(V_, lag)[:, :].T
Y_cov_hat_p_ = _convex_climatology_persistence(Y_cli_hat_p_, Y_per_hat_p_, ACF_)
#print(Y_per_hat_p_.shape, Y_cli_hat_p_.shape, Y_cov_hat_p_.shape)

# Align Dataset with baseline forecats / time series
V_p_ = V_[:, lag + 1:].T
W_p_ = W_[:, lag + 1:].T
X_p_ = np.swapaxes(X_, 0, 2)[:, lag + 1:, :]
Y_p_ = np.swapaxes(Y_, 0, 2)[:, lag + 1:, :]
Z_p_ = np.swapaxes(Z_, 0, 2)[:, lag + 1:, :]
#print(V_p_.shape, W_p_.shape, X_p_.shape, Y_p_.shape, Z_p_.shape)

# Compute Dataset samples in training and testing partition
N_day_samples, N_hours = V_p_.shape
N_day_samples_tr       = int(N_day_samples*r_cv)
N_day_samples_ts       = N_day_samples - N_day_samples_tr
print(N_hours, N_day_samples, N_day_samples_tr, N_day_samples_ts)

Y_per_hat_p_tr_ = Y_per_hat_p_[:N_day_samples_tr, :]
Y_per_hat_p_ts_ = Y_per_hat_p_[-N_day_samples_ts:, :]
#print(Y_per_hat_p_tr_.shape, Y_per_hat_p_ts_.shape)
Y_cli_hat_p_tr_ = Y_cli_hat_p_[:N_day_samples_tr, :]
Y_cli_hat_p_ts_ = Y_cli_hat_p_[-N_day_samples_ts:, :]
#print(Y_cli_hat_p_tr_.shape, Y_cli_hat_p_ts_.shape)
Y_cov_hat_p_tr_ = Y_cov_hat_p_[:N_day_samples_tr, :]
Y_cov_hat_p_ts_ = Y_cov_hat_p_[-N_day_samples_ts:, :]
#print(Y_cov_hat_p_tr_.shape, Y_cov_hat_p_ts_.shape)

V_p_tr_ = V_p_[:N_day_samples_tr, :]
V_p_ts_ = V_p_[-N_day_samples_ts:, :]
#print(V_p_tr_.shape, V_p_ts_.shape)
W_p_tr_ = W_p_[:N_day_samples_tr, :]
W_p_ts_ = W_p_[-N_day_samples_ts:, :]
#print(W_p_tr_.shape, W_p_ts_.shape)
X_p_tr_ = X_p_[:, :N_day_samples_tr, :]
X_p_ts_ = X_p_[:, -N_day_samples_ts:, :]
#print(X_p_tr_.shape, X_p_ts_.shape)
Y_p_tr_ = Y_p_[:, :N_day_samples_tr, :]
Y_p_ts_ = Y_p_[:, -N_day_samples_ts:, :]
#print(Y_p_tr_.shape, Y_p_ts_.shape)
Z_p_tr_ = Z_p_[[0, 3], :N_day_samples_tr, :]
Z_p_ts_ = Z_p_[[0, 3], -N_day_samples_ts:, :]
#print(Z_p_tr_.shape, Z_p_ts_.shape)

# Form Autoregresive features
V_ar_ = []
for l in range(lag):
    V_ar_.append(V_[:, lag -l:-1 -l][..., np.newaxis])
V_ar_    = np.concatenate(V_ar_, axis = 2)
V_ar_tr_ = np.swapaxes(V_ar_[:, :N_day_samples_tr, :], 0, 2)
V_ar_ts_ = np.swapaxes(V_ar_[:, -N_day_samples_ts:, :], 0, 2)
print(V_ar_tr_.shape, V_ar_ts_.shape)
#del V_, W_, X_, Y_, Z_

# Compute Baseline Error Metrics
E_per_ = _compute_error_metrics(V_p_ts_, Y_per_hat_p_ts_)
E_cli_ = _compute_error_metrics(V_p_ts_, Y_cli_hat_p_ts_)
E_cov_ = _compute_error_metrics(V_p_ts_, Y_cov_hat_p_ts_)
E_iso_ = _compute_error_metrics(V_p_ts_, W_p_ts_)
e_per_ = np.mean(E_per_, axis = 0)
e_cli_ = np.mean(E_cli_, axis = 0)
e_cov_ = np.mean(E_cov_, axis = 0)
e_iso_ = np.mean(E_iso_, axis = 0)
#print(e_per_, e_cli_, e_cov_, e_iso_)
# Define Dataset for the structure of a global model
X_tr_, y_tr_ = _global_model_structure(Y_p_tr_, V_p_tr_)
X_ts_, y_ts_ = _global_model_structure(Y_p_ts_, V_p_ts_)
#print(X_tr_.shape, y_tr_.shape, X_ts_.shape, y_ts_.shape)

# Parameters K-fold Cross-Validation
E_val_sparse_ = np.zeros((N_kfolds, N_hours, 5))
# Initialize iteration counter
k = 0
# Loop over validation k-folds
for idx_tr_, idx_ts_ in KFold(n_splits     = N_kfolds,
                              random_state = None,
                              shuffle      = False).split(X_tr_):

    # Split Validation partition in training and testing set
    X_val_tr_, X_val_ts_ = X_tr_[idx_tr_, :], X_tr_[idx_ts_, :]
    y_val_tr_, y_val_ts_ = y_tr_[idx_tr_, :], y_tr_[idx_ts_, :]
    print(k, X_val_tr_.shape, y_val_tr_.shape, X_val_ts_.shape, y_val_ts_.shape)

    # Training sparse linear model
    _SLR, y_val_ts_hat_ = _sparse_coefficients_model(X_val_tr_, X_val_ts_, y_val_tr_, y_val_ts_, sparse_param, sparse_model)
    _, Y_val_ts_        = _revert_global_model_structure(X_val_ts_, y_val_ts_[:, 0], N_hours)
    _, Y_val_ts_hat_    = _revert_global_model_structure(X_val_ts_, y_val_ts_hat_, N_hours)
    # Compute Spare Model validation Error Metrics
    E_val_sparse_[k, ...] = _compute_error_metrics(Y_val_ts_, Y_val_ts_hat_)

    # Go to the next iteration
    k += 1

E_val_sparse_ = np.mean(E_val_sparse_, axis = 0)
e_val_sparse_ = np.mean(E_val_sparse_, axis = 0)
print(E_val_sparse_.shape, e_val_sparse_.shape)

_SLR, y_ts_hat_ = _sparse_coefficients_model(X_tr_, X_ts_, y_tr_, y_ts_, sparse_param, sparse_model)
# Revert gobal model structure to the orginal shape
_, Y_ts_hat_ = _revert_global_model_structure(X_ts_, y_ts_hat_, N_hours)
print(Y_ts_hat_.shape)

# Compute Spare Model Error Metrics
E_sparse_ = _compute_error_metrics(V_p_ts_, Y_ts_hat_)
e_sparse_ = np.mean(E_sparse_, axis = 0)
print(E_sparse_.shape, e_sparse_.shape)

# Get sparse model coefficients
idx_ = _SLR.coef_ != 0.
print(idx_.shape, idx_.sum())

# Select the set of covariantes that is going to be used in the regression
if covarites_set == 0:
    Y_pp_tr_ = np.concatenate((Y_p_tr_, V_ar_tr_, Z_p_tr_), axis = 0)
    Y_pp_ts_ = np.concatenate((Y_p_ts_, V_ar_ts_, Z_p_ts_), axis = 0)
    # Get Index of covarites that are going to be used in the regression
    idx_p_ = np.concatenate((idx_, np.array((Y_pp_tr_.shape[0] - idx_.shape[0])*[True])), axis = 0)
if covarites_set == 1:
    Y_pp_tr_ = np.concatenate((Y_p_tr_, V_ar_tr_), axis = 0)
    Y_pp_ts_ = np.concatenate((Y_p_ts_, V_ar_ts_), axis = 0)
    # Get Index of covarites that are going to be used in the regression
    idx_p_ = np.concatenate((idx_, np.array((Y_pp_tr_.shape[0] - idx_.shape[0])*[True])), axis = 0)
if covarites_set == 2:
    Y_pp_tr_ = Y_p_tr_.copy()
    Y_pp_ts_ = Y_p_ts_.copy()
    idx_p_   = idx_.copy()
print(Y_pp_tr_.shape, Y_pp_ts_.shape, V_p_tr_.shape, V_p_ts_.shape)

# Define Training and testing set
X_tr_ = np.swapaxes(Y_pp_tr_[idx_p_, ...], 0, 1)
X_ts_ = np.swapaxes(Y_pp_ts_[idx_p_, ...], 0, 1)
Y_tr_ = V_p_tr_.copy()
Y_ts_ = V_p_ts_.copy()
print(X_tr_.shape, X_ts_.shape, Y_tr_.shape, Y_ts_.shape)

# Parameters K-fold Cross-Validation
E_val_fc_ = np.zeros((N_kfolds, N_hours, 5))
# Initialize iteration counter
k = 0
# Loop over validation k-folds
for idx_tr_, idx_ts_ in KFold(n_splits     = N_kfolds,
                              random_state = None,
                              shuffle      = False).split(X_tr_):

    # Split Validation partition in training and testing set
    X_val_tr_, X_val_ts_ = X_tr_[idx_tr_, ...], X_tr_[idx_ts_, ...]
    Y_val_tr_, Y_val_ts_ = Y_tr_[idx_tr_, ...], Y_tr_[idx_ts_, ...]
    print(k, X_val_tr_.shape, Y_val_tr_.shape, X_val_ts_.shape, Y_val_ts_.shape)

    # Validate expert models
    Y_val_ts_hat_ = _expert_models(X_val_tr_, X_val_ts_, Y_val_tr_, Y_val_ts_, fc_param, chain_model, fc_model)

    # Compute Expert Models Validation Error Metrics
    E_val_fc_[k, ...] = _compute_error_metrics(Y_val_ts_, Y_val_ts_hat_)
    # Go to the next iteration
    k += 1

E_val_fc_ = np.mean(E_val_fc_, axis = 0)
e_val_fc_ = np.mean(E_val_fc_, axis = 0)
print(E_val_fc_.shape, e_val_fc_.shape)

# Traning and testing expert models
Y_ts_hat_ = _expert_models(X_tr_, X_ts_, Y_tr_, Y_ts_, fc_param, chain_model, fc_model)

# Compute Expert Models Error Metrics
E_chain_ = _compute_error_metrics(Y_ts_, Y_ts_hat_)
e_chain_ = np.mean(E_chain_, axis = 0)
print(E_chain_.shape, e_chain_.shape)

# Save sparse model validation results in pickle file
_save_results(E_val_sparse_, e_val_sparse_, key, name = r'CV-SM')
# Save sparse model testing results in pickle file
_save_results(E_sparse_, e_sparse_, key, name = r'TS-SM')
# Save expert models validation results in pickle file
_save_results(E_val_fc_, e_val_fc_, key, name = r'CV-EM')
# Save expert models testing results in pickle file
_save_results(E_chain_, e_chain_, key, name = r'TS-EM')
