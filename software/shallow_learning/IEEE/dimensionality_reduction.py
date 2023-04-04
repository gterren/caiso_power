import pickle, glob, os, blosc, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import sleep, time
from datetime import datetime, date, timedelta

from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso, ElasticNet, OrthogonalMatchingPursuit, Lars, Ridge, LassoLars, ARDRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score

from mpi4py import MPI

path_to_pds = r"/home/gterren/caiso_power/data/datasets/"
path_to_aux = r"/home/gterren/caiso_power/data/auxiliary/"
path_to_rst = r"/home/gterren/caiso_power/results/"

# Get Grid Dimensions
N = 104
M = 88

# Get MPI node information
def _get_node_info(verbose = False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    if verbose:
        print('>> MPI: Name: {} Rank: {} Size: {}'.format(name, rank, size) )
    return int(rank), int(size), comm

# Load data in a compressed file
def _load_data_in_chunks(years_, path):
    def __load_data_in_compressed_file(file):
        with open(file, "rb") as f:
            data_ = f.read()
        return pickle.loads(blosc.decompress(data_))

    data_ = []

    # Loop over processed years
    for year in years_:
        # Find processed data from that year
        files_ = glob.glob(path + "ac_{}_*".format(year))
        # Define the maximum feasible number of chunks
        N_min_chunks = len(files_)
        # Loop over all possible chunks
        for i in range(N_min_chunks):
            X_, Y_, Z_ = [], [], []
            for j in range(N_min_chunks):
                # Load data if extis
                try:
                    file_name = path + "ac_{}_{}-{}.dat".format(year, i, j)
                    data_p_   = __load_data_in_compressed_file(file_name)
                    print(file_name)

                    X_.append(data_p_[0])
                    Y_.append(data_p_[1])
                    Z_.append(data_p_[2])

                except:
                    continue

            # Concatenate data if files existed
            if len(X_) > 0:
                X_ = np.concatenate(X_, axis = -1)
                Y_ = np.concatenate(Y_, axis = -1)
                Z_ = np.concatenate(Z_, axis = -1)
                print(X_.shape, Y_.shape, Z_.shape)

                data_.append([X_, Y_, Z_])

    return data_


def _define_data_structure(data_, x_idx_ = None, y_idx_ = None, z_idx_ = None):

    X_, Y_, Z_ = [], [], []
    for i in range(len(data_)):
        X_.append(data_[i][0][x_idx_, ...])
        Y_.append(data_[i][1][y_idx_, ...])
        Z_.append(data_[i][2][z_idx_, ...])

    X_ = np.concatenate(X_, axis = -1)
    Y_ = np.concatenate(Y_, axis = -1)
    Z_ = np.concatenate(Z_, axis = -1)

    return X_, Y_, Z_

# Training and testing Linear Support Vector
def _linear_SVR(X_tr_, X_ts_, y_tr_, params_):
    # C must be a non-negative float i.e. in [0, inf).
    epsilon, C = params_
    _SVR = LinearSVR(epsilon       = epsilon,
                     C             = C,
                     tol           = 1e-5,
                     loss          = 'squared_epsilon_insensitive',
                     fit_intercept = True,
                     dual          = False).fit(X_tr_, y_tr_)
    return _SVR.predict(X_ts_)

# Training and testing Lasso
def _lasso(X_tr_, X_ts_, y_tr_, params_):
    # alpha must be a non-negative float i.e. in [0, inf).
    alpha  = params_
    _Lasso = Lasso(alpha         = alpha,
                   tol           = 1e-5,
                   normalize     = False,
                   fit_intercept = True).fit(X_tr_, y_tr_)
    return _Lasso.predict(X_ts_)

# Training and testing Elastic Net
def _elastic_net(X_tr_, X_ts_, y_tr_, params_):
    # a * ||w||_1 + 0.5 * b * ||w||_2^2
    # alpha = a + b and l1_ratio = a / (a + b)
    # alpha =  [0, inf) l1_ratio = [0, 1]
    alpha, l1_ratio = params_
    _EN = ElasticNet(alpha         = alpha,
                     l1_ratio      = l1_ratio,
                     tol           = 1e-5,
                     fit_intercept = True,
                     normalize     = False).fit(X_tr_, y_tr_)
    return _EN.predict(X_ts_)

# Training and testing Ridge regression
def _ridge_regression(X_tr_, X_ts_, y_tr_, params_):
    # alpha must be a non-negative float i.e. in [0, inf)
    alpha  = params_
    _Ridge = Ridge(alpha         = alpha,
                   tol           = 1e-5,
                   normalize     = False,
                   fit_intercept = True).fit(X_tr_, y_tr_)
    return _Ridge.predict(X_ts_)

# Training and testing linear regression trained least-angle regression
def _lars(X_tr_, X_ts_, y_tr_, params_):
    # The machine-precision regularization in the computation of the Cholesky diagonal factors. Increase this for very ill-conditioned systems.
    eps, n_nonzero_coefs = params_
    _Lars = Lars(eps             = eps,
                 n_nonzero_coefs = n_nonzero_coefs,
                 normalize       = False,
                 fit_intercept   = True).fit(X_tr_, y_tr_)
    return _Lars.predict(X_ts_)

# Training and testing Lasso trained with least-angle algorithm
def _lasso_lars(X_tr_, X_ts_, y_tr_, params_):
    # The machine-precision regularization in the computation of the Cholesky diagonal factors. Increase this for very ill-conditioned systems.
    alpha, eps = params_
    _LL = LassoLars(alpha           = alpha,
                    eps             = eps,
                    normalize       = False,
                    fit_intercept   = True).fit(X_tr_, y_tr_)
    return _LL.predict(X_ts_)

# Training and testing Beyesilan Linear Regrssion with Automatic relevance determination
def _ARD_Bayesian_liner(X_tr_, X_ts_, y_tr_, params_):
    # The machine-precision regularization in the computation of the Cholesky diagonal factors. Increase this for very ill-conditioned systems.
    threshold_lambda = params_
    _ARD = ARDRegression(threshold_lambda = threshold_lambda,
                         tol              = 1e-5,
                         normalize        = False,
                         fit_intercept    = True).fit(X_tr_, y_tr_)
    return _ARD.predict(X_ts_)

# Training and testing Orthogonal Matching Pursuit
def _orthogonal_matching_pursuit(X_tr_, X_ts_, y_tr_, params_):
    n_nonzero_coefs = params_
    _OMP = OrthogonalMatchingPursuit(n_nonzero_coefs = n_nonzero_coefs,
                                     tol             = 1e-5,
                                     normalize       = False,
                                     fit_intercept   = True).fit(X_tr_, y_tr_)
    return _OMP.predict(X_ts_)

# Get User Parameters
resource  = int(sys.argv[1])
region    = int(sys.argv[2])
heuristic = int(sys.argv[3])
x_scaler  = int(sys.argv[4])
y_scaler  = int(sys.argv[5])
N_kfolds  = 3
r_tr      = .4

# MPI job variables
i_job, N_jobs, comm = _get_node_info(verbose = True)

# Load all chunks of data of a given list of years
data_ = _load_data_in_chunks([2020], path_to_pds)
print(len(data_))

# Load the index of US land in the NOAA operational forecast
US_land_ = pd.read_pickle(path_to_aux + r"USland_0.125_(-125,-112)_(32,43).pkl")
# Load the index of Demand, Solar, and Wind land in the NOAA operational forecast
D_den_, S_den_, W_den_ = pd.read_pickle(path_to_aux + r"density_grid_0.125_(-125,-112)_(32,43).pkl")
print(US_land_.shape, D_den_.shape, S_den_.shape, W_den_.shape)

X_, Y_, Z_ = _define_data_structure(data_, x_idx_ = [[1, 2], [6, 7, 8, 9, 10], [0, 3, 4, 6, 14]][resource],
                                           y_idx_ = [[-5, -4, -3], [-2, -1]][resource],
                                           z_idx_ = [0, 1, 2, 3])
print(X_.shape, Y_.shape, Z_.shape)

# Concatenate all covariates after each other
X_p_ = []
for d in range(X_.shape[0]):
    if heuristic == 0: X_p_.append(X_[d, ...])
    if heuristic == 1: X_p_.append(X_[d, US_land_ > 0., ...])
    if heuristic == 2: X_p_.append(X_[d, D_den_ > 0., ...])
    if heuristic == 3: X_p_.append(X_[d, S_den_ > 0., ...])
    if heuristic == 4: X_p_.append(X_[d, W_den_ > 0., ...])
    if heuristic == 5: X_p_.append(X_[d, (D_den_ + S_den_ + W_den_) > 0., ...])
X_p_ = np.concatenate(X_p_, axis = 0)

# Get regional targes
Y_p_ = Y_[region, :, :]
print(X_p_.shape, Y_p_.shape)

# Transform dataset resolution in hours instead of days
X_pp_, y_pp_ = [], []
for n in range(X_p_.shape[-1]):
    X_pp_.append(X_p_[..., n])
    y_pp_.append(Y_p_[..., n])
X_pp_ = np.concatenate(X_pp_, axis = 1).T
y_pp_ = np.concatenate(y_pp_, axis = 0)[:, np.newaxis]
print(X_pp_.shape, y_pp_.shape)

# Split Dataset in training and testing
N_samples    = X_p_.shape[0]
N_samples_tr = int(N_samples*r_tr)
N_samples_ts = N_samples - N_samples_tr
X_tr_ = X_pp_[:N_samples_tr, :]
X_ts_ = X_pp_[-N_samples_ts:, :]
y_tr_ = y_pp_[:N_samples_tr, :]
y_ts_ = y_pp_[-N_samples_ts:, :]
print(X_tr_.shape, y_tr_.shape, X_ts_.shape, y_ts_.shape)

# Generate parameters when one parameter
N_params      = 36
alpha_        = np.logspace(-5, 5, N_params)
beta_         = np.linspace(10, 1e4, N_params, dtype = int)
eta_          = np.logspace(-5, 0, N_params)
lambda_       = np.logspace(1, 10, N_params)
Lasso_params_ = [alpha_[i]  for i in range(N_params)]
Ridge_params_ = [alpha_[i]  for i in range(N_params)]
ARD_params_   = [lambda_[i] for i in range(N_params)]
OMP_params_   = [beta_[i]   for i in range(N_params)]

# Generate combinations when two parameters
N_params     = 6
alpha_       = np.logspace(-5, 5, N_params)
beta_        = np.linspace(10, 1e4, N_params, dtype = int)
epsilon_     = np.logspace(-4, 4, N_params)
eta_         = np.logspace(-5, 0, N_params)
SVR_params_  = [[alpha_[i], alpha_[j]]   for i in range(N_params) for j in range(N_params)]
EN_params_   = [[alpha_[i], eta_[j]]     for i in range(N_params) for j in range(N_params)]
Lars_params_ = [[epsilon_[i], beta_[j]]  for i in range(N_params) for j in range(N_params)]
LL_params_   = [[alpha_[i], epsilon_[j]] for i in range(N_params) for j in range(N_params)]


# Parameters K-fold Cross-Validation
E_val_ = np.zeros((N_kfolds, 5, 36, 8))
k  = 0
for idx_tr_, idx_ts_ in KFold(n_splits     = N_kfolds,
                              random_state = None,
                              shuffle      = False).split(X_tr_):

    # Split Validation partition in training and testing set
    X_val_tr_, X_val_ts_ = X_tr_[idx_tr_, :], X_tr_[idx_ts_, :]
    y_val_tr_, y_val_ts_ = y_tr_[idx_tr_, :], y_tr_[idx_ts_, :]
    print(k, X_val_tr_.shape, y_val_tr_.shape, X_val_ts_.shape, y_val_ts_.shape)

    # Standardize covariates and/or predictors
    _x_scaler = StandardScaler().fit(X_val_tr_)
    _y_scaler = StandardScaler().fit(y_val_tr_)
    print(_x_scaler.mean_.shape, _y_scaler.mean_.shape)
    if x_scaler == 1:
        X_val_tr_p_ = _x_scaler.transform(X_val_tr_)
        X_val_ts_p_ = _x_scaler.transform(X_val_ts_)
    else:
        X_val_tr_p_ = X_val_tr_.copy()
        X_val_ts_p_ = X_val_ts_.copy()
        
    if y_scaler == 1:
        y_val_tr_p_ = _y_scaler.transform(y_val_tr_)[:, 0]
    else:
        y_val_tr_p_ = y_val_tr_[:, 0].copy()
    print(X_val_tr_p_.shape, X_val_ts_p_.shape, X_val_ts_p_.shape)

    # Validate Linear Support Vector for Regression and compute RMSE error
    y_val_ts_p_hat_ = _linear_SVR(X_val_tr_p_, X_val_ts_p_, y_val_tr_p_, SVR_params_[i_job])
    if y_scaler == 1: y_val_ts_hat_ = _y_scaler.inverse_transform(y_val_ts_p_hat_[:, np.newaxis])
    E_val_[k, 0, i_job, 0] = mean_squared_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 1, i_job, 0] = mean_absolute_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 2, i_job, 0] = mean_absolute_percentage_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 3, i_job, 0] = r2_score(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 4, i_job, 0] = explained_variance_score(y_val_ts_, y_val_ts_hat_)
    # Validate Lasso and compute RMSE error
    y_val_ts_p_hat_ = _lasso(X_val_tr_p_, X_val_ts_p_, y_val_tr_p_, Lasso_params_[i_job])
    if y_scaler == 1: y_val_ts_hat_ = _y_scaler.inverse_transform(y_val_ts_p_hat_[:, np.newaxis])
    E_val_[k, 0, i_job, 1] = mean_squared_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 1, i_job, 1] = mean_absolute_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 2, i_job, 1] = mean_absolute_percentage_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 3, i_job, 1] = r2_score(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 4, i_job, 1] = explained_variance_score(y_val_ts_, y_val_ts_hat_)
    # Validate Elastic Net and compute RMSE error
    y_val_ts_p_hat_ = _elastic_net(X_val_tr_p_, X_val_ts_p_, y_val_tr_p_, EN_params_[i_job])
    if y_scaler == 1: y_val_ts_hat_ = _y_scaler.inverse_transform(y_val_ts_p_hat_[:, np.newaxis])
    E_val_[k, 0, i_job, 2] = mean_squared_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 1, i_job, 2] = mean_absolute_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 2, i_job, 2] = mean_absolute_percentage_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 3, i_job, 2] = r2_score(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 4, i_job, 2] = explained_variance_score(y_val_ts_, y_val_ts_hat_)
    # Validate Ridge Regression and compute RMSE error
    y_val_ts_p_hat_ = _ridge_regression(X_val_tr_p_, X_val_ts_p_, y_val_tr_p_, Ridge_params_[i_job])
    if y_scaler == 1: y_val_ts_hat_ = _y_scaler.inverse_transform(y_val_ts_p_hat_[:, np.newaxis])
    E_val_[k, 0, i_job, 3] = mean_squared_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 1, i_job, 3] = mean_absolute_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 2, i_job, 3] = mean_absolute_percentage_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 3, i_job, 3] = r2_score(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 4, i_job, 3] = explained_variance_score(y_val_ts_, y_val_ts_hat_)
    # Validate linear regression trained least-angle regression algoritm and compute RMSE error
    y_val_ts_p_hat_ = _lars(X_val_tr_p_, X_val_ts_, y_val_tr_p_, Lars_params_[i_job])
    if y_scaler == 1: y_val_ts_hat_ = _y_scaler.inverse_transform(y_val_ts_p_hat_[:, np.newaxis])
    E_val_[k, 0, i_job, 4] = mean_squared_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 1, i_job, 4] = mean_absolute_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 2, i_job, 4] = mean_absolute_percentage_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 3, i_job, 4] = r2_score(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 4, i_job, 4] = explained_variance_score(y_val_ts_, y_val_ts_hat_)
    # Validate Lasso trained with least-angle regression algorithm and compute RMSE error
    y_val_ts_p_hat_ = _lasso_lars(X_val_tr_p_, X_val_ts_, y_val_tr_p_, LL_params_[i_job])
    if y_scaler == 1: y_val_ts_hat_ = _y_scaler.inverse_transform(y_val_ts_p_hat_[:, np.newaxis])
    E_val_[k, 0, i_job, 5] = mean_squared_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 1, i_job, 5] = mean_absolute_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 2, i_job, 5] = mean_absolute_percentage_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 3, i_job, 5] = r2_score(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 4, i_job, 5] = explained_variance_score(y_val_ts_, y_val_ts_hat_)
    # Validate Bayesian Linear regression with automatic relevance determination and compute RMSE error
    y_val_ts_p_hat_ = _ARD_Bayesian_liner(X_val_tr_p_, X_val_ts_, y_val_tr_p_, ARD_params_[i_job])
    if y_scaler == 1: y_val_ts_hat_ = _y_scaler.inverse_transform(y_val_ts_p_hat_[:, np.newaxis])
    E_val_[k, 0, i_job, 6] = mean_squared_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 1, i_job, 6] = mean_absolute_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 2, i_job, 6] = mean_absolute_percentage_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 3, i_job, 6] = r2_score(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 4, i_job, 6] = explained_variance_score(y_val_ts_, y_val_ts_hat_)
    # Validate Orthogonal Matching Pursuit and compute RMSE error
    y_val_ts_p_hat_ = _orthogonal_matching_pursuit(X_val_tr_p_, X_val_ts_p_, y_val_tr_p_, OMP_params_[i_job])
    if y_scaler == 1: y_val_ts_hat_ = _y_scaler.inverse_transform(y_val_ts_p_hat_[:, np.newaxis])
    E_val_[k, 0, i_job, 7] = mean_squared_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 1, i_job, 7] = mean_absolute_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 2, i_job, 7] = mean_absolute_percentage_error(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 3, i_job, 7] = r2_score(y_val_ts_, y_val_ts_hat_)
    E_val_[k, 4, i_job, 7] = explained_variance_score(y_val_ts_, y_val_ts_hat_)

    # Got to next fold
    k += 1

# Average Scores from each fold
E_val_ = np.mean(E_val_, axis = 0)

# Standardize covariates and/or predictors
_x_scaler = StandardScaler().fit(X_tr_)
_y_scaler = StandardScaler().fit(y_tr_)
print(_x_scaler.mean_.shape, _y_scaler.mean_.shape)
if x_scaler == 1:
    X_tr_p_ = _x_scaler.transform(X_tr_)
    X_ts_p_ = _x_scaler.transform(X_ts_)
else:
    X_tr_p_ = X_tr_.copy()
    X_ts_p_ = X_ts_.copy()

if y_scaler == 1:
    y_tr_p_ = _y_scaler.transform(y_tr_)[:, 0]
else:
    y_tr_p_ = y_tr_[:, 0].copy()

print(X_tr_p_.shape, X_ts_p_.shape, y_tr_p_.shape)

E_ts_ = np.zeros((5, 36, 8))

# Training and Testing Linear Support Vector for Regression and compute RMSE error
y_ts_p_hat_ = _linear_SVR(X_tr_p_, X_ts_p_, y_tr_p_, SVR_params_[i_job])
if y_scaler == 1: y_ts_hat_ = _y_scaler.inverse_transform(y_ts_p_hat_[:, np.newaxis])
E_ts_[0, i_job, 0] = mean_squared_error(y_ts_, y_ts_hat_)
E_ts_[1, i_job, 0] = mean_absolute_error(y_ts_, y_ts_hat_)
E_ts_[2, i_job, 0] = mean_absolute_percentage_error(y_ts_, y_ts_hat_)
E_ts_[3, i_job, 0] = r2_score(y_ts_, y_ts_hat_)
E_ts_[4, i_job, 0] = explained_variance_score(y_ts_, y_ts_hat_)

# Training and Testing Lasso and compute RMSE error
y_ts_p_hat_ = _lasso(X_tr_p_, X_ts_p_, y_tr_p_, Lasso_params_[i_job])
if y_scaler == 1: y_ts_hat_ = _y_scaler.inverse_transform(y_ts_p_hat_[:, np.newaxis])
E_ts_[0, i_job, 1] = mean_squared_error(y_ts_, y_ts_hat_)
E_ts_[1, i_job, 1] = mean_absolute_error(y_ts_, y_ts_hat_)
E_ts_[2, i_job, 1] = mean_absolute_percentage_error(y_ts_, y_ts_hat_)
E_ts_[3, i_job, 1] = r2_score(y_ts_, y_ts_hat_)
E_ts_[4, i_job, 1] = explained_variance_score(y_ts_, y_ts_hat_)

# Training and Testing Elastic Net and compute RMSE error
y_ts_p_hat_ = _elastic_net(X_tr_p_, X_ts_p_, y_tr_p_, EN_params_[i_job])
if y_scaler == 1: y_ts_hat_ = _y_scaler.inverse_transform(y_ts_p_hat_[:, np.newaxis])
E_ts_[0, i_job, 2] = mean_squared_error(y_ts_, y_ts_hat_)
E_ts_[1, i_job, 2] = mean_absolute_error(y_ts_, y_ts_hat_)
E_ts_[2, i_job, 2] = mean_absolute_percentage_error(y_ts_, y_ts_hat_)
E_ts_[3, i_job, 2] = r2_score(y_ts_, y_ts_hat_)
E_ts_[4, i_job, 2] = explained_variance_score(y_ts_, y_ts_hat_)

# Training and Testing Ridge Regression and compute RMSE error
y_ts_p_hat_ = _ridge_regression(X_tr_p_, X_ts_p_, y_tr_p_, Ridge_params_[i_job])
if y_scaler == 1: y_ts_hat_ = _y_scaler.inverse_transform(y_ts_p_hat_[:, np.newaxis])
E_ts_[0, i_job, 3] = mean_squared_error(y_ts_, y_ts_hat_)
E_ts_[1, i_job, 3] = mean_absolute_error(y_ts_, y_ts_hat_)
E_ts_[2, i_job, 3] = mean_absolute_percentage_error(y_ts_, y_ts_hat_)
E_ts_[3, i_job, 3] = r2_score(y_ts_, y_ts_hat_)
E_ts_[4, i_job, 3] = explained_variance_score(y_ts_, y_ts_hat_)

# Training and Testing linear regression trained least-angle regression and compute RMSE error
y_ts_p_hat_ = _lars(X_tr_p_, X_ts_p_, y_tr_p_, Lars_params_[i_opt])
if y_scaler == 1: y_ts_hat_ = _y_scaler.inverse_transform(y_ts_p_hat_[:, np.newaxis])
E_ts_[0, i_job, 4] = mean_squared_error(y_ts_, y_ts_hat_)
E_ts_[1, i_job, 4] = mean_absolute_error(y_ts_, y_ts_hat_)
E_ts_[2, i_job, 4] = mean_absolute_percentage_error(y_ts_, y_ts_hat_)
E_ts_[3, i_job, 4] = r2_score(y_ts_, y_ts_hat_)
E_ts_[4, i_job, 4] = explained_variance_score(y_ts_, y_ts_hat_)

# Training and Testing lassso trained least-angle regression algorithm and compute RMSE error
y_ts_p_hat_ = _lasso_lars(X_tr_p_, X_ts_p_, y_tr_p_, LL_params_[i_opt])
if y_scaler == 1: y_ts_hat_ = _y_scaler.inverse_transform(y_ts_p_hat_[:, np.newaxis])
E_ts_[0, i_job, 5] = mean_squared_error(y_ts_, y_ts_hat_)
E_ts_[1, i_job, 5] = mean_absolute_error(y_ts_, y_ts_hat_)
E_ts_[2, i_job, 5] = mean_absolute_percentage_error(y_ts_, y_ts_hat_)
E_ts_[3, i_job, 5] = r2_score(y_ts_, y_ts_hat_)
E_ts_[4, i_job, 5] = explained_variance_score(y_ts_, y_ts_hat_)

# Training and Testing Bayesian liner regression with automatic relevance determination
y_ts_p_hat_ = _ARD_Bayesian_liner(X_tr_p_, X_ts_p_, y_tr_p_, ARD_params_[i_opt])
if y_scaler == 1: y_ts_hat_ = _y_scaler.inverse_transform(y_ts_p_hat_[:, np.newaxis])
E_ts_[0, i_job, 6] = mean_squared_error(y_ts_, y_ts_hat_)
E_ts_[1, i_job, 6] = mean_absolute_error(y_ts_, y_ts_hat_)
E_ts_[2, i_job, 6] = mean_absolute_percentage_error(y_ts_, y_ts_hat_)
E_ts_[3, i_job, 6] = r2_score(y_ts_, y_ts_hat_)
E_ts_[4, i_job, 6] = explained_variance_score(y_ts_, y_ts_hat_)

# Training and Testing Orthogonal Matching Pursuit and compute RMSE error
y_ts_p_hat_ = _orthogonal_matching_pursuit(X_tr_p_, X_ts_p_, y_tr_p_, OMP_params_[i_job])
if y_scaler == 1: y_ts_hat_ = _y_scaler.inverse_transform(y_ts_p_hat_[:, np.newaxis])
E_ts_[0, i_job, 7] = mean_squared_error(y_ts_, y_ts_hat_)
E_ts_[1, i_job, 7] = mean_absolute_error(y_ts_, y_ts_hat_)
E_ts_[2, i_job, 7] = mean_absolute_percentage_error(y_ts_, y_ts_hat_)
E_ts_[3, i_job, 7] = r2_score(y_ts_, y_ts_hat_)
E_ts_[4, i_job, 7] = explained_variance_score(y_ts_, y_ts_hat_)
#
# # Combine results from all jobs
# if i_job != 0:
#     # Send to the source node
#     comm.send(E_val_[i_job, :], dest = 0,
#                                 tag  = 1)
#     # Send to the source node
#     comm.send(E_ts_[:, i_job, :], dest = 0,
#                                   tag  = 11)
# else:
#     # Collect data from source nodes
#     for i in range(1, N_jobs):
#         E_val_[i, :] = comm.recv(source = i,
#                                  tag    = 1)
#
#     # Collect data from source nodes
#     for i in range(1, N_jobs):
#         E_ts_[:, i, :] = comm.recv(source = i,
#                                    tag    = 11)

for i in range(E_ts_.shape[-1]):
    # Save pickle file
    np.savetxt(path_to_rst + r"CV_{}_{}-{}-{}_{}-{}.pkl".format(i, resource, region, heuristic, x_scaler, y_scaler), E_val_[..., i], delimiter = ",")
    np.savetxt(path_to_rst + r"TS_{}_{}-{}-{}_{}-{}.pkl".format(i, resource, region, heuristic, x_scaler, y_scaler), E_ts_[..., i], delimiter = ",")
