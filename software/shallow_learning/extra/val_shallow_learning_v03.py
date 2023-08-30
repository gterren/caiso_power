import sys, time, math

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

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

# Parallelize experiment combinations
def _split_parameters_set_into_jobs(i_job, N_jobs, N_thetas):
    return [np.linspace(0, N_thetas - 1, N_thetas, dtype = int)[i_job::N_jobs] for i_job in range(N_jobs)][i_job]


# Parallelize experiment combinations in batches
def _split_parameters_set_into_jobs_per_batch(i_job, N_jobs, i_batch, N_batches, N_thetas):
    N_thetas_per_batch = math.ceil(N_thetas/N_batches)
    N_thetas_per_job   = math.ceil(N_thetas_per_batch/N_jobs)
    # Get thetas indexes in Batch
    idx_thetas_in_batch_ = [np.arange(N_thetas, dtype = int)[i*N_thetas_per_batch:(i + 1)*N_thetas_per_batch] for i in range(N_batches)][i_batch]
    # Get thetas indexes in Job
    idx_thetas_in_job_   = [idx_thetas_in_batch_[i*N_thetas_per_job:(i + 1)*N_thetas_per_job] for i in range(N_jobs)][i_job]
    return idx_thetas_in_job_

# Get Grid Dimensions
N = 104
M = 88
# Assets in resources: {load:5, solar:3, wind:2}
i_resources_ = [1]
# Resources: [{MWD, PGE, SCE, SDGE, VEA}, {NP, SP, ZP}, {NP, SP}]
i_assets_ = [[1, 2, 3], [0, 1, 2], [0, 1]]
# Cross-validation configuration
N_kfolds = 5
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
# Dense learning model index
DL = int(sys.argv[6])
# Split Cross-validation in Batches
i_batch   = int(sys.argv[7])
N_batches = 6

# Define identification experiment key
key = '{}{}_{}{}{}{}_{}{}_{}{}_{}{}'.format(N_lags, i_mask, AR, CS, TM, RC, x_sl_stnd, y_sl_stnd, x_dl_stnd, y_dl_stnd, SL, DL)
print(key)

# MPI job variables
i_job, N_jobs, comm = _get_node_info()
print(i_job, N_jobs)

# Load the index of US land in the NOAA operational forecast
US_land_ = pd.read_pickle(path_to_aux + r"USland_0.125_(-125,-112)_(32,43).pkl")
# Load the index of Demand, Solar, and Wind land in the NOAA operational forecast
D_den_, S_den_, W_den_ = pd.read_pickle(path_to_aux + r"density_grid_0.125_(-125,-112)_(32,43).pkl")
print(US_land_.shape, D_den_.shape, S_den_.shape, W_den_.shape)
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
#Y_ac_, Y_fc_, X_ac_, X_fc_, Z_, g_sl_, g_dl_ = _structure_dataset(data_, i_resource, M_[i_mask], tau)
Y_ac_, Y_fc_, X_ac_, X_fc_, Z_, g_sl_, g_dl_, assets_ = _multisource_structure_dataset(data_, i_resources_, i_assets_, M_[i_mask], tau)
#print(Y_ac_.shape, Y_fc_.shape, X_ac_.shape, X_fc_.shape, Z_.shape, g_sl_.shape, g_dl_.shape)
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
xis_     = [0, 1, 4, 5, 6, 7] # ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']

# Get combination of possible parameters
thetas_, N_thetas = _get_cv_param(alphas_, betas_, omegas_, gammas_, etas_, lambdas_, xis_, SL, DL)

# Initialize parameters error metric variables
S_dl_theta_ = np.zeros((N_thetas, 7))

# Parallelize over possible parameters combinations
for i_theta in _split_parameters_set_into_jobs_per_batch(i_job, N_jobs, i_batch, N_batches, N_thetas):
    print(i_theta, N_thetas, thetas_[i_theta])

    # Initialize iteration counter and validation score matrices
    i_theta_  = [i_theta, i_theta, i_theta]
    i_fold    = 0
    S_dl_val_ = np.zeros((N_kfolds, 7))
    t_init    = time.time()
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
        X_sl_val_tr_stnd_, Y_sl_val_tr_stnd_, X_sl_val_ts_stnd_, sl_scaler_ = _spare_learning_stand(X_sl_val_tr_, Y_sl_val_tr_, X_sl_val_ts_,
                                                                                                    x_sl_stnd, y_sl_stnd)
        #print(X_sl_val_tr_stnd_.shape, Y_sl_val_tr_stnd_.shape, X_sl_val_ts_stnd_.shape, Y_sl_val_ts_.shape)

        # Fit sparse learning model
        W_hat_ = _fit_sparse_learning(X_sl_val_tr_stnd_, X_sl_val_ts_stnd_, Y_sl_val_tr_stnd_, Y_sl_val_ts_, g_sl_,
                                      thetas_, i_theta_, SL, y_sl_stnd)
        # Split dense learning validation partition in training and testing set
        X_dl_val_tr_, X_dl_val_ts_ = X_dl_tr_[idx_tr_, :], X_dl_tr_[idx_ts_, :]
        Y_dl_val_tr_, Y_dl_val_ts_ = Y_dl_tr_[idx_tr_, :], Y_dl_tr_[idx_ts_, :]
        #print(i_fold, X_dl_val_tr_.shape, Y_dl_val_tr_.shape, X_dl_val_ts_.shape, Y_dl_val_ts_.shape)

        # Standardize dense learning dataset
        X_dl_val_tr_stnd_, Y_dl_val_tr_stnd_, X_dl_val_ts_stnd_, dl_scaler_ = _dense_learning_stand(X_dl_val_tr_, Y_dl_val_tr_, X_dl_val_ts_,
                                                                                                    x_dl_stnd, y_dl_stnd)
        #print(X_dl_val_tr_stnd_.shape, Y_dl_val_tr_stnd_.shape, Y_dl_val_tr_.shape)

        # Fit Dense Learning Bayesian Model Chain
        models_ = _fit_dense_learning(X_dl_val_tr_stnd_, Y_dl_val_tr_stnd_, Y_dl_val_tr_, W_hat_, g_dl_, thetas_, i_theta_, RC, DL)

        # Make probabilistic prediction
        M_dl_val_ts_hat_, S2_dl_val_ts_hat_ = _pred_prob_dist(models_, dl_scaler_, X_dl_val_ts_stnd_, Y_dl_val_ts_, W_hat_, g_dl_, thetas_, i_theta_, RC, DL, y_dl_stnd)

        # Joint probabilistic prediction
        Y_dl_val_ts_hat_ = _joint_prob_prediction(models_, dl_scaler_, X_dl_val_ts_stnd_, Y_dl_val_ts_, W_hat_, g_dl_, thetas_, i_theta_, RC, DL, y_dl_stnd, N_samples = 100)

        # Evaluate Continuous Ranked Probability Score
        CRPS_dl_ = np.mean(_CRPS(Y_dl_val_ts_, Y_dl_val_ts_hat_), axis = -1)
        # Evaluate energy score
        ES_dl_ = _ES(Y_dl_val_ts_, Y_dl_val_ts_hat_)
        # Evaluate multivariate variogram score
        VS_dl_ = _VS(Y_dl_val_ts_, Y_dl_val_ts_hat_, p = .5)
        # Evaluate ignorance score
        IGS_dl_ = _IGS(Y_dl_val_ts_, M_dl_val_ts_hat_, S2_dl_val_ts_hat_)

        # Evaluate dense learning validation deterministic error metrics
        E_dl_det_  = _det_metrics(Y_dl_val_ts_, M_dl_val_ts_hat_)
        e_dl_det_ = np.mean(np.mean(E_dl_det_, axis = -1), axis = 0)

        S_dl_val_[i_fold, 0] = np.mean(np.mean(CRPS_dl_, axis = -1))
        S_dl_val_[i_fold, 1] = np.mean(np.mean(ES_dl_, axis = -1))
        S_dl_val_[i_fold, 2] = np.mean(np.mean(VS_dl_, axis = -1))
        S_dl_val_[i_fold, 3] = np.mean(IGS_dl_)
        S_dl_val_[i_fold, 4] = e_dl_det_[0]
        S_dl_val_[i_fold, 5] = e_dl_det_[1]
        S_dl_val_[i_fold, 6] = e_dl_det_[2]

        # Go to the next iteration
        i_fold += 1

    t_end = time.time() - t_init

    # Save averaged cross-validation errors for a given parameters set
    S_dl_theta_[i_theta, ...] = np.mean(S_dl_val_, axis = 0)

    # Save parameter combination in .csv file
    _save_val_in_csv_file(S_dl_theta_[i_theta, ...], [key, t_end, i_theta, thetas_[i_theta]], assets_, path_to_rst, 'Val.csv')

    # Generate spase learning training and testing dataset in the correct format
    X_sl_tr_test_, Y_sl_tr_test_ = _sparse_learning_dataset(X_sl_tr_, Y_sl_tr_)
    X_sl_ts_test_, Y_sl_ts_test_ = _sparse_learning_dataset(X_sl_ts_, Y_sl_ts_)
    #print(X_sl_tr_test_.shape, Y_sl_tr_test_.shape, X_sl_ts_test_.shape, Y_sl_ts_test_.shape)

    # Standardize spase learning dataset
    X_sl_tr_stnd_, Y_sl_tr_stnd_, X_sl_ts_stnd_, sl_scaler_ = _spare_learning_stand(X_sl_tr_test_, Y_sl_tr_test_, X_sl_ts_test_,
                                                                                    x_sl_stnd, y_sl_stnd)
    #print(X_sl_tr_stnd_.shape, Y_sl_tr_stnd_.shape, X_sl_ts_stnd_.shape, Y_sl_ts_test_.shape)

    t_init = time.time()

    # Fit sparse learning model
    W_hat_ = _fit_sparse_learning(X_sl_tr_stnd_, X_sl_ts_stnd_, Y_sl_tr_stnd_, Y_sl_ts_test_, g_sl_, thetas_, i_theta_, SL, y_sl_stnd)

    # Standardize dense learning dataset
    X_dl_tr_stnd_, Y_dl_tr_stnd_, X_dl_ts_stnd_, dl_scaler_ = _dense_learning_stand(X_dl_tr_, Y_dl_tr_, X_dl_ts_, x_dl_stnd, y_dl_stnd)
    #print(X_dl_val_tr_.shape, Y_dl_val_tr_.shape, X_dl_val_ts_.shape)

    # Fit dense dearning - Bayesian model chain
    models_ = _fit_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_sl_tr_, W_hat_, g_dl_, thetas_, i_theta_, RC, DL)

    # Make multitask probabilistic prediction
    M_dl_ts_hat_, S2_dl_ts_hat_ = _pred_prob_dist(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, thetas_, i_theta_, RC, DL, y_dl_stnd)

    # Make multitask joint probabilistic prediction
    Y_dl_ts_hat_ = _joint_prob_prediction(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, thetas_, i_theta_, RC, DL, y_dl_stnd, N_samples = 100)

    t_end = time.time() - t_init

    # Evaluate Continuous Ranked Probability Score
    CRPS_dl_ = np.mean(np.mean(_CRPS(Y_dl_ts_, Y_dl_ts_hat_), axis = -1), axis = 1)
    # Evaluate energy score
    ES_dl_ = np.mean(_ES(Y_dl_ts_, Y_dl_ts_hat_), axis = 1)
    # Evaluate multivariate variogram score
    VS_dl_ = np.mean(_VS(Y_dl_ts_, Y_dl_ts_hat_, p = .5), axis = 1)
    # Evaluate ignorance score
    IGS_dl_ = _IGS(Y_dl_ts_, M_dl_ts_hat_, S2_dl_ts_hat_)
    # Evaluate dense learning validation deterministic error metrics
    E_dl_ = np.mean(np.mean(_det_metrics(Y_dl_ts_, M_dl_ts_hat_), axis = -1), axis = 0)
    S_dl_ = np.array([CRPS_dl_.mean(), ES_dl_.mean(), VS_dl_.mean(), IGS_dl_.mean(), E_dl_[0], E_dl_[1], E_dl_[2]])
    # Save parameter combination in .csv file
    _save_val_in_csv_file(S_dl_, [key, t_end, i_theta, thetas_[i_theta]], assets_, path_to_rst, 'Test.csv')

    # Save parameter combination in .csv file
    _save_test_in_csv_file(CRPS_dl_, [key, t_end, i_theta, thetas_[i_theta]], assets_, path_to_rst, 'CRPSTest.csv')
    _save_test_in_csv_file(ES_dl_, [key, t_end, i_theta, thetas_[i_theta]], assets_, path_to_rst, 'ESTest.csv')
    _save_test_in_csv_file(VS_dl_, [key, t_end, i_theta, thetas_[i_theta]], assets_, path_to_rst, 'VS05Test.csv')
    _save_test_in_csv_file(IGS_dl_, [key, t_end, i_theta, thetas_[i_theta]], assets_, path_to_rst, 'IGSTest.csv')


