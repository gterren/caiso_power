import sys, time

import numpy as np
import pandas as pd

from utils import *
from scoring_utils import *
from loading_utils import *
from aux_utils import *

# MPI job variables
i_job, N_jobs, comm = _get_node_info()

# Degine path to data
path_to_aux = r"/home/gterren/caiso_power/data/auxiliary/"
path_to_rst = r"/home/gterren/caiso_power/results/MTGPR/"
path_to_prc = r"/home/gterren/caiso_power/data/processed/"

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

R = int(sys.argv[1])
# Assets in resources: {load:5, solar:3, wind:2}
i_resources_ = [R]
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
SL = int(sys.argv[2])
DL = int(sys.argv[3])
sl_method = sl_methods_[SL]
dl_method = dl_methods_[DL]
print(sl_method, dl_method)

# Generate input and output file names
resource =  '_'.join([resources_[i_resource] for i_resource in i_resources_])
dataset  =  '_'.join(['{}-{}'.format(resources_[i_resource], '-'.join(map(str, i_assets_[i_resource]))) for i_resource in i_resources_]) + '_M{}.pkl'.format(i_mask)
print(resource, dataset)

# Define identification experiment key
i_batch   = int(sys.argv[4])
N_batches = 1
# Sparse learning model standardization
x_sl_stnd, y_sl_stnd = [[[1, 1],[0, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[1, 1],[0, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[1, 1],[1, 1],[1, 1],[1, 1],[1, 1]][DL],
                        [[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]][DL],
                        [0, 0]][SL]
# Dense learning model standardization
x_dl_stnd, y_dl_stnd = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]][DL]
# Sparse and dense learning hyper-parameters
alphas_  = [0.00001, 0.001, 0.01, 0.1, 1., 10.]
betas_   = [10, 20, 40, 80, 160, 320]
omegas_  = [0.01, 0.25, 0.5, 0.75]
gammas_  = [0.1, 1., 10.]
etas_    = [0.25, 0.5, 0.75, 1.]
lambdas_ = [1., 10., 100., 1000.]
#xis_     = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12] # ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']
xis_     = [0, 1, 4, 5, 6, 7] # ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']
kappas_1_ = [25, 50, 100, 200]
kappas_2_ = [0.1, 0.05]
kappas_3_ = [.4, .5, .6]
kappas_4_ = [.1, .2, .3, .4, .5]
kappas_   = [kappas_1_, kappas_2_, kappas_3_, kappas_4_]
# Get combination of possible parameters
exp_, N_thetas = _get_cv_param(alphas_, betas_, omegas_, gammas_, etas_, lambdas_, xis_, kappas_, SL, DL)
i_exps_        = _experiments_index_batch_job(exp_, i_batch, N_batches, i_job, N_jobs)
print(i_exps_, len(i_exps_), len(exp_), N_thetas)

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

# Y_per_fc_ts_ = Y_per_fc_ts_[..., 6:-5]
# Y_ca_fc_ts_  = Y_ca_fc_ts_[..., 6:-5]
# Y_clm_fc_ts_ = Y_clm_fc_ts_[..., 6:-5]

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
R_dl_ts_ = []

for i_exp in i_exps_:
    # Initialize constants
    theta_  = exp_[i_exp]
    thetas_ = [theta_, theta_, theta_]
    key     = r'{}-{}_{}{}-{}{}_{}'.format(sl_method, dl_method, x_sl_stnd, y_sl_stnd, x_dl_stnd, y_dl_stnd, theta_)
    print(i_job, i_exp, x_sl_stnd, y_sl_stnd, x_dl_stnd, y_dl_stnd, thetas_)

    # Standardize sparse learning dataset
    X_sl_tr_stnd_, Y_sl_tr_stnd_, X_sl_ts_stnd_, sl_scaler_ = _sparse_learning_stand(X_sl_tr_, Y_sl_tr_, X_sl_ts_, x_sl_stnd, y_sl_stnd)
    #print(X_sl_tr_stnd_.shape, Y_sl_tr_stnd_.shape, X_sl_ts_stnd_.shape)

    # Fit sparse learning model
    t_sl_tr              = time.time()
    W_hat_, Y_sl_ts_hat_ = _fit_sparse_learning(X_sl_tr_stnd_, X_sl_ts_stnd_, Y_sl_tr_stnd_, Y_sl_ts_, g_sl_, thetas_, sl_scaler_, SL, y_sl_stnd)
    t_sl_tr              = time.time() - t_sl_tr

    # Standardize dense learning dataset
    X_dl_tr_stnd_, Y_dl_tr_stnd_, X_dl_ts_stnd_, dl_scaler_ = _dense_learning_stand(X_dl_tr_, Y_dl_tr_, X_dl_ts_, x_dl_stnd, y_dl_stnd)
    #print(X_dl_val_tr_.shape, Y_dl_val_tr_.shape, X_dl_val_ts_.shape)

    # Fit multitask dense learning - Bayesian model chain
    t_dl_tr = time.time()
    models_ = _fit_multitask_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_, W_hat_, g_dl_, thetas_, RC, DL, key)
    t_dl_tr = time.time() - t_dl_tr

    # Make multitask probabilistic prediction
    t_ts                                      = time.time()
    M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_ = _multitask_pred_prob_dist(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd)
    t_ts                                      = time.time() - t_ts

    # Make multitask joint probabilistic prediction
    t_prob_ts    = time.time()
    Y_dl_ts_hat_ = _multitask_joint_prediction(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd, N_samples = 100)
    t_prob_ts    = time.time() - t_prob_ts

    print(M_dl_ts_hat_.shape, S2_dl_ts_hat_.shape, C_dl_ts_hat_.shape)
    # Testing scores
    E_dl_ts_  = _baseline_det_metrics(Y_dl_ts_, M_dl_ts_hat_)
    P_dl_ts_  = _prob_metrics(Y_dl_ts_, M_dl_ts_hat_, S2_dl_ts_hat_, Y_dl_ts_hat_)
    MV_dl_ts_ = _multivariate_prob_metrics(Y_dl_ts_, M_dl_ts_hat_, C_dl_ts_hat_, Y_dl_ts_hat_)

    N_feaures_ = [(W_hat_[:, tsk] != 0.).sum() for tsk in range(W_hat_.shape[1])]
    N_feaures  = int(np.sum(N_feaures_))
    #print(N_feaures_,N_feaures )

    meta_ = pd.DataFrame([i_exp, SL, x_sl_stnd, y_sl_stnd, DL, x_dl_stnd, y_dl_stnd, theta_, N_feaures_, N_feaures, t_sl_tr, t_dl_tr, t_ts, t_prob_ts],
                         index = ['experiment',
                                  'sparse_method',
                                  'x_sl_std',
                                  'y_sl_std',
                                  'dense_method',
                                  'x_dl_std',
                                  'y_dl_std',
                                  'parameters',
                                  'all_dimensions',
                                  'dimensions',
                                  'sparse_training_time',
                                  'dense_training_time',
                                  'testing_time',
                                  'prob_testing_time']).T

    R_dl_ts_.append(pd.concat([meta_,
                               _flatten_DataFrame(P_dl_ts_),
                               _flatten_DataFrame(E_dl_ts_),
                               _flatten_DataFrame(MV_dl_ts_)], axis = 1))

    _combine_parallel_results(comm, pd.concat(R_dl_ts_, axis = 0), i_job, N_jobs, path_to_rst, file_name = 'test-{}-{}-{}.csv'.format(resource, sl_method, dl_method))
