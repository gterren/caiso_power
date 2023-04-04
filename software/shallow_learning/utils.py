import pickle, glob, os, blosc, csv

import numpy as np

from time import sleep
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from mpi4py import MPI
from itertools import product

from GP_utils import *

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
    # Open a BLOSC compressed file
    def __load_data_in_compressed_file(file):
        with open(file, "rb") as f:
            data_ = f.read()
        return pickle.loads(blosc.decompress(data_))
    # Loop over processed years
    data_ = []
    for year in years_:
        # Find processed data from that year
        files_ = glob.glob(path + "{}_*".format(year))
        # Define the maximum feasible number of chunks
        N_min_chunks = len(files_)
        # Loop over all possible chunks
        for i in range(N_min_chunks):
            V_, W_, X_, Y_, Z_ = [], [], [], [], []
            for j in range(N_min_chunks):
                # Load data if extis
                try:
                    file_name = path + "{}_{}-{}.dat".format(year, i, j)
                    data_p_   = __load_data_in_compressed_file(file_name)
                    # Append together all chucks
                    V_.append(data_p_[0])
                    W_.append(data_p_[1])
                    X_.append(data_p_[2])
                    Y_.append(data_p_[3])
                    Z_.append(data_p_[4])
                    print(file_name)
                except:
                    continue
            # Concatenate data if files existed
            if len(X_) > 0:
                V_ = np.concatenate(V_, axis = 0)
                W_ = np.concatenate(W_, axis = 0)
                X_ = np.concatenate(X_, axis = 0)
                Y_ = np.concatenate(Y_, axis = 0)
                Z_ = np.concatenate(Z_, axis = 0)
                data_.append([V_, W_, X_, Y_, Z_])
    return data_

# v = {MWD (ac), PGE (ac), SCE (ac), SDGE (ac), VEA (ac), NP15 solar (ac), SP15 solar (ac), ZP26 solar (ac), NP15 wind (ac), SP15 wind (ac)}
# w = {MWD (fc), PGE (fc), SCE (fc), SDGE (fc), VEA (fc), NP15 solar (fc), SP15 solar (fc), ZP26 solar (fc), NP15 wind (fc), SP15 wind (fc)}
# X = {PRES (ac), DSWRF (ac), DLWRF (ac), DPT (ac), RH (ac), TMP (ac), W_10 (ac), W_60 (ac), W_80 (ac), W_100 (ac), W_120 (ac),
#      DI (ac), WC (ac), HCDH (ac), GSI (ac)}
# Y = {PRES (fc), PRATE (fc), DSWRF (fc), DLWRF (fc), DPT (fc), RH (fc), TMP (fc), W_10 (fc), W_60 (fc), W_80 (fc), W_100 (fc), W_120 (fc),
#      DI (fc), WC (fc), HCDH (fc), GSI (fc)}
# z = {year, month, day, yday, hour, weekday, weekend, isdst, holiday}
# DSWRF = Diffuse Radiation
# Is only water pumping... (?)
def _structure_dataset(data_, i_resource, F_idx_, tau, v_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
                                                       w_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
                                                       x_idx_ = [[0, 2, 3, 4, 5, 11, 12, 13], [1, 2, 14], [6, 7, 8, 9, 10]],
                                                       y_idx_ = [[0, 1, 3, 4, 5, 6, 12, 13, 14], [2, 3, 15], [7, 8, 9, 10, 11]],
                                                       z_idx_ = [[0], [0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 6, 7]]):
    v_idx_ = v_idx_[i_resource]
    w_idx_ = w_idx_[i_resource]
    x_idx_ = x_idx_[i_resource]
    y_idx_ = y_idx_[i_resource]
    z_idx_ = z_idx_[i_resource]
    #F_idx_ = F_idx_[i_resource]
    # Concatenate all chucks of data in matrix form
    V_, W_, X_, Y_, Z_ = [], [], [], [], []
    for i in range(len(data_)):
        V_.append(data_[i][0][:, v_idx_])
        W_.append(data_[i][1][:, w_idx_])
        X_.append(data_[i][2][:, x_idx_, :])
        Y_.append(data_[i][3][:, y_idx_, :])
        Z_.append(data_[i][4][:, z_idx_])
        #print(i, data_[i][0][:, v_idx_].shape, data_[i][1][:, w_idx_].shape)
    V_ = np.concatenate(V_, axis = 0)
    W_ = np.concatenate(W_, axis = 0)
    X_ = np.concatenate(X_, axis = 0)
    Y_ = np.concatenate(Y_, axis = 0)
    Z_ = np.concatenate(Z_, axis = 0)
    #print(V_.shape, W_.shape, X_.shape, Y_.shape, Z_.shape)
    # Apply features selection heuristic
    V_p_ = V_[:, :]
    W_p_ = W_[:, :]
    X_p_ = X_[..., F_idx_ > tau]
    Y_p_ = Y_[..., F_idx_ > tau]
    G_sl_ = np.concatenate([i*np.ones((X_p_.shape[-2], 1)) for i in range(X_p_.shape[-1])], axis = 1) # Group Lasso index for spatial features
    #G_sl_ = np.concatenate([i*np.ones((1, X_p_.shape[-1])) for i in range(X_p_.shape[-2])], axis = 0) # Group Lasso index for weather features
    G_dl_ = np.concatenate([i*np.ones((1, X_p_.shape[-1])) for i in range(X_p_.shape[-2])], axis = 0)
    #print(V_p_.shape, W_p_.shape, X_p_.shape, Y_p_.shape, G_sl_.shape, G_dl_.shape)
    del V_, W_, X_, Y_
    # Concatenate all the dimensions
    X_pp_, Y_pp_, G_sl_p_, G_dl_p_ = [], [], [], []
    for d in range(X_p_.shape[1]):
        X_pp_.append(X_p_[:, d, :])
        Y_pp_.append(Y_p_[:, d, :])
        G_sl_p_.append(G_sl_[d, :])
        G_dl_p_.append(G_dl_[d, :])

    X_pp_   = np.concatenate(X_pp_, axis = -1)
    Y_pp_   = np.concatenate(Y_pp_, axis = -1)
    G_sl_p_ = np.concatenate(G_sl_p_, axis = -1)
    G_dl_p_ = np.concatenate(G_dl_p_, axis = -1)
    #print(X_pp_.shape, Y_pp_.shape, G_sl_p_.shape, G_dl_p_.shape)
    del X_p_, Y_p_, G_sl_, G_dl_
    # Concatenate by hours
    V_pp_, W_pp_, X_ppp_, Y_ppp_, Z_p_ = [], [], [], [], []
    for n in range(int(V_p_.shape[0]/24)):
        k = n*24
        l = (n + 1)*24
        V_pp_.append(V_p_[k:l, ...][:,np.newaxis])
        W_pp_.append(W_p_[k:l, ...][:, np.newaxis])
        X_ppp_.append(X_pp_[k:l, ...][:, np.newaxis, :])
        Y_ppp_.append(Y_pp_[k:l, ...][:, np.newaxis, :])
        Z_p_.append(Z_[k:l, ...][:, np.newaxis, :])
    V_pp_  = np.concatenate(V_pp_, axis = 1)
    W_pp_  = np.concatenate(W_pp_, axis = 1)
    X_ppp_ = np.concatenate(X_ppp_, axis = 1)
    Y_ppp_ = np.concatenate(Y_ppp_, axis = 1)
    Z_p_   = np.concatenate(Z_p_, axis = 1)
    return V_pp_, W_pp_, X_ppp_, Y_ppp_, Z_p_, G_sl_p_, G_dl_p_

# Split Dataset in training and testing
def _training_and_testing_dataset(X_, r_tr = 0.75):
    # Compute Dataset samples in training and testing partition
    N_samples    = X_.shape[0]
    N_samples_tr = int(N_samples*r_tr)
    N_samples_ts = N_samples - N_samples_tr
    #print(N_samples, N_samples_tr, N_samples_ts)
    # Make partions
    X_tr_ = X_[:N_samples_tr, ...]
    X_ts_ = X_[-N_samples_ts:, ...]
    return X_tr_, X_ts_

# Naive and CAISO forecasts as baselines
def _naive_forecasts(Y_ac_, Y_fc_, lag):
    # Persistent Forecast
    Y_per_fc_ = Y_ac_[:, lag - 1:-2, ...]
    # CAISO Forecast
    Y_ca_fc_  = Y_fc_[:, lag + 1:, ...]
    # Climatology
    Y_clm_fc_ = np.concatenate([Y_ac_[:, lag - (l + 1):-(2 + l), ...][..., np.newaxis] for l in range(lag)], axis = -1)
    Y_clm_fc_ = np.mean(np.swapaxes(np.swapaxes(Y_clm_fc_, 0, 1), 1, 2), axis = -1)
    #print(Y_per_fc_.shape, Y_ca_fc_.shape)
    Y_ca_fc_  = np.swapaxes(np.swapaxes(Y_ca_fc_, 0, 1), -2, -1)
    Y_per_fc_ = np.swapaxes(np.swapaxes(Y_per_fc_, 0, 1), -2, -1)

    return Y_per_fc_, Y_ca_fc_, Y_clm_fc_

# Generate sparse learning dataset
def _sparse_learning_dataset(X_ac_, Y_ac_):
    # Define sparse learning regression dataset
    y_sl_ = []
    X_sl_ = []
    for i in range(Y_ac_.shape[0]):
        y_sl_.append(Y_ac_[i, ...])
        X_sl_.append(X_ac_[i, ...])
    y_sl_ = np.concatenate(y_sl_, axis = 1).T
    X_sl_ = np.concatenate(X_sl_, axis = 1).T
    #print(y_sl_.shape, X_sl_.shape)
    return X_sl_, y_sl_

# Generate dense learning dataset
def _dense_learning_dataset(X_fc_, Y_ac_, Z_, G_, lag, AR = 0, CS = 0, TM = 0):
    # Observations previous to the forecasting event
    X_ar_ = np.swapaxes(np.swapaxes(Y_ac_[:-6, lag:-1], -1, -2), -2, -3)
    # Observations from previous hours to the forecasting event
    X_cs_ = np.swapaxes(np.swapaxes(np.concatenate([Y_ac_[:, lag - (l + 1):-(2 + l), ...][..., np.newaxis] for l in range(lag)], axis = -1), -1, -2), -2, -3)
    X_cs_ = np.swapaxes(np.swapaxes(X_cs_, -1, -2), -2, -3)
    X_ar_ = np.concatenate([X_ar_[i, ...] for i in range(X_ar_.shape[0])], axis = 0)
    X_ar_ = np.swapaxes(np.concatenate([X_ar_[np.newaxis, ...] for _ in range(X_cs_.shape[0])], axis = 0), -1, -2)
    X_cs_ = np.swapaxes(np.concatenate([X_cs_[:, i, ...] for i in range(X_cs_.shape[1])], axis = 1), -1, -2)
    #print(X_ar_.shape, X_cs_.shape)
    # Adjust timestamps signal and covariates
    X_dl_ = X_fc_[:, lag + 1:, :]
    Z_dl_ = Z_[:, lag + 1:, ...]
    #print(X_dl_.shape, Z_dl_.shape)
    # Get group index for kernel learning
    g_ar_ = np.ones((X_ar_.shape[-1],))*(np.unique(G_)[-1] + 1)
    g_cs_ = np.ones((X_cs_.shape[-1],))*(np.unique(G_)[-1] + 2)
    g_tm_ = np.ones((Z_dl_.shape[-1],))*(np.unique(G_)[-1] + 3)
    #print(G_.shape, g_ar_.shape, g_cs_.shape, g_dl_.shape, G_dl_.shape)
    # Form covariate vector for dense learning
    Y_dl_ = np.swapaxes(np.swapaxes(Y_ac_[:, lag + 1:, ...], 0, 1), -2, -1)
    if AR == 1:
        X_dl_ = np.concatenate((X_dl_, X_ar_), axis = 2)
        G_    = np.concatenate([G_, g_ar_], axis = 0)
    if CS == 1:
        X_dl_ = np.concatenate((X_dl_, X_cs_), axis = 2)
        G_    = np.concatenate([G_, g_cs_], axis = 0)
    if TM == 1:
        X_dl_ = np.concatenate((X_dl_, Z_dl_), axis = 2)
        G_    = np.concatenate([G_, g_tm_], axis = 0)
    X_dl_ = np.swapaxes(np.swapaxes(X_dl_, 0, 1), -2, -1)
    #print(Y_dl_.shape, X_dl_.shape)
    #print(np.unique(G_dl_))
    return X_dl_, Y_dl_, G_

# Use sparse learning model to make a prediction and retrive model optimal parameters
def _sparse_learning_predict(_SL, X_, g_):
    # Sparse learning prediction
    y_hat_ = _SL.predict(X_)
    # Sparse learning optimal model coefficient
    idx_   = g_ != g_[-1]
    w_hat_ = _SL.coef_
    if w_hat_.ndim > 1:
        #if w_hat_.shape[0] < w_hat_.shape[1]: w_hat_ = w_hat_.T
        return y_hat_, w_hat_[idx_, 0]
    else:
        return y_hat_, w_hat_[idx_]

# Standardize spare learning dataset
def _spare_learning_stand(X_sl_tr_, y_sl_tr_, X_sl_ts_, x_stand = 0, y_stand = 0):
    X_sl_tr_p_ = X_sl_tr_.copy()
    X_sl_ts_p_ = X_sl_ts_.copy()
    y_sl_tr_p_ = y_sl_tr_.copy()
    # Define Standardization functions
    _x_sl_scaler = StandardScaler().fit(X_sl_tr_)
    _y_sl_scaler = StandardScaler().fit(y_sl_tr_)
    #print(_x_sl_scaler.mean_.shape, _y_sl_scaler.mean_.shape)
    # Standardize dataset
    if x_stand == 1: X_sl_tr_p_ = _x_sl_scaler.transform(X_sl_tr_)
    if x_stand == 1: X_sl_ts_p_ = _x_sl_scaler.transform(X_sl_ts_)
    if y_stand == 1: y_sl_tr_p_ = _y_sl_scaler.transform(y_sl_tr_)
    #print(X_sl_tr_p_.shape, X_sl_ts_p_.shape, y_sl_tr_p_.shape)
    return X_sl_tr_p_, y_sl_tr_p_, X_sl_ts_p_, [_x_sl_scaler, _y_sl_scaler]

# Standardize dense learning dataset
def _dense_learning_stand(X_dl_tr_, Y_dl_tr_, X_dl_ts_, x_stand = 0, y_stand = 0):
    X_dl_tr_p_ = X_dl_tr_.copy()
    X_dl_ts_p_ = X_dl_ts_.copy()
    Y_dl_tr_p_ = Y_dl_tr_.copy()
    x_dl_scaler_ = []
    y_dl_scaler_ = []
    for i_hour in range(24):
        # Define Standardization functions
        _x_dl_scaler = StandardScaler().fit(X_dl_tr_[..., i_hour])
        _y_dl_scaler = StandardScaler().fit(Y_dl_tr_[..., i_hour])
        x_dl_scaler_.append(_x_dl_scaler)
        y_dl_scaler_.append(_y_dl_scaler)
        #print(i_hour, _x_dl_scaler.mean_.shape, _y_dl_scaler.mean_.shape)
        # Standardize dataset
        if x_stand == 1: X_dl_tr_p_[..., i_hour] = _x_dl_scaler.transform(X_dl_tr_[..., i_hour])
        if x_stand == 1: X_dl_ts_p_[..., i_hour] = _x_dl_scaler.transform(X_dl_ts_[..., i_hour])
        if y_stand == 1: Y_dl_tr_p_[..., i_hour] = _y_dl_scaler.transform(Y_dl_tr_[..., i_hour])
    return X_dl_tr_p_, Y_dl_tr_p_, X_dl_ts_p_, [x_dl_scaler_, y_dl_scaler_]

# Continuous Rank Probability Score
def _CRPS(x_, mu_, s_):
    _N = norm(0, 1)
    a_ = (x_ - mu_)/s_
    return np.mean(s_*(1/np.sqrt(np.pi) - 2.*_N.pdf(a_) - a_*(2.*_N.cdf(a_) - 1.)))

# Root Mean Squared Error
def _RMSE(y_, y_hat_):
    return np.sqrt(np.mean((y_ - y_hat_)**2, axis = 0))

# Mean Absolute Error
def _MAE(y_, y_hat_):
    return np.mean(np.absolute(y_ - y_hat_), axis = 0)

# Mean Bias Error
def _MBE(y_, y_hat_):
    return np.mean((y_ - y_hat_), axis = 0)

# Negative Log Predictive Probability
def _NLPP(y_, y_hat_, s_hat_):
    return - np.sum([norm(y_hat_[i], s_hat_[i]).logpdf(y_[i]) for i in range(y_hat_.shape[-1])])

# Compute probabilistic scores
def _prob_metrics(Y_, Y_hat_, S_hat_):
    scores_ = []
    # Samples / Tasks / Forecasting horizons
    for tsk in range(Y_hat_.shape[1]):
        NLPP_ = np.array([_NLPP(Y_[..., tsk, hrzn], Y_hat_[..., tsk, hrzn], S_hat_[..., tsk, hrzn]) for hrzn in range(Y_.shape[-1])])[np.newaxis, :]
        CRPS_ = np.array([_CRPS(Y_[..., tsk, hrzn], Y_hat_[..., tsk, hrzn], S_hat_[..., tsk, hrzn]) for hrzn in range(Y_.shape[-1])])[np.newaxis, :]
        scores_.append(np.concatenate((NLPP_, CRPS_), axis = 0)[:, np.newaxis, :])
    return np.swapaxes(np.concatenate(scores_, axis = 1), 0, 1)

# Compute deterministic scores
def _det_metrics(Y_, Y_hat_):
    scores_ = []
    # Samples / Tasks / Forecasting horizons
    for tsk in range(Y_hat_.shape[1]):
        RMSE_ = np.array([_RMSE(Y_[..., tsk, hrzn], Y_hat_[..., tsk, hrzn]) for hrzn in range(Y_.shape[-1])])[np.newaxis, :]
        MAE_  = np.array([ _MAE(Y_[..., tsk, hrzn], Y_hat_[..., tsk, hrzn]) for hrzn in range(Y_.shape[-1])])[np.newaxis, :]
        MBE_  = np.array([ _MBE(Y_[..., tsk, hrzn], Y_hat_[..., tsk, hrzn]) for hrzn in range(Y_.shape[-1])])[np.newaxis, :]
        scores_.append(np.concatenate((RMSE_, MAE_, MBE_), axis = 0)[:, np.newaxis, :])
    return np.swapaxes(np.concatenate(scores_, axis = 1), 0, 1)

# Compute deterministic scores for sparse model
def _sparse_det_metrics(Y_, Y_hat_):
    scores_ = []
    # Samples / Tasks / Forecasting horizons
    for tsk in range(Y_hat_.shape[1]):
        scores_.append(np.array([_RMSE(Y_[..., tsk], Y_hat_[..., tsk]),
                                  _MAE(Y_[..., tsk], Y_hat_[..., tsk]),
                                  _MBE(Y_[..., tsk], Y_hat_[..., tsk])])[..., np.newaxis])

    return np.concatenate(scores_, axis = 1).T

# # Define Recursive dataset
# def _dense_learning_recursive_dataset(X_, Y_, Y_hat_, g_, W_hat_, RC, hrzn, tsk):
#     # Find 0 coefficients obtained from sparse learning model
#     #w_hat_ = np.sum(W_hat_, axis = 1)
#     w_hat_ = W_hat_[..., tsk]
#     idx_   = w_hat_ != 0.
#     #print(idx_.sum(), w_hat_.shape[0], W_hat_.shape)
#     if RC:
#         # Form recursive dataset and add feature sources indexes
#         #Y_hat_rc_ = np.concatenate([Y_hat_[..., tsk, :hrzn] for tsk in range(Y_hat_.shape[1])], axis = 1)
#         Y_hat_rc_ = Y_hat_[..., tsk, :hrzn]
#         X_rc_     = np.concatenate([X_[:, :w_hat_.shape[0], hrzn][:, idx_], X_[:, w_hat_.shape[0]:, hrzn], Y_hat_rc_], axis = 1)
#         g_rc_     = np.concatenate([g_[:w_hat_.shape[0]][idx_], g_[w_hat_.shape[0]:], np.ones((Y_hat_rc_.shape[1],))*(np.unique(g_)[-1] + 1)], axis = 0)
#         #print(Y_hat_rc_.shape, X_.shape, X_rc_.shape, g_.shape, g_rc_.shape)
#     else:
#         X_rc_ = np.concatenate([X_[:, :w_hat_.shape[0], hrzn][:, idx_], X_[:, w_hat_.shape[0]:, hrzn]], axis = 1)
#         g_rc_ = np.concatenate([g_[:w_hat_.shape[0]][idx_], g_[w_hat_.shape[0]:]], axis = 0)
#
#     return X_rc_, Y_[..., hrzn], g_rc_

# Define Recursive dataset
def _dense_learning_recursive_dataset(X_, Y_, Y_hat_, g_, W_hat_, RC, hrzn, tsk = None):
    # Find 0 coefficients obtained from sparse learning model
    if tsk == None: idx_ = np.sum(W_hat_, axis = 1) != 0.
    else:           idx_ = W_hat_[..., tsk] != 0.
    if RC:
        # Form recursive dataset and add feature sources indexes
        if tsk == None: Y_hat_rc_ = np.concatenate([Y_hat_[..., tsk, :hrzn] for tsk in range(Y_hat_.shape[1])], axis = 1)
        else:           Y_hat_rc_ = Y_hat_[..., tsk, :hrzn]
        X_rc_ = np.concatenate([X_[:, :W_hat_.shape[0], hrzn][:, idx_], X_[:, W_hat_.shape[0]:, hrzn], Y_hat_rc_], axis = 1)
        g_rc_ = np.concatenate([g_[:W_hat_.shape[0]][idx_], g_[W_hat_.shape[0]:], np.ones((Y_hat_rc_.shape[1],))*(np.unique(g_)[-1] + 1)], axis = 0)
        #print(Y_.shape, Y_hat_rc_.shape, X_.shape, X_rc_.shape, g_.shape, g_rc_.shape)
    else:
        X_rc_ = np.concatenate([X_[:, :W_hat_.shape[0], hrzn][:, idx_],
                                X_[:, W_hat_.shape[0]:, hrzn]], axis = 1)
        g_rc_ = np.concatenate([g_[:W_hat_.shape[0]][idx_], g_[W_hat_.shape[0]:]], axis = 0)
        #print(Y_.shape, X_.shape, X_rc_.shape, g_.shape, g_rc_.shape)
    return X_rc_, Y_[..., hrzn], g_rc_

# Define Recursive dataset
def _dense_learning_recursive_dataset(X_, Y_, Y_hat_, g_, W_hat_, RC, hrzn, tsk = None):
    # Find 0 coefficients obtained from sparse learning model
    if tsk == None: idx_ = np.sum(W_hat_, axis = 1) != 0.
    else:           idx_ = W_hat_[..., tsk] != 0.
    if RC:
        # Form recursive dataset and add feature sources indexes
        if tsk == None: Y_hat_rc_ = np.concatenate([Y_hat_[..., tsk, :hrzn] for tsk in range(Y_hat_.shape[1])], axis = 1)
        else:           Y_hat_rc_ = Y_hat_[..., tsk, :hrzn]
        X_rc_ = np.concatenate([X_[:, :W_hat_.shape[0], hrzn][:, idx_], X_[:, W_hat_.shape[0]:, hrzn], Y_hat_rc_], axis = 1)
        g_rc_ = np.concatenate([g_[:W_hat_.shape[0]][idx_], g_[W_hat_.shape[0]:], np.ones((Y_hat_rc_.shape[1],))*(np.unique(g_)[-1] + 1)], axis = 0)
        #print(Y_.shape, Y_hat_rc_.shape, X_.shape, X_rc_.shape, g_.shape, g_rc_.shape)
    else:
        X_rc_ = np.concatenate([X_[:, :W_hat_.shape[0], hrzn][:, idx_],
                                X_[:, W_hat_.shape[0]:, hrzn]], axis = 1)
        g_rc_ = np.concatenate([g_[:W_hat_.shape[0]][idx_], g_[W_hat_.shape[0]:]], axis = 0)
        #print(Y_.shape, X_.shape, X_rc_.shape, g_.shape, g_rc_.shape)
    return X_rc_, Y_[..., hrzn], g_rc_

# Get combination of possible parameters
def _get_cv_param(alphas_, betas_, omegas_, gammas_, etas_, lambdas_, xis_, sl, dl):
    thetas_ = []
    # Lasso parameters
    if sl == 0:
        thetas_.append(list(alphas_))
    # Orthogonal Matching Pursuit parameters
    if sl == 1:
        thetas_.append(list(betas_))
    # Elastic Net parameters
    if sl == 2:
        thetas_.append(list(alphas_))
        thetas_.append(list(omegas_))
    # Group lasso parameters
    if sl == 3:
        thetas_.append(list(gammas_))
        thetas_.append(list(etas_))
    # Bayesian Linear regression with ARD mechanism
    if dl == 1:
        thetas_.append(list(lambdas_))
    # Gaussian processes Kernels
    if dl == 2:
        thetas_.append(list(xis_))
    return list(product(*thetas_)), len(list(product(*thetas_)))

# Parallelize experiment combinations
def split_experiments_into_jobs_per_batches(exps_, i_batch, N_batches, i_job, N_jobs):
    exps_batch_ = np.linspace(0, len(exps_) - 1, len(exps_), dtype = int)[i_batch::N_batches]
    return [exps_batch_[i_job::N_jobs] for i_job in range(N_jobs)]

# Save in the next row of a .csv file
def _save_val_in_csv_file(data_, meta_, i_resource, path, name):
    for i_asset in range(data_.shape[0]):
        file_name = r'{}{}{}-{}'.format(path, i_resource, i_asset, name)
        row_      = meta_ + data_[i_asset, :].tolist()
        csv.writer(open(file_name, 'a')).writerow(row_)

def GaussianProcess(X_, y_, g_, xi, RC, hrzn, max_training_iter = 1000,
                                              n_random_init     = 2,
                                              early_stop        = 10,
                                              GP                = 'GPytorch'):
    # Model hyperparameter configurations
    kernels_ = ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']
    degrees_ = [0., 0., 2., 3., 1./2., 3./2., 5./2., 0.]
    params_  = [kernels_[xi], degrees_[xi], RC, hrzn, max_training_iter, n_random_init, early_stop]

    if GP == 'GPytorch': return _GPR_fit(X_, y_, g_, params_)
    if GP == 'sklearn':  return _skGPR_fit(X_, y_, g_, params_)

def MultiTaskGaussianProcess(X_, y_, g_, xi, RC, hrzn, max_training_iter = 1000,
                                                       n_random_init     = 2,
                                                       early_stop        = 10,
                                                       MTGP              = 'Bonilla'):
    # Model hyperparameter configurations
    kernels_ = ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']
    degrees_ = [0., 0., 2., 3., 1./2., 3./2., 5./2., 0.]
    params_  = [kernels_[xi], degrees_[xi], RC, hrzn, max_training_iter, n_random_init, early_stop]

    if MTGP == 'Bonilla': return _MTGPR_fit(X_, y_, g_, params_)
    if MTGP == 'Oscar':   return _cMTGPR_fit(X_, y_, g_, params_)

# Denser learning single-task mean and standard deviation
def _dense_learning_predict(_DL, X_, DL, GP   = 'GPytorch',
                                         MTGP = 'Bonilla'):
    # Linear models prediction
    if (DL == 0) | (DL == 1):
        m_hat_, s_hat_ = _DL.predict(X_, return_std = True)
        return m_hat_, s_hat_, np.sqrt(1./_DL.alpha_)
    # Gaussian process prediction
    if DL == 2:
        if GP == 'GPytorch':
            return _GPR_predict(_DL, X_)
        if GP == 'sklearn':
            m_hat_, s_hat_ = _DL.predict(X_, return_std = True)
            return m_hat_, s_hat_, np.sqrt(np.exp(_DL.kernel.theta[-1]))

    if DL == 3:
        if MTGP == 'Bonilla': return _MTGPR_predict(_DL, X_)
        if MTGP == 'Oscar':   return _cMTGPR_predict(_DL, X_)
        
# Save in the next row of a .csv file
def _save_test_in_csv_file(data_, key, times, i_theta_, thetas_, i_resource, path, name):
    for i_asset in range(data_.shape[0]):
        file_name = r'{}{}{}-{}'.format(path, i_resource, i_asset, name)
        row_      = [key] + [times] + [i_theta_[i_asset]] + [thetas_[i_asset] ] + data_[i_asset, :].tolist()
        csv.writer(open(file_name, 'a')).writerow(row_)

# Save in the next row of a .csv file
def _save_baselines_in_csv_file(data_, i_resource, path, name):
    for i_asset in range(data_.shape[1]):
        file_name = r'{}{}{}-{}'.format(path, i_resource, i_asset, name)
        row_ = []
        for i_model in range(data_.shape[2]):
            row_ += data_[i_asset, :, i_model].tolist()
        csv.writer(open(file_name, 'a')).writerow(row_)

# Save in the predictions in a .pkl file
def _save_pred_in_pkl_file(data_, key, i_resource, path, name):
    file_name = r'{}predictions/{}-{}{}'.format(path, i_resource, key, name)
    print(file_name)
    with open(file_name, 'wb') as _f:
        pickle.dump(data_, _f, protocol = pickle.HIGHEST_PROTOCOL)

__all__ = ['_get_node_info',
           '_load_data_in_chunks',
           '_structure_dataset',
           '_training_and_testing_dataset',
           '_naive_forecasts',
           '_sparse_learning_dataset',
           '_dense_learning_dataset',
           '_sparse_learning_predict',
           '_dense_learning_stand',
           '_spare_learning_stand',
           '_spare_learning_stand',
           '_prob_metrics',
           '_det_metrics',
           '_sparse_det_metrics',
           '_dense_learning_recursive_dataset',
           '_get_cv_param',
           'split_experiments_into_jobs_per_batches',
           '_save_val_in_csv_file',
           'GaussianProcess',
           'MultiTaskGaussianProcess',
           '_dense_learning_predict',
           '_save_test_in_csv_file',
           '_save_baselines_in_csv_file',
           '_save_pred_in_pkl_file']
