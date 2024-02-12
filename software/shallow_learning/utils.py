import pickle, glob, os, blosc, csv

import numpy as np

from time import sleep
from datetime import datetime, date, timedelta
from itertools import product, chain
from group_lasso import GroupLasso
from scipy.stats import norm, multivariate_normal
import properscoring as ps

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, OrthogonalMatchingPursuit, Ridge, ARDRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor

GroupLasso.LOG_LOSSES = True

from GP_utils import *

from ngboost import NGBRegressor
from ngboost.scores import LogScore, CRPScore
from ngboost.distns import Normal, MultivariateNormal

# # Load data in a compressed file
# def _load_data_in_chunks(years_, path):
#     # Open a BLOSC compressed file
#     def __load_data_in_compressed_file(file):
#         with open(file, "rb") as f:
#             data_ = f.read()
#         return pickle.loads(blosc.decompress(data_))
#     # Loop over processed years
#     data_ = []
#     for year in years_:
#         # Find processed data from that year
#         files_ = glob.glob(path + "{}_*".format(year))
#         # Define the maximum feasible number of chunks
#         N_min_chunks = len(files_)
#         # Loop over all possible chunks
#         for i in range(N_min_chunks):
#             V_, W_, X_, Y_, Z_ = [], [], [], [], []
#             for j in range(N_min_chunks):
#                 # Load data if extis
#                 try:
#                     file_name = path + "{}_{}-{}.dat".format(year, i, j)
#                     data_p_   = __load_data_in_compressed_file(file_name)
#                     # Append together all chucks
#                     V_.append(data_p_[0])
#                     W_.append(data_p_[1])
#                     X_.append(data_p_[2])
#                     Y_.append(data_p_[3])
#                     Z_.append(data_p_[4])
#                     print(file_name)
#                 except:
#                     continue
#             # Concatenate data if files existed
#             if len(X_) > 0:
#                 V_ = np.concatenate(V_, axis = 0)
#                 W_ = np.concatenate(W_, axis = 0)
#                 X_ = np.concatenate(X_, axis = 0)
#                 Y_ = np.concatenate(Y_, axis = 0)
#                 Z_ = np.concatenate(Z_, axis = 0)
#                 data_.append([V_, W_, X_, Y_, Z_])
#     return data_
#
# # v = {MWD (ac), PGE (ac), SCE (ac), SDGE (ac), VEA (ac), NP15 solar (ac), SP15 solar (ac), ZP26 solar (ac), NP15 wind (ac), SP15 wind (ac)}
# # w = {MWD (fc), PGE (fc), SCE (fc), SDGE (fc), VEA (fc), NP15 solar (fc), SP15 solar (fc), ZP26 solar (fc), NP15 wind (fc), SP15 wind (fc)}
# # X = {PRES (ac), DSWRF (ac), DLWRF (ac), DPT (ac), RH (ac), TMP (ac), W_10 (ac), W_60 (ac), W_80 (ac), W_100 (ac), W_120 (ac),
# #      DI (ac), WC (ac), HCDH (ac), GSI (ac)}
# # Y = {PRES (fc), PRATE (fc), DSWRF (fc), DLWRF (fc), DPT (fc), RH (fc), TMP (fc), W_10 (fc), W_60 (fc), W_80 (fc), W_100 (fc), W_120 (fc),
# #      DI (fc), WC (fc), HCDH (fc), GSI (fc)}
# # z = {year, month, day, yday, hour, weekday, weekend, isdst, holiday}
# # DSWRF = Diffuse Radiation
# # Is only water pumping... (?)
# def _structure_dataset(data_, i_resource, F_idx_, tau, v_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
#                                                        w_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
#                                                        x_idx_ = [[0, 1, 3, 4, 5, 11, 12, 13, 14], [1, 2, 14], [6, 7, 8, 9, 10]],
#                                                        y_idx_ = [[0, 1, 2, 4, 5, 6, 12, 13, 14, 15], [2, 3, 15], [7, 8, 9, 10, 11]],
#                                                        z_idx_ = [[0], [0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 6, 7]]):
#     v_idx_ = v_idx_[i_resource]
#     w_idx_ = w_idx_[i_resource]
#     x_idx_ = x_idx_[i_resource]
#     y_idx_ = y_idx_[i_resource]
#     z_idx_ = z_idx_[i_resource]
#     #F_idx_ = F_idx_[i_resource]
#     # Concatenate all chucks of data in matrix form
#     V_, W_, X_, Y_, Z_ = [], [], [], [], []
#     for i in range(len(data_)):
#         V_.append(data_[i][0][:, v_idx_])
#         W_.append(data_[i][1][:, w_idx_])
#         X_.append(data_[i][2][:, x_idx_, :])
#         Y_.append(data_[i][3][:, y_idx_, :])
#         Z_.append(data_[i][4][:, z_idx_])
#         #print(i, data_[i][0][:, v_idx_].shape, data_[i][1][:, w_idx_].shape)
#     V_ = np.concatenate(V_, axis = 0)
#     W_ = np.concatenate(W_, axis = 0)
#     X_ = np.concatenate(X_, axis = 0)
#     Y_ = np.concatenate(Y_, axis = 0)
#     Z_ = np.concatenate(Z_, axis = 0)
#     #print(V_.shape, W_.shape, X_.shape, Y_.shape, Z_.shape)
#     # Apply features selection heuristic
#     V_p_ = V_[:, :]
#     W_p_ = W_[:, :]
#     X_p_ = X_[..., F_idx_ > tau]
#     Y_p_ = Y_[..., F_idx_ > tau]
#     G_sl_ = np.concatenate([i*np.ones((X_p_.shape[-2], 1)) for i in range(X_p_.shape[-1])], axis = 1) # Group Lasso index for spatial features
#     #G_sl_ = np.concatenate([i*np.ones((1, X_p_.shape[-1])) for i in range(X_p_.shape[-2])], axis = 0) # Group Lasso index for weather features
#     G_dl_ = np.concatenate([i*np.ones((1, X_p_.shape[-1])) for i in range(X_p_.shape[-2])], axis = 0)
#     #print(V_p_.shape, W_p_.shape, X_p_.shape, Y_p_.shape, G_sl_.shape, G_dl_.shape)
#     del V_, W_, X_, Y_
#     # Concatenate all the dimensions
#     X_pp_, Y_pp_, G_sl_p_, G_dl_p_ = [], [], [], []
#     for d in range(X_p_.shape[1]):
#         X_pp_.append(X_p_[:, d, :])
#         Y_pp_.append(Y_p_[:, d, :])
#         G_sl_p_.append(G_sl_[d, :])
#         G_dl_p_.append(G_dl_[d, :])
#
#     X_pp_   = np.concatenate(X_pp_, axis = -1)
#     Y_pp_   = np.concatenate(Y_pp_, axis = -1)
#     G_sl_p_ = np.concatenate(G_sl_p_, axis = -1)
#     G_dl_p_ = np.concatenate(G_dl_p_, axis = -1)
#     #print(X_pp_.shape, Y_pp_.shape, G_sl_p_.shape, G_dl_p_.shape)
#     del X_p_, Y_p_, G_sl_, G_dl_
#     # Concatenate by hours
#     V_pp_, W_pp_, X_ppp_, Y_ppp_, Z_p_ = [], [], [], [], []
#     for n in range(int(V_p_.shape[0]/24)):
#         k = n*24
#         l = (n + 1)*24
#         V_pp_.append(V_p_[k:l, ...][:,np.newaxis])
#         W_pp_.append(W_p_[k:l, ...][:, np.newaxis])
#         X_ppp_.append(X_pp_[k:l, ...][:, np.newaxis, :])
#         Y_ppp_.append(Y_pp_[k:l, ...][:, np.newaxis, :])
#         Z_p_.append(Z_[k:l, ...][:, np.newaxis, :])
#     V_pp_  = np.concatenate(V_pp_, axis = 1)
#     W_pp_  = np.concatenate(W_pp_, axis = 1)
#     X_ppp_ = np.concatenate(X_ppp_, axis = 1)
#     Y_ppp_ = np.concatenate(Y_ppp_, axis = 1)
#     Z_p_   = np.concatenate(Z_p_, axis = 1)
#     return V_pp_, W_pp_, X_ppp_, Y_ppp_, Z_p_, G_sl_p_, G_dl_p_
#
# # Load data combining multiple sources
# def _multisource_structure_dataset(data_, i_resources_, i_assets_, F_idx_, tau, v_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
#                                                                                 w_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
#                                                                                 x_idx_ = [[0, 1, 3, 4, 5, 11, 12, 13, 14], [1, 2, 14], [6, 7, 8, 9, 10]],
#                                                                                 y_idx_ = [[0, 1, 2, 4, 5, 6, 12, 13, 14, 15], [2, 3, 15], [7, 8, 9, 10, 11]],
#                                                                                 z_idx_ = [[0], [0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 6, 7]]):
#     # Combine necessary index for experiments
#     v_all_idx_ = []
#     w_all_idx_ = []
#     x_all_idx_ = []
#     y_all_idx_ = []
#     z_all_idx_ = []
#     for i_resource in i_resources_:
#         for i_asset in i_assets_[i_resource]:
#             v_all_idx_.append(v_idx_[i_resource][i_asset])
#             w_all_idx_.append(w_idx_[i_resource][i_asset])
#         x_all_idx_.append(x_idx_[i_resource])
#         y_all_idx_.append(y_idx_[i_resource])
#         z_all_idx_.append(z_idx_[i_resource])
#     # Get unique index
#     v_idx_ = v_all_idx_
#     w_idx_ = w_all_idx_
#     x_idx_ = list(set(chain.from_iterable(x_all_idx_)))
#     y_idx_ = list(set(chain.from_iterable(y_all_idx_)))
#     z_idx_ = list(set(chain.from_iterable(z_all_idx_)))
#     print(v_idx_)
#     print(w_idx_)
#     print(x_idx_)
#     print(y_idx_)
#     print(z_idx_)
#     # Concatenate all chucks of data in matrix form
#     V_, W_, X_, Y_, Z_ = [], [], [], [], []
#     for i in range(len(data_)):
#         V_.append(data_[i][0][:, v_idx_])
#         W_.append(data_[i][1][:, w_idx_])
#         X_.append(data_[i][2][:, x_idx_, :])
#         Y_.append(data_[i][3][:, y_idx_, :])
#         Z_.append(data_[i][4][:, z_idx_])
#         #print(i, data_[i][0][:, v_idx_].shape, data_[i][1][:, w_idx_].shape)
#     V_ = np.concatenate(V_, axis = 0)
#     W_ = np.concatenate(W_, axis = 0)
#     X_ = np.concatenate(X_, axis = 0)
#     Y_ = np.concatenate(Y_, axis = 0)
#     Z_ = np.concatenate(Z_, axis = 0)
#     #print(V_.shape, W_.shape, X_.shape, Y_.shape, Z_.shape)
#     # Apply features selection heuristic
#     V_p_ = V_[:, :]
#     W_p_ = W_[:, :]
#     X_p_ = X_[..., F_idx_ > tau]
#     Y_p_ = Y_[..., F_idx_ > tau]
#     G_sl_ = np.concatenate([i*np.ones((X_p_.shape[-2], 1)) for i in range(X_p_.shape[-1])], axis = 1) # Group Lasso index for spatial features
#     #G_sl_ = np.concatenate([i*np.ones((1, X_p_.shape[-1])) for i in range(X_p_.shape[-2])], axis = 0) # Group Lasso index for weather features
#     G_dl_ = np.concatenate([i*np.ones((1, X_p_.shape[-1])) for i in range(X_p_.shape[-2])], axis = 0)
#     #print(V_p_.shape, W_p_.shape, X_p_.shape, Y_p_.shape, G_sl_.shape, G_dl_.shape)
#     del V_, W_, X_, Y_
#     # Concatenate all the dimensions
#     X_pp_, Y_pp_, G_sl_p_, G_dl_p_ = [], [], [], []
#     for d in range(X_p_.shape[1]):
#         X_pp_.append(X_p_[:, d, :])
#         Y_pp_.append(Y_p_[:, d, :])
#         G_sl_p_.append(G_sl_[d, :])
#         G_dl_p_.append(G_dl_[d, :])
#
#     X_pp_   = np.concatenate(X_pp_, axis = -1)
#     Y_pp_   = np.concatenate(Y_pp_, axis = -1)
#     G_sl_p_ = np.concatenate(G_sl_p_, axis = -1)
#     G_dl_p_ = np.concatenate(G_dl_p_, axis = -1)
#     #print(X_pp_.shape, Y_pp_.shape, G_sl_p_.shape, G_dl_p_.shape)
#     del X_p_, Y_p_, G_sl_, G_dl_
#     # Concatenate by hours
#     V_pp_, W_pp_, X_ppp_, Y_ppp_, Z_p_ = [], [], [], [], []
#     for n in range(int(V_p_.shape[0]/24)):
#         k = n*24
#         l = (n + 1)*24
#         V_pp_.append(V_p_[k:l, ...][:,np.newaxis])
#         W_pp_.append(W_p_[k:l, ...][:, np.newaxis])
#         X_ppp_.append(X_pp_[k:l, ...][:, np.newaxis, :])
#         Y_ppp_.append(Y_pp_[k:l, ...][:, np.newaxis, :])
#         Z_p_.append(Z_[k:l, ...][:, np.newaxis, :])
#     V_pp_  = np.concatenate(V_pp_, axis = 1)
#     W_pp_  = np.concatenate(W_pp_, axis = 1)
#     X_ppp_ = np.concatenate(X_ppp_, axis = 1)
#     Y_ppp_ = np.concatenate(Y_ppp_, axis = 1)
#     Z_p_   = np.concatenate(Z_p_, axis = 1)
#     return V_pp_, W_pp_, X_ppp_, Y_ppp_, Z_p_, G_sl_p_, G_dl_p_, v_all_idx_

# # Split Dataset in training and testing
# def _training_and_testing_dataset(X_, r_tr = 0.75):
#     # Compute Dataset samples in training and testing partition
#     N_samples    = X_.shape[0]
#     N_samples_tr = int(N_samples*r_tr)
#     N_samples_ts = N_samples - N_samples_tr
#     #print(N_samples, N_samples_tr, N_samples_ts)
#     # Make partions
#     X_tr_ = X_[:N_samples_tr, ...]
#     X_ts_ = X_[-N_samples_ts:, ...]
#     return X_tr_, X_ts_

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

# # Generate sparse learning dataset
# def _sparse_learning_dataset(X_ac_, Y_ac_):
#     # Define sparse learning regression dataset
#     y_sl_ = []
#     X_sl_ = []
#     for i in range(Y_ac_.shape[0]):
#         y_sl_.append(Y_ac_[i, ...])
#         X_sl_.append(X_ac_[i, ...])
#     y_sl_ = np.concatenate(y_sl_, axis = 1).T
#     X_sl_ = np.concatenate(X_sl_, axis = 1).T
#     #print(y_sl_.shape, X_sl_.shape)
#     return X_sl_, y_sl_

# # Generate dense learning dataset
# def _dense_learning_dataset(X_fc_, Y_ac_, Z_, G_, lag, AR = 0, CS = 0, TM = 0):
#     # Observations previous to the forecasting event
#     X_ar_ = np.swapaxes(np.swapaxes(Y_ac_[:-6, lag:-1], -1, -2), -2, -3)
#     # Observations from previous hours to the forecasting event
#     X_cs_ = np.swapaxes(np.swapaxes(np.concatenate([Y_ac_[:, lag - (l + 1):-(2 + l), ...][..., np.newaxis] for l in range(lag)], axis = -1), -1, -2), -2, -3)
#     X_cs_ = np.swapaxes(np.swapaxes(X_cs_, -1, -2), -2, -3)
#     X_ar_ = np.concatenate([X_ar_[i, ...] for i in range(X_ar_.shape[0])], axis = 0)
#     X_ar_ = np.swapaxes(np.concatenate([X_ar_[np.newaxis, ...] for _ in range(X_cs_.shape[0])], axis = 0), -1, -2)
#     X_cs_ = np.swapaxes(np.concatenate([X_cs_[:, i, ...] for i in range(X_cs_.shape[1])], axis = 1), -1, -2)
#     #print(X_ar_.shape, X_cs_.shape)
#     # Adjust timestamps signal and covariates
#     X_dl_ = X_fc_[:, lag + 1:, :]
#     Z_dl_ = Z_[:, lag + 1:, ...]
#     #print(X_dl_.shape, Z_dl_.shape)
#     # Get group index for kernel learning
#     g_ar_ = np.ones((X_ar_.shape[-1],))*(np.unique(G_)[-1] + 1)
#     g_cs_ = np.ones((X_cs_.shape[-1],))*(np.unique(G_)[-1] + 2)
#     g_tm_ = np.ones((Z_dl_.shape[-1],))*(np.unique(G_)[-1] + 3)
#     #print(G_.shape, g_ar_.shape, g_cs_.shape, g_dl_.shape, G_dl_.shape)
#     # Form covariate vector for dense learning
#     Y_dl_ = np.swapaxes(np.swapaxes(Y_ac_[:, lag + 1:, ...], 0, 1), -2, -1)
#     if AR == 1:
#         X_dl_ = np.concatenate((X_dl_, X_ar_), axis = 2)
#         G_    = np.concatenate([G_, g_ar_], axis = 0)
#     if CS == 1:
#         X_dl_ = np.concatenate((X_dl_, X_cs_), axis = 2)
#         G_    = np.concatenate([G_, g_cs_], axis = 0)
#     if TM == 1:
#         X_dl_ = np.concatenate((X_dl_, Z_dl_), axis = 2)
#         G_    = np.concatenate([G_, g_tm_], axis = 0)
#     X_dl_ = np.swapaxes(np.swapaxes(X_dl_, 0, 1), -2, -1)
#     #print(Y_dl_.shape, X_dl_.shape)
#     #print(np.unique(G_dl_))
#     return X_dl_, Y_dl_, G_

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

# # Define Recursive dataset
# def _dense_learning_recursive_dataset(X_, Y_, Y_hat_, g_, W_hat_, RC, hrzn, tsk = None):
#     # Find 0 coefficients obtained from sparse learning model
#     if tsk == None:
#         idx_ = np.sum(W_hat_, axis = 1) != 0.
#     else:
#         idx_ = W_hat_[..., tsk] != 0.
#
#     if RC:
#         # Form recursive dataset and add feature sources indexes
#         if tsk == None: Y_hat_rc_ = np.concatenate([Y_hat_[..., tsk, :hrzn] for tsk in range(Y_hat_.shape[1])], axis = 1)
#         else:           Y_hat_rc_ = Y_hat_[..., tsk, :hrzn]
#         X_rc_ = np.concatenate([X_[:, :W_hat_.shape[0], hrzn][:, idx_], X_[:, W_hat_.shape[0]:, hrzn], Y_hat_rc_], axis = 1)
#         g_rc_ = np.concatenate([g_[:W_hat_.shape[0]][idx_], g_[W_hat_.shape[0]:], np.ones((Y_hat_rc_.shape[1],))*(np.unique(g_)[-1] + 1)], axis = 0)
#         #print(Y_.shape, Y_hat_rc_.shape, X_.shape, X_rc_.shape, g_.shape, g_rc_.shape)
#     else:
#         X_rc_ = np.concatenate([X_[:, :W_hat_.shape[0], hrzn][:, idx_],
#                                 X_[:, W_hat_.shape[0]:, hrzn]], axis = 1)
#         g_rc_ = np.concatenate([g_[:W_hat_.shape[0]][idx_], g_[W_hat_.shape[0]:]], axis = 0)
#         #print(Y_.shape, X_.shape, X_rc_.shape, g_.shape, g_rc_.shape)
#     return X_rc_, Y_[..., hrzn], g_rc_

# Define Recursive dataset
def _dense_learning_recursive_dataset(X_, Y_, Y_hat_, g_, W_hat_, RC, hrzn, tsk = None):
    # Find 0 coefficients obtained from sparse learning model
    if tsk == None:
        idx_ = np.sum(W_hat_, axis = 1) != 0.
    else:
        idx_ = W_hat_[..., tsk] != 0.
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
def _get_cv_param(alphas_, betas_, omegas_, gammas_, etas_, lambdas_, xis_, kappas_, sl, dl):
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
    if (dl == 2) or (dl == 3):
        thetas_.append(list(xis_))
    # Natural Gradient Boosting number of estimator
    if dl == 4:
        thetas_.append(list(kappas_[0]))
        thetas_.append(list(kappas_[1]))
        thetas_.append(list(kappas_[2]))
        thetas_.append(list(kappas_[3]))

    return list(product(*thetas_)), len(list(product(*thetas_)))

# # Parallelize experiment combinations
# def _split_experiments_into_jobs_per_batches(exps_, i_batch, N_batches, i_job, N_jobs):
#     exps_batch_ = np.linspace(0, len(exps_) - 1, len(exps_), dtype = int)[i_batch::N_batches]
#     return [exps_batch_[i_job::N_jobs] for i_job in range(N_jobs)]

# # Save in the next row of a .csv file
# def _save_val_in_csv_file(data_, meta_, i_resource, path, name):
#     for i_asset in range(data_.shape[0]):
#         file_name = r'{}{}{}-{}'.format(path, i_resource, i_asset, name)
#         row_      = meta_ + data_[i_asset, :].tolist()
#         csv.writer(open(file_name, 'a')).writerow(row_)

# Natural Gradiente Boosting for Regression
def _NaturalGradientBoostingRegression(X_, Y_, g_, n_estimators   = 100,
                                                   minibatch_frac = 0.5,
                                                   learning_rate  = 0.01,
                                                   col_sample     = 0.5):
    #print(n_estimators, minibatch_frac, learning_rate, col_sample)
    return [NGBRegressor(Base             = DecisionTreeRegressor(criterion = 'friedman_mse',
                                                                  max_depth = 5),
                         Dist             = Normal,
                         Score            = LogScore,
                         n_estimators     = n_estimators,
                         minibatch_frac   = minibatch_frac,
                         learning_rate    = learning_rate,
                         col_sample       = col_sample,
                         tol              = 1e-4,
                         natural_gradient = True,
                         verbose          = False,
                         verbose_eval     = 1).fit(X_, Y_[:, i_tsk], early_stopping_rounds = 10) for i_tsk in range(Y_.shape[-1])]


# Bayesian Linear Regression with prior on the parameters
def _BayesianLinearRegression(X_, Y_, g_, max_iter = 1000):
    return [BayesianRidge(max_iter = max_iter,
                          tol      = 0.001).fit(X_, Y_[:, i_tsk]) for i_tsk in range(Y_.shape[-1])]

# Relevance Vector Machine for Regression with prior on the parameters
def _RelevanceVectorMachine(X_, Y_, g_, threshold_lambda, max_iter = 1000):
    return [ARDRegression(threshold_lambda = threshold_lambda,
                          n_iter           = max_iter,
                          tol              = 0.001).fit(X_, Y_[:, i_tsk]) for i_tsk in range(Y_.shape[-1])]

# def GaussianProcess(X_, y_, g_, xi, RC, hrzn, max_iter   = 1000,
#                                               n_init     = 9,
#                                               early_stop = 10,
#                                               GP         = 'GPytorch'):
#     print(X_, y_.shape)
#     # Model hyperparameter configurations
#     kernels_ = ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ', 'PW', 'PW', 'PW', 'PW']
#     degrees_ = [0., 0., 2., 3., 1./2., 3./2., 5./2., 0., 0, 1, 2, 3]
#     params_  = [kernels_[xi], degrees_[xi], RC, hrzn, max_iter, n_init, early_stop]
#
#     if GP == 'GPytorch': return _GPR_fit(X_, y_, g_, params_)
#     if GP == 'sklearn':  return _skGPR_fit(X_, y_, g_, params_)

# Gassuain Process for Regression
def _GaussianProcess(X_, Y_, g_, hrzn, xi, max_iter   = 1000,
                                           n_init     = 5,
                                           early_stop = 5,
                                           key        = ''):
    # Model hyperparameter configurations
    kernels_ = ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ',
                'linear_exp_rbf', 'linear_exp_matern', 'linear_exp_matern', 'linear_exp_matern', 'linear_exp_rq']
    degrees_ = [0., 0., 2., 3., 1./2., 3./2., 5./2., 0., 0, 1./2., 3./2., 5./2.,0.]
    params_  = [kernels_[xi], degrees_[xi], hrzn, max_iter, n_init, early_stop, key]
    return [_GPR_fit(X_, Y_[:, i_tsk], g_, params_) for i_tsk in range(Y_.shape[-1])]

def _MultiTaskGaussianProcess(X_, Y_, g_, xi, max_iter   = 250,
                                              n_init     = 5,
                                              early_stop = 10):
    # Model hyperparameter configurations
    kernels_ = ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']
    degrees_ = [0., 0., 2., 3., 1./2., 3./2., 5./2., 0.]

    params_  = [kernels_[xi], degrees_[xi], max_iter, n_init, early_stop]
    return _tcMTGPR_fit(X_, Y_, g_, params_)

# Dense learning multitask predictive mean
def _dense_learning_predict_recursive(_model, X_, DL):
    # Multitask Gaussian process prediction
    if DL == 3:
        Y_hat_ = _tcMTGPR_predict(_model, X_, return_cov = False)
    else:
        # Linear models prediction
        if (DL == 0) | (DL == 1):
            Y_hat_ = _model.predict(X_, return_std = False)
        # Gaussian process prediction
        if DL == 2:
            Y_hat_ = _GPR_predict(_model, X_, return_var = False)
        # Natural gradiate boosting prediction
        if DL == 4:
            Y_hat_ = _NGBR_predict(_model, X_, return_var = False)
    return Y_hat_

def _NGBR_predict(_model, X_, return_var = False):
    if return_var:
        _F_hat = _model.pred_dist(X_)
        return _F_hat.params['loc'], _F_hat.params['scale']
    else:
        return _model.predict(X_)

# Dense learning multitask predictive mean and covariance and noise covariance
def _dense_learning_multitask_predict(models_, X_, DL):
    # Multitask Gaussian process prediction
    if DL == 3:
        Y_hat_, S2_hat_ = _tcMTGPR_predict(models_, X_, return_cov = True)
    else:
        # Initialize predictive distribution parameters
        Y_hat_    = np.zeros((X_.shape[0], len(models_)))
        S2_hat_   = np.zeros((X_.shape[0], len(models_), len(models_)))
        for i_mdl in range(len(models_)):
            # Linear models prediction
            if (DL == 0) | (DL == 1):
                Y_hat_[:, i_mdl], s_hat_ = models_[i_mdl].predict(X_, return_std = True)
                S2_hat_[:, i_mdl, i_mdl] = s_hat_**2 + 1./models_[i_mdl].alpha_**2
            # Gaussian process prediction
            if DL == 2:
                Y_hat_[:, i_mdl], S2_hat_[:, i_mdl, i_mdl] = _GPR_predict(models_[i_mdl], X_, return_var = True)
    return Y_hat_, S2_hat_

# Draw random samples from a multivariate normal distribution
def _sample_multivariate_normal(Y_hat_, S2_hat_, N_samples = 1):
    if N_samples == 1: Z_ = np.zeros((Y_hat_.shape[0], Y_hat_.shape[1]))
    else:              Z_ = np.zeros((Y_hat_.shape[0], Y_hat_.shape[1], N_samples))
    for i in range(Y_hat_.shape[0]):
        min_eig = np.min(np.real(np.linalg.eigvals(S2_hat_[i, :])))
        if min_eig < 0:
            print('adding jitter')
            S2_hat_[i, :] -= 10*min_eig * np.eye(S2_hat_[i, :].shape)
        Z_[i, ...] = multivariate_normal(Y_hat_[i, :], S2_hat_[i, :], allow_singular = True).rvs(N_samples).T
    return Z_

# Draw random samples from a normal distribution
def _sample_normal(Y_hat_, S2_hat_, N_samples = 1):
    if N_samples == 1: Z_ = np.zeros((Y_hat_.shape[0], Y_hat_.shape[1]))
    else:              Z_ = np.zeros((Y_hat_.shape[0], Y_hat_.shape[1], N_samples))
    for i in range(Y_hat_.shape[0]):
        Z_[i, ...] = multivariate_normal(Y_hat_[i, :], np.diag(S2_hat_[i, :]), allow_singular = True).rvs(N_samples).T
    return Z_

# Dense learning single-task mean and standard deviation
def _dense_learning_predict(_DL, X_, DL):
    # Linear models prediction
    if (DL == 0) | (DL == 1):
        m_hat_, s_hat_ = _DL.predict(X_, return_std = True)
        return m_hat_, s_hat_**2 + 1./_DL.alpha_
    # Gaussian process prediction
    if DL == 2:
        return _GPR_predict(_DL, X_, return_var = True)
    # Multitask Gaussian process prediction
    if DL == 3:
        return _tcMTGPR_fit(_DL, X_, return_var = True)
    # Natural Gradient Booostion distribution prediction
    if DL == 4:
        m_hat_, s_hat_ = _NGBR_predict(_DL, X_, return_var = True)
        return m_hat_, s_hat_

# Define Recursive dataset
def _dense_learning_recursive_prediction(X_, Y_, Y_hat_, g_, W_hat_, RC, hrzn, tsk = None):
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

# Fit dense learning - Bayesian model chain
def _fit_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_, W_hat_, g_dl_, thetas_, RC, DL, key = ''):
    print('Dense learning model training...')
    # Initialization multi-task predictive mean
    Y_dl_tr_hat_ = np.zeros(Y_dl_tr_.shape)
    models_      = []
    # Train an expert models for each hour
    for hrzn in range(Y_dl_tr_hat_.shape[2]):
        model_ = []
        # Train an expert models for nodel
        for tsk in range(Y_dl_tr_hat_.shape[1]):
            # Define training and testing recursive dataset
            X_dl_tr_rc_, Y_dl_tr_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_hat_, g_dl_, W_hat_, RC, hrzn, tsk)
            print(tsk, hrzn, X_dl_tr_rc_.shape, Y_dl_tr_rc_.shape, g_dl_rc_.shape)
            # Bayesian Linear Regression with Hyperprior
            if DL == 0:
                _DL = _BayesianLinearRegression(X_dl_tr_rc_, Y_dl_tr_rc_[:, tsk][:, np.newaxis], g_dl_rc_)[0]
            # Linear Relevance Vector Machine Regression with Hyperprior
            if DL == 1:
                _DL = _RelevanceVectorMachine(X_dl_tr_rc_, Y_dl_tr_rc_[:, tsk][:, np.newaxis], g_dl_rc_, threshold_lambda = thetas_[tsk][-1])[0]
            # Gaussian Process for Regression with Hyperprior
            if DL == 2:
                _DL = _GaussianProcess(X_dl_tr_rc_, Y_dl_tr_rc_[:, tsk][:, np.newaxis], g_dl_rc_, hrzn, xi = thetas_[tsk][-1], key = key)[0]
            if DL == 4:
                _DL = _NaturalGradientBoostingRegression(X_dl_tr_rc_, Y_dl_tr_rc_[:, tsk][:, np.newaxis], g_dl_rc_, n_estimators   = thetas_[tsk][-4],
                                                                                                                    learning_rate  = thetas_[tsk][-3],
                                                                                                                    minibatch_frac = thetas_[tsk][-2],
                                                                                                                    col_sample     = thetas_[tsk][-1])[0]
            # Make prediction for recursive model
            Y_dl_tr_hat_[..., tsk, hrzn] = _dense_learning_predict_recursive(_DL, X_dl_tr_rc_, DL)
            # Save Multitask models
            model_.append(_DL)
        # Save multihorizon models
        models_.append(model_)
    return models_

# Fit sparse learning model
def _fit_sparse_learning(X_sl_tr_stnd_, X_sl_ts_stnd_, Y_sl_tr_stnd_, Y_sl_ts_, g_sl_, thetas_, sl_scaler_, SL, y_sl_stnd):
    print('Sparse learning model training...')
    # Initialization prediction mean and weights
    W_hat_       = np.ones((X_sl_ts_stnd_[:, g_sl_ != g_sl_[-1]].shape[1], Y_sl_ts_.shape[1]))
    Y_sl_ts_hat_ = np.zeros(Y_sl_ts_.shape)
    if SL != 4:
        # Train independent multi-task models for each hour
        for tsk in range(Y_sl_ts_.shape[1]):
            #print(tsk, thetas_[tsk], X_sl_tr_stnd_.shape, Y_sl_tr_stnd_.shape)
            # Lasso (linear regression with l_1 norm applied to the coefficients.)
            if SL == 0: _SL = Lasso(alpha    = thetas_[tsk][0],
                                    max_iter = 2000,
                                    tol      = 0.001).fit(X_sl_tr_stnd_, Y_sl_tr_stnd_[:, tsk])

            # Orthogonal Matching Persuit (linear regression with l_0 norm applied to the coefficients.)
            if SL == 1: _SL = OrthogonalMatchingPursuit(n_nonzero_coefs = thetas_[tsk][0]).fit(X_sl_tr_stnd_, Y_sl_tr_stnd_[:, tsk])

            # Elastic net (linear regression with l_1 and l_2 norm apply to coefficients)
            if SL == 2: _SL = ElasticNet(alpha    = thetas_[tsk][0],
                                         l1_ratio = thetas_[tsk][1],
                                         max_iter = 2000,
                                         tol      = 0.001).fit(X_sl_tr_stnd_, Y_sl_tr_stnd_[:, tsk])

            # Group lasso (linear regression with l_1 norm apply to coefficients and regularization applied coefficients by group)
            if SL == 3: _SL = GroupLasso(groups          = g_sl_,
                                         l1_reg          = thetas_[tsk][1]*thetas_[tsk][0],
                                         group_reg       = (1 - thetas_[tsk][1])*thetas_[tsk][0],
                                         n_iter          = 1000,
                                         scale_reg       = "inverse_group_size",
                                         supress_warning = True,
                                         tol             = 0.001).fit(X_sl_tr_stnd_, Y_sl_tr_stnd_[:, tsk])

            # Spare learning single-task prediction and optimal model coefficients
            Y_sl_ts_hat_[:, tsk], W_hat_[:, tsk] = _sparse_learning_predict(_SL, X_sl_ts_stnd_, g_sl_)

        # Undo standardization from sparse learning prediction
        if y_sl_stnd == 1:
            Y_sl_ts_hat_ = sl_scaler_[1].inverse_transform(Y_sl_ts_hat_)
    return W_hat_, Y_sl_ts_hat_

# Fit multitask dense learning - Bayesian model chain
def _fit_multitask_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_, W_hat_, g_dl_, thetas_, RC, DL, key = ''):
    # Initialization multi-task predictive mean
    Y_dl_tr_hat_ = np.zeros(Y_dl_tr_.shape)
    models_      = []
    # Train an expert models for each hour
    for hrzn in range(Y_dl_tr_hat_.shape[2]):
        tsk = 0
        # Define training and testing recursive dataset
        X_dl_tr_rc_, Y_dl_tr_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_hat_, g_dl_, W_hat_, RC, hrzn, tsk = None)
        print(hrzn, X_dl_tr_rc_.shape, Y_dl_tr_rc_.shape, g_dl_rc_.shape)
        # Bayesian Linear Regression with Hyperprior
        if DL == 0: _DL = _BayesianLinearRegression(X_dl_tr_rc_, Y_dl_tr_rc_, g_dl_rc_)
        # Linear Relevance Vector Machine Regression with Hyperprior
        if DL == 1: _DL = _RelevanceVectorMachine(X_dl_tr_rc_, Y_dl_tr_rc_, g_dl_rc_, threshold_lambda = thetas_[tsk][-1])
        # Gaussian Process for Regression with Hyperprior
        if DL == 2: _DL = _GaussianProcess(X_dl_tr_rc_, Y_dl_tr_rc_, g_dl_rc_, hrzn, xi = thetas_[tsk][-1], key = key)
        # Multi-Task Gaussian Process for Regression with Hyperprior
        if DL == 3: _DL = _MultiTaskGaussianProcess(X_dl_tr_rc_, Y_dl_tr_rc_, g_dl_rc_, xi = thetas_[tsk][-1])

        if DL != 3:
            for tsk in range(Y_dl_tr_hat_.shape[1]):
                Y_dl_tr_hat_[..., tsk, hrzn] = _dense_learning_predict_recursive(_DL[tsk], X_dl_tr_rc_, DL)
        else:
            Y_dl_tr_hat_[..., hrzn], _ = _dense_learning_multitask_predict(_DL, X_dl_tr_rc_, DL)
        models_.append(_DL)

    return models_

# Predictive probabilistic distribution
def _pred_prob_dist(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd):
    #(24, 3, 3) (3, 3) | (24, 3, 24) (8, 3, 24) (3, 24) - Multitask samples
    Y_dl_ts_stnd_hat_ = np.zeros(Y_dl_ts_.shape)
    M_dl_ts_hat_      = np.zeros(Y_dl_ts_.shape)
    S2_dl_ts_hat_     = np.zeros(Y_dl_ts_.shape)
    C_dl_ts_hat_      = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2]))
    #print(Y_dl_ts_stnd_hat_.shape, M_dl_ts_hat_.shape, S2_dl_ts_hat_.shape, C_dl_ts_hat_.shape)
    # Train an expert models for each hour
    for hrzn in range(Y_dl_ts_stnd_hat_.shape[2]):
        # Train an expert models for each hour
        for tsk in range(Y_dl_ts_stnd_hat_.shape[1]):
            # Define training and testing recursive dataset
            X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_, g_dl_, W_hat_, RC, hrzn, tsk)
            #print(key, thetas_[i_theta_[tsk]], hrzn, X_dl_ts_rc_.shape, Y_dl_ts_rc_.shape, g_dl_rc_.shape)
            Y_dl_ts_stnd_hat_[..., tsk, hrzn], S2_dl_ts_hat_[..., tsk, hrzn] = _dense_learning_predict(models_[hrzn][tsk], X_dl_ts_rc_, DL)
        # Undo standardization from dense learning multi-task prediction
        if y_dl_stnd == 1:
            # To undo predictive covariance necessary covariance instead of variance
            M_dl_ts_hat_[..., hrzn]  = dl_scaler_[1][hrzn].inverse_transform(Y_dl_ts_stnd_hat_[..., hrzn])
            S2_dl_ts_hat_[..., hrzn] = dl_scaler_[1][hrzn].var_*S2_dl_ts_hat_[..., hrzn]
        for obv in range(C_dl_ts_hat_.shape[0]):
            C_dl_ts_hat_[obv, ..., hrzn] = np.diag(S2_dl_ts_hat_[obv, ..., hrzn])
    return M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_

# Make multitask probabilistic prediction
# def _multitask_prob_predict(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_,
#                             thetas_, i_theta_, RC, DL, y_dl_stnd, N_samples = 100):
#     #(24, 3, 3) (3, 3) | (24, 3, 24) (8, 3, 24) (3, 24) - Multitask samples
#     Y_dl_ts_stnd_hat_ = np.zeros(Y_dl_ts_.shape)
#     Y_dl_ts_hat_      = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2], N_samples))
#     M_dl_ts_hat_      = np.zeros(Y_dl_ts_.shape)
#     S2_dl_ts_hat_     = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2]))
#     #print(Y_dl_ts_hat_.shape, M_dl_ts_hat_.shape, S2_dl_ts_hat_.shape, S2_dl_noise_.shape)
#     # Train an expert models for each hour
#     for hrzn in range(Y_dl_ts_hat_.shape[2]):
#         # Define training and testing recursive dataset
#         X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_, g_dl_, W_hat_, RC, hrzn, tsk = None)
#         #print(key, hrzn, X_dl_ts_rc_.shape, Y_dl_ts_rc_.shape, g_dl_rc_.shape)
#         Y_dl_ts_stnd_hat_[..., hrzn], S2_dl_ts_hat_[..., hrzn] = _dense_learning_multitask_predict(models_[hrzn], X_dl_ts_rc_, DL)
#         # Undo standardization from dense learning multi-task prediction
#         if y_dl_stnd == 1:
#             # To undo predictive covariance necessary covariance instead of variance
#             Cov_ = np.sqrt(dl_scaler_[1][hrzn].var_[:, np.newaxis]) @ np.sqrt(dl_scaler_[1][hrzn].var_[:, np.newaxis].T)
#             M_dl_ts_hat_[..., hrzn]  = dl_scaler_[1][hrzn].inverse_transform(Y_dl_ts_stnd_hat_[..., hrzn])
#             S2_dl_ts_hat_[..., hrzn] = Cov_*S2_dl_ts_hat_[..., hrzn]
#         # Sample Predictive Posterior Distribution
#         Y_dl_ts_hat_[..., hrzn, :] = _sample_multivariate_normal(M_dl_ts_hat_[..., hrzn], S2_dl_ts_hat_[..., hrzn], N_samples)
#     return Y_dl_ts_hat_, M_dl_ts_hat_, S2_dl_ts_hat_

# Multitask predictive probabilistic distribution
def _multitask_pred_prob_dist(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd):
    #(24, 3, 3) (3, 3) | (24, 3, 24) (8, 3, 24) (3, 24) - Multitask samples
    Y_dl_ts_stnd_hat_ = np.zeros(Y_dl_ts_.shape)
    M_dl_ts_hat_      = np.zeros(Y_dl_ts_.shape)
    S2_dl_ts_hat_     = np.zeros(Y_dl_ts_.shape)
    C_dl_ts_hat_      = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2]))
    #print(Y_dl_ts_stnd_hat_.shape, M_dl_ts_hat_.shape, C_dl_ts_hat_.shape)
    # Train an expert models for each hour
    for hrzn in range(Y_dl_ts_stnd_hat_.shape[2]):
        # Define training and testing recursive dataset
        X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_                    = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_, g_dl_, W_hat_, RC, hrzn, tsk = None)
        Y_dl_ts_stnd_hat_[..., hrzn], C_dl_ts_hat_[..., hrzn] = _dense_learning_multitask_predict(models_[hrzn], X_dl_ts_rc_, DL)
        # Undo standardization from dense learning multi-task prediction
        if y_dl_stnd == 1:
            # To undo predictive covariance necessary covariance instead of variance
            Cov_ = np.sqrt(dl_scaler_[1][hrzn].var_[:, np.newaxis]) @ np.sqrt(dl_scaler_[1][hrzn].var_[:, np.newaxis].T)
            M_dl_ts_hat_[..., hrzn]  = dl_scaler_[1][hrzn].inverse_transform(Y_dl_ts_stnd_hat_[..., hrzn])
            C_dl_ts_hat_[..., hrzn] = Cov_*C_dl_ts_hat_[..., hrzn]

    for smpl in range(S2_dl_ts_hat_.shape[0]):
        for tsk in range(S2_dl_ts_hat_.shape[-1]):
            S2_dl_ts_hat_[smpl, :, tsk] = np.diag(C_dl_ts_hat_[smpl, ..., tsk])

    return M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_

# # Multitask predictive probabilistic distribution
# def _multitask_pred_prob_dist_v0(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, thetas_, i_theta_, RC, DL, y_dl_stnd):
#     #(24, 3, 3) (3, 3) | (24, 3, 24) (8, 3, 24) (3, 24) - Multitask samples
#     Y_dl_ts_stnd_hat_ = np.zeros(Y_dl_ts_.shape)
#     M_dl_ts_hat_      = np.zeros(Y_dl_ts_.shape)
#     S2_dl_ts_hat_     = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2]))
#     #print(Y_dl_ts_stnd_hat_.shape, M_dl_ts_hat_.shape, S2_dl_ts_hat_.shape)
#     # Train an expert models for each hour
#     for hrzn in range(Y_dl_ts_stnd_hat_.shape[2]):
#         # Define training and testing recursive dataset
#         X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_, g_dl_, W_hat_, RC, hrzn, tsk = None)
#         #print(key, hrzn, X_dl_ts_rc_.shape, Y_dl_ts_rc_.shape, g_dl_rc_.shape)
#         Y_dl_ts_stnd_hat_[..., hrzn], S2_dl_ts_hat_[..., hrzn] = _dense_learning_multitask_predict(models_[hrzn], X_dl_ts_rc_, DL)
#         # Undo standardization from dense learning multi-task prediction
#         if y_dl_stnd == 1:
#             # To undo predictive covariance necessary covariance instead of variance
#             Cov_ = np.sqrt(dl_scaler_[1][hrzn].var_[:, np.newaxis]) @ np.sqrt(dl_scaler_[1][hrzn].var_[:, np.newaxis].T)
#             M_dl_ts_hat_[..., hrzn]  = dl_scaler_[1][hrzn].inverse_transform(Y_dl_ts_stnd_hat_[..., hrzn])
#             S2_dl_ts_hat_[..., hrzn] = Cov_*S2_dl_ts_hat_[..., hrzn]
#     return M_dl_ts_hat_, S2_dl_ts_hat_

# # Make joint probabilistic predictions
# def _joint_prob_predict(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_,
#                         thetas_, i_theta_, RC, DL, y_dl_stnd, N_samples = 100):
#     # Variable initialization
#     Y_dl_ts_stnd_hat_ = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2], N_samples))
#     Y_dl_ts_hat_      = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2], N_samples))
#     M_dl_ts_hat_      = np.zeros(Y_dl_ts_.shape)
#     S2_dl_ts_hat_     = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2]))
#     # Draw samples from predictive posterior
#     for smpl in range(N_samples):
#         # Train an expert models for each hour
#         for hrzn in range(Y_dl_ts_hat_.shape[2]):
#             # Train an expert models for each hour
#             for tsk in range(Y_dl_ts_hat_.shape[1]):
#                 # Define training and testing recursive dataset
#                 X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_[..., smpl], g_dl_, W_hat_, RC, hrzn, tsk)
#                 #print(key, thetas_[i_theta_[tsk]], smpl, hrzn, X_dl_ts_rc_.shape, Y_dl_ts_rc_.shape, g_dl_rc_.shape)
#                 M_dl_ts_hat_[..., tsk, hrzn], S2_dl_ts_hat_[..., tsk, hrzn] = _dense_learning_predict(models_[hrzn][tsk], X_dl_ts_rc_, DL)
#             # Undo standardization from dense learning multi-task prediction
#             if y_dl_stnd == 1:
#                 # To undo predictive covariance necessary covariance instead of variance
#                 M_dl_ts_hat_[..., hrzn]  = dl_scaler_[1][hrzn].inverse_transform(M_dl_ts_hat_[..., hrzn])
#                 S2_dl_ts_hat_[..., hrzn] = dl_scaler_[1][hrzn].var_*S2_dl_ts_hat_[..., hrzn]
#             # Sample Predictive Posterior Distribution
#             Y_dl_ts_hat_[..., hrzn, smpl] = _sample_normal(M_dl_ts_hat_[..., hrzn], S2_dl_ts_hat_[..., hrzn])
#             # Standardized predictors for recursive model
#             if y_dl_stnd == 1:
#                 Y_dl_ts_stnd_hat_[..., hrzn, smpl] = dl_scaler_[1][hrzn].transform(Y_dl_ts_hat_[..., hrzn, smpl])
#     return Y_dl_ts_hat_

# Joint probabilistic predictions
def _joint_prob_prediction(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd, N_samples = 100):
    # Variable initialization
    Y_dl_ts_stnd_hat_ = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2], N_samples))
    Y_dl_ts_hat_      = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2], N_samples))
    M_dl_ts_hat_      = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1]))
    S2_dl_ts_hat_     = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1]))
    # Draw samples from predictive posterior
    for smpl in range(N_samples):
        # Train an expert models for each hour
        for hrzn in range(Y_dl_ts_hat_.shape[2]):
            # Train an expert models for each hour
            for tsk in range(Y_dl_ts_hat_.shape[1]):
                # Define training and testing recursive dataset
                X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_              = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_[..., smpl], g_dl_, W_hat_, RC, hrzn, tsk)
                M_dl_ts_hat_[..., tsk], S2_dl_ts_hat_[..., tsk] = _dense_learning_predict(models_[hrzn][tsk], X_dl_ts_rc_, DL)
            # Undo standardization from dense learning multi-task prediction
            if y_dl_stnd == 1:
                # To undo predictive covariance necessary covariance instead of variance
                M_dl_ts_hat_  = dl_scaler_[1][hrzn].inverse_transform(M_dl_ts_hat_)
                S2_dl_ts_hat_ = dl_scaler_[1][hrzn].var_*S2_dl_ts_hat_
            # Sample Predictive Posterior Distribution
            Y_dl_ts_hat_[..., hrzn, smpl] = _sample_multivariate_normal(M_dl_ts_hat_, np.concatenate([np.diag(S2_dl_ts_hat_[i, :])[np.newaxis, ...] for i in range(S2_dl_ts_hat_.shape[0])], axis = 0))
            # Standardized predictors for recursive model
            if y_dl_stnd == 1:
                Y_dl_ts_stnd_hat_[..., hrzn, smpl] = dl_scaler_[1][hrzn].transform(Y_dl_ts_hat_[..., hrzn, smpl])
    return Y_dl_ts_hat_

# Multitask joint prediction
def _multitask_joint_prediction(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd, N_samples = 100):
    # Variable initialization
    Y_dl_ts_stnd_hat_ = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2], N_samples))
    Y_dl_ts_hat_      = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2], N_samples))
    #print(Y_dl_ts_stnd_hat_.shape, Y_dl_ts_hat_.shape)
    # Draw samples from predictive posterior
    for smpl in range(N_samples):
        # Train an expert models for each hour
        for hrzn in range(Y_dl_ts_hat_.shape[2]):
            # Define training and testing recursive dataset
            X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_[..., smpl], g_dl_, W_hat_, RC, hrzn, tsk = None)
            M_dl_ts_hat_, S2_dl_ts_hat_        = _dense_learning_multitask_predict(models_[hrzn], X_dl_ts_rc_, DL)
            # Undo standardization from dense learning multi-task prediction
            if y_dl_stnd == 1:
                # To undo predictive covariance necessary covariance instead of variance
                Cov_          = np.sqrt(dl_scaler_[1][hrzn].var_[:, np.newaxis]) @ np.sqrt(dl_scaler_[1][hrzn].var_[:, np.newaxis].T)
                M_dl_ts_hat_  = dl_scaler_[1][hrzn].inverse_transform(M_dl_ts_hat_)
                S2_dl_ts_hat_ = Cov_*S2_dl_ts_hat_
            # Sample Predictive Posterior Distribution
            Y_dl_ts_hat_[..., hrzn, smpl] = _sample_multivariate_normal(M_dl_ts_hat_, S2_dl_ts_hat_)
            # Standardized predictors for recursive model
            if y_dl_stnd == 1:
                Y_dl_ts_stnd_hat_[..., hrzn, smpl] = dl_scaler_[1][hrzn].transform(Y_dl_ts_hat_[..., hrzn, smpl])
    return Y_dl_ts_hat_

# # Multitask joint prediction
# def _multitask_joint_prediction_v0(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, thetas_, i_theta_, RC, DL, y_dl_stnd, N_samples = 100):
#     # Variable initialization
#     Y_dl_ts_stnd_hat_ = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2], N_samples))
#     Y_dl_ts_hat_      = np.zeros((Y_dl_ts_.shape[0], Y_dl_ts_.shape[1], Y_dl_ts_.shape[2], N_samples))
#     #print(Y_dl_ts_stnd_hat_.shape, Y_dl_ts_hat_.shape)
#     # Draw samples from predictive posterior
#     for smpl in range(N_samples):
#         # Train an expert models for each hour
#         for hrzn in range(Y_dl_ts_hat_.shape[2]):
#             # Define training and testing recursive dataset
#             X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_[..., smpl], g_dl_, W_hat_, RC, hrzn, tsk = None)
#             #print(key, thetas_[i_theta_[tsk]], smpl, hrzn, X_dl_ts_rc_.shape, Y_dl_ts_rc_.shape, g_dl_rc_.shape)
#             M_dl_ts_hat_, S2_dl_ts_hat_ = _dense_learning_multitask_predict(models_[hrzn], X_dl_ts_rc_, DL)
#             # Undo standardization from dense learning multi-task prediction
#             if y_dl_stnd == 1:
#                 # To undo predictive covariance necessary covariance instead of variance
#                 Cov_          = np.sqrt(dl_scaler_[1][hrzn].var_[:, np.newaxis]) @ np.sqrt(dl_scaler_[1][hrzn].var_[:, np.newaxis].T)
#                 M_dl_ts_hat_  = dl_scaler_[1][hrzn].inverse_transform(M_dl_ts_hat_)
#                 S2_dl_ts_hat_ = Cov_*S2_dl_ts_hat_
#             # Sample Predictive Posterior Distribution
#             Y_dl_ts_hat_[..., hrzn, smpl] = _sample_multivariate_normal(M_dl_ts_hat_, S2_dl_ts_hat_)
#             # Standardized predictors for recursive model
#             if y_dl_stnd == 1:
#                 Y_dl_ts_stnd_hat_[..., hrzn, smpl] = dl_scaler_[1][hrzn].transform(Y_dl_ts_hat_[..., hrzn, smpl])
#     return Y_dl_ts_hat_

__all__ = ['_naive_forecasts',
           '_sparse_learning_predict',
           '_dense_learning_recursive_dataset',
           '_get_cv_param',
           '_BayesianLinearRegression',
           '_GaussianProcess',
           '_MultiTaskGaussianProcess',
           '_RelevanceVectorMachine',
           '_dense_learning_predict_recursive',
           '_dense_learning_predict',
           '_dense_learning_multitask_predict',
           '_sample_multivariate_normal',
           '_sample_normal',
           '_fit_dense_learning',
           '_fit_sparse_learning',
           '_fit_multitask_dense_learning',
           '_pred_prob_dist',
           '_multitask_pred_prob_dist',
           '_joint_prob_prediction',
           '_multitask_joint_prediction']
