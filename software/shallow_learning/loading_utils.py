import pickle, glob, blosc, csv

import numpy as np
import pandas as pd

from itertools import product, chain

from sklearn.preprocessing import StandardScaler

# Wind speed extrapolation at multiple altitudes (10, 60, 80, 100, and 120 m)
def _extrapolate_wind(M_10_, M_80_):
    # Compute power law
    def __power_law(m_1_, h_1, h_2, alpha_):
        return m_1_ * (h_2/h_1)**alpha_
    # Compute power law exponent
    alpha_ = (np.log(M_10_) - np.log(M_80_))/(np.log(80.) - np.log(10.))
    # Compute wind speed applying power law
    M_60_  = __power_law(M_10_, 60., 10., alpha_)
    M_100_ = __power_law(M_80_, 100., 80., alpha_)
    M_120_ = __power_law(M_80_, 120., 80., alpha_)
    return M_60_, M_100_, M_120_

def _periodic(x_, period):
    return np.cos(2.*np.pi*x_/period)

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

                    data_p_[2][:, 7, :], data_p_[2][:, 9, :], data_p_[2][:, 10, :]  = _extrapolate_wind(data_p_[2][:, 6, :], data_p_[2][:, 8, :])
                    data_p_[3][:, 8, :], data_p_[3][:, 10, :], data_p_[3][:, 11, :] = _extrapolate_wind(data_p_[3][:, 7, :], data_p_[3][:, 9, :])

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
                V_  = np.concatenate(V_, axis = 0)
                W_  = np.concatenate(W_, axis = 0)
                X_  = np.concatenate(X_, axis = 0)
                Y_  = np.concatenate(Y_, axis = 0)
                ZZ_ = np.concatenate(Z_, axis = 0)
                Z_  = ZZ_.astype(float).copy()

                Z_[:, 1] = _periodic(Z_[:, 1], period = 12)
                Z_[:, 2] = _periodic(Z_[:, 2], period = 31)
                Z_[:, 3] = _periodic(Z_[:, 3], period = 365)
                Z_[:, 4] = _periodic(Z_[:, 4], period = 24)
                Z_[:, 5] = _periodic(Z_[:, 5], period = 7)
                Z_[Z_[:, 6] == 0, 6] = -1.
                Z_[Z_[:, 7] == 0, 7] = -1.
                Z_[Z_[:, 8] == 0, 8] = -1.
                data_.append([V_, W_, X_, Y_, Z_, ZZ_])
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
# Load data combining multiple sources
def _multisource_structure_dataset(data_, i_resources_, i_assets_, F_idx_, tau, v_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
                                                                                w_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
                                                                                x_idx_ = [[2, 3, 4, 5, 11], [1, 2, 14], [7, 8, 9, 10]],
                                                                                y_idx_ = [[3, 4, 5, 6, 12], [2, 3, 15], [8, 9, 10, 11]],
                                                                                z_idx_ = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 7], [0, 1, 2, 3, 4, 7]]):
    # Combine necessary index for experiments
    v_all_idx_ = []
    w_all_idx_ = []
    x_all_idx_ = []
    y_all_idx_ = []
    z_all_idx_ = []
    for i_resource in i_resources_:
        for i_asset in i_assets_[i_resource]:
            v_all_idx_.append(v_idx_[i_resource][i_asset])
            w_all_idx_.append(w_idx_[i_resource][i_asset])
        x_all_idx_.append(x_idx_[i_resource])
        y_all_idx_.append(y_idx_[i_resource])
        z_all_idx_.append(z_idx_[i_resource])
    # Get unique index
    v_idx_ = v_all_idx_
    w_idx_ = w_all_idx_
    x_idx_ = sorted(list(set(chain.from_iterable(x_all_idx_))))
    y_idx_ = sorted(list(set(chain.from_iterable(y_all_idx_))))
    z_idx_ = sorted(list(set(chain.from_iterable(z_all_idx_))))
    print(v_idx_)
    print(w_idx_)
    print(x_idx_)
    print(y_idx_)
    print(z_idx_)
    # Concatenate all chucks of data in matrix form
    V_, W_, X_, Y_, Z_, ZZ_ = [], [], [], [], [], []
    for i in range(len(data_)):
        V_.append(data_[i][0][:, v_idx_])
        W_.append(data_[i][1][:, w_idx_])
        X_.append(data_[i][2][:, x_idx_, :])
        Y_.append(data_[i][3][:, y_idx_, :])
        Z_.append(data_[i][4][:, z_idx_])
        ZZ_.append(data_[i][5][:, z_idx_])
        #print(i, data_[i][0][:, v_idx_].shape, data_[i][1][:, w_idx_].shape)
    V_  = np.concatenate(V_, axis = 0)
    W_  = np.concatenate(W_, axis = 0)
    X_  = np.concatenate(X_, axis = 0)
    Y_  = np.concatenate(Y_, axis = 0)
    Z_  = np.concatenate(Z_, axis = 0)
    ZZ_ = np.concatenate(ZZ_, axis = 0)
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
    V_pp_, W_pp_, X_ppp_, Y_ppp_, Z_p_, ZZ_p_ = [], [], [], [], [], []
    for n in range(int(V_p_.shape[0]/24)):
        k = n*24
        l = (n + 1)*24
        V_pp_.append(V_p_[k:l, ...][:,np.newaxis])
        W_pp_.append(W_p_[k:l, ...][:, np.newaxis])
        X_ppp_.append(X_pp_[k:l, ...][:, np.newaxis, :])
        Y_ppp_.append(Y_pp_[k:l, ...][:, np.newaxis, :])
        Z_p_.append(Z_[k:l, ...][:, np.newaxis, :])
        ZZ_p_.append(ZZ_[k:l, ...][:, np.newaxis, :])

    V_pp_  = np.concatenate(V_pp_, axis = 1)
    W_pp_  = np.concatenate(W_pp_, axis = 1)
    X_ppp_ = np.concatenate(X_ppp_, axis = 1)
    Y_ppp_ = np.concatenate(Y_ppp_, axis = 1)
    Z_p_   = np.concatenate(Z_p_, axis = 1)
    ZZ_p_  = np.concatenate(ZZ_p_, axis = 1)
    return V_pp_, W_pp_, X_ppp_, Y_ppp_, Z_p_, ZZ_p_, G_sl_p_, G_dl_p_, v_all_idx_

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

# Generate sparse learning dataset
def _sparse_learning_dataset_format(X_ac_, Y_ac_):
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
def _dense_learning_dataset(X_fc_, Y_ac_, Z_, G_, lag, AR = 0,
                                                       CS = 0,
                                                       TM = 0):
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
    print(g_ar_)
    print(g_cs_)
    print(g_tm_)
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

# Standardize dense learning dataset
def _dense_learning_stand(X_dl_tr_, Y_dl_tr_, X_dl_ts_, x_stand = 0,
                                                        y_stand = 0):
    X_dl_tr_p_ = X_dl_tr_.copy()
    X_dl_ts_p_ = X_dl_ts_.copy()
    Y_dl_tr_p_ = Y_dl_tr_.copy()
    x_dl_scaler_ = []
    y_dl_scaler_ = []
    for i_hour in range(X_dl_tr_.shape[-1]):
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


# Standardize spare learning dataset
def _sparse_learning_stand(X_sl_tr_, y_sl_tr_, X_sl_ts_, x_stand = 0, y_stand = 0):
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

# Loading spatial masks
def _load_spatial_masks(i_resources_, path, map_file_name   = r"USland_0.125_(-125,-112)_(32,43).pkl",
                                            masks_file_name = r"density_grid_0.125_(-125,-112)_(32,43).pkl"):
    # Load the index of US land in the NOAA operational forecast
    US_land_ = pd.read_pickle(path + map_file_name)
    # Load the index of Demand, Solar, and Wind land in the NOAA operational forecast
    D_den_, S_den_, W_den_ = pd.read_pickle(path + masks_file_name)
    #print(US_land_.shape, D_den_.shape, S_den_.shape, W_den_.shape)
    # Define spatial feature masks
    F_ = np.zeros(US_land_.shape)
    for i_resource in i_resources_:
        F_ += [D_den_, S_den_, W_den_][i_resource]
    #M_ = [np.ones(US_land_.shape), US_land_, D_den_ + S_den_ + W_den_, [D_den_, S_den_, W_den_][i_resource]]
    return [np.ones(US_land_.shape), US_land_, D_den_ + S_den_ + W_den_, F_]

__all__ = ['_load_data_in_chunks',
           '_load_spatial_masks',
           '_multisource_structure_dataset',
           '_training_and_testing_dataset',
           '_sparse_learning_dataset_format',
           '_dense_learning_dataset',
           '_dense_learning_stand',
           '_sparse_learning_stand']
