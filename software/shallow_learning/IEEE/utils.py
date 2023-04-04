import pickle, glob, os, blosc

import numpy as np
import pandas as pd

from time import sleep
from datetime import datetime, date, timedelta
from mpi4py import MPI

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

# Process all chunks of data to form a dataset with a given strcuture
def _structure_dataset(data_, i_resource, i_asset, v_idx_ = None,
                                                   w_idx_ = None,
                                                   x_idx_ = None,
                                                   y_idx_ = None,
                                                   z_idx_ = None,
                                                   D_idx_ = None):
    v_idx_ = v_idx_[i_resource]
    w_idx_ = w_idx_[i_resource]
    x_idx_ = x_idx_[i_resource]
    y_idx_ = y_idx_[i_resource]
    z_idx_ = z_idx_[i_resource]
    # Concatenate all chucks of data in matrix form
    V_, W_, X_, Y_, Z_ = [], [], [], [], []
    for i in range(len(data_)):
        V_.append(data_[i][0][:, v_idx_])
        W_.append(data_[i][1][:, w_idx_])
        X_.append(data_[i][2][:, x_idx_, :])
        Y_.append(data_[i][3][:, y_idx_, :])
        Z_.append(data_[i][4][:, z_idx_])
    V_ = np.concatenate(V_, axis = 0)
    W_ = np.concatenate(W_, axis = 0)
    X_ = np.concatenate(X_, axis = 0)
    Y_ = np.concatenate(Y_, axis = 0)
    Z_ = np.concatenate(Z_, axis = 0)
    print(V_.shape, W_.shape, X_.shape, Y_.shape, Z_.shape)
    # Apply features selection heuristic
    V_p_ = V_[:, i_asset]
    W_p_ = W_[:, i_asset]
    X_p_ = X_[..., D_idx_ > 0.]
    Y_p_ = Y_[..., D_idx_ > 0.]
    del V_, W_, X_, Y_
    # Concatenate all the dimensions
    X_pp_, Y_pp_ = [], []
    for d in range(X_p_.shape[1]):
        X_pp_.append(X_p_[:, d, :])
        Y_pp_.append(Y_p_[:, d, :])
    X_pp_ = np.concatenate(X_pp_, axis = -1)
    Y_pp_ = np.concatenate(Y_pp_, axis = -1)
    print(X_pp_.shape, Y_pp_.shape)
    del X_p_, Y_p_
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
    del W_p_, Y_pp_, Z_
    return V_pp_, W_pp_, X_ppp_, Y_ppp_, Z_p_
    #return V_pp_, V_p_, X_pp_, Y_ppp_, Z_p_

# Use daylight data and align DST with solar time
def _DST_aligning_solar_data(V_, W_, X_, Y_, Z_):

    # Daylight DST days to solar time
    idx_dst_0_ = np.array([False, False, False, False, False, True,
                           True, True, True, True, True, True,
                           True, True, True, True, True, True,
                           True, True, True, False, False, False])
    idx_dst_1_ = np.array([False, False, False, False, False, True,
                           True, True, True, True, True, True,
                           True, True, True, True, True, True,
                           True, True, True, False, False, False])
    # Find DST days and no-DST days
    idx_dts_p_0_ = Z_[0, :, -2] == 0.
    idx_dts_p_1_ = Z_[0, :, -2] != 0.
    # Correct shifting finltering out DST and no-DST days
    V_p_0_ = V_[idx_dst_0_, :][:, idx_dts_p_0_]
    V_p_1_ = V_[idx_dst_1_, :][:, idx_dts_p_1_]
    W_p_0_ = W_[idx_dst_0_, :][:, idx_dts_p_0_]
    W_p_1_ = W_[idx_dst_1_, :][:, idx_dts_p_1_]
    X_p_0_ = X_[idx_dst_0_, :, :][:, idx_dts_p_0_, :]
    X_p_1_ = X_[idx_dst_1_, :, :][:, idx_dts_p_1_, :]
    Y_p_0_ = Y_[idx_dst_0_, :, :][:, idx_dts_p_0_, :]
    Y_p_1_ = Y_[idx_dst_1_, :, :][:, idx_dts_p_1_, :]
    Z_p_0_ = Z_[idx_dst_0_, :, :][:, idx_dts_p_0_, :]
    Z_p_1_ = Z_[idx_dst_1_, :, :][:, idx_dts_p_1_, :]
    # Recostruct time series
    V_p_ = np.zeros((idx_dst_0_.sum(), V_.shape[-1]))
    W_p_ = np.zeros((idx_dst_0_.sum(), W_.shape[-1]))
    X_p_ = np.zeros((idx_dst_0_.sum(), X_.shape[1], X_.shape[2]))
    Y_p_ = np.zeros((idx_dst_0_.sum(), Y_.shape[1], Y_.shape[2]))
    Z_p_ = np.zeros((idx_dst_0_.sum(), Z_.shape[1], Z_.shape[2]))
    # Correct DST shifting
    V_p_[:, idx_dts_p_0_]    = V_p_0_
    V_p_[:, idx_dts_p_1_]    = V_p_1_
    W_p_[:, idx_dts_p_0_]    = W_p_0_
    W_p_[:, idx_dts_p_1_]    = W_p_1_
    X_p_[:, idx_dts_p_0_, :] = X_p_0_
    X_p_[:, idx_dts_p_1_, :] = X_p_1_
    Y_p_[:, idx_dts_p_0_, :] = Y_p_0_
    Y_p_[:, idx_dts_p_1_, :] = Y_p_1_
    Z_p_[:, idx_dts_p_0_, :] = Z_p_0_
    Z_p_[:, idx_dts_p_1_, :] = Z_p_1_
    return V_p_, W_p_, X_p_, Y_p_, Z_p_

# Root Mean Squared Error
def _RMSE(Y_, Y_hat_):
    return np.sqrt(np.mean((Y_ - Y_hat_)**2, axis = 0))

# Mean Absolute Error
def _MAE(Y_, Y_hat_):
    return np.mean(np.absolute(Y_ - Y_hat_), axis = 0)

# Mean Absolute Percentage Error
def _MAPE(Y_, Y_hat_):
    return 100.*np.mean(np.absolute((Y_ - Y_hat_)/Y_), axis = 0)

# Mean Bias Error
def _MBE(Y_, Y_hat_):
    return np.mean((Y_ - Y_hat_), axis = 0)

# Coefficient of determination
def _R2(Y_, Y_hat_):
    return 1. - (np.sum((Y_ - Y_hat_)**2, axis = 0)/np.sum((Y_ - np.mean(Y_))**2, axis = 0))

# Defien data structure to train a global model
def _global_model_structure(X_, Y_):
    X_p_, y_p_ = [], []
    # Concatenate all days in a single calumn
    for i_day in range(Y_.shape[0]):
        X_p_.append(X_[:, i_day, :])
        y_p_.append(Y_[i_day, :])
    X_p_ = np.concatenate(X_p_, axis = 1).T
    y_p_ = np.concatenate(y_p_, axis = 0)[:, np.newaxis]
    return X_p_, y_p_

# Defien data structure to train a global model
def _revert_global_model_structure(X_, Y_, N_hours):
    N_samples  = Y_.shape[0]
    N_days     = int(N_samples/N_hours)
    X_p_, y_p_ = [], []
    # Concatenate all days in a single calumn
    for i_day in range(N_days):
        i = i_day*N_hours
        j = (i_day + 1)*N_hours

        X_p_.append(X_[i:j, :][np.newaxis, ...])
        y_p_.append(Y_[i:j][:, np.newaxis])

    X_p_ = np.swapaxes(np.concatenate(X_p_, axis = 0).T, 1, 2)
    y_p_ = np.concatenate(y_p_, axis = 1).T
    return X_p_, y_p_

def _compute_error_metrics(Y_, Y_hat_):
    N_hours = Y_.shape[-1]
    RMSE_ = np.array([_RMSE(Y_[..., i], Y_hat_[..., i]) for i in range(N_hours)])[:, np.newaxis]
    MAE_  = np.array([_MAE(Y_[..., i], Y_hat_[..., i]) for i in range(N_hours)])[:, np.newaxis]
    MAPE_ = np.array([_MAPE(Y_[..., i], Y_hat_[..., i]) for i in range(N_hours)])[:, np.newaxis]
    MBE_  = np.array([_MBE(Y_[..., i], Y_hat_[..., i]) for i in range(N_hours)])[:, np.newaxis]
    R2_   = np.array([_R2(Y_[..., i], Y_hat_[..., i]) for i in range(N_hours)])[:, np.newaxis]
    return np.concatenate((RMSE_, MAE_, MAPE_, MBE_, R2_), axis = 1)

# Autocorrelation Function
def _ACF(x_, lag):
    return np.array([1] + [np.corrcoef(x_[:-i], x_[i:])[0, 1] for i in range(1, lag + 1)])[1:][:, np.newaxis]

# Persistent Ferecast
def _persistence(Y_):
    return Y_[:, :-1]

# Climatological Ferecast
def _climatology(Y_, lag = 6):
    N_predictions = Y_.shape[-1]- lag
    Y_hat_ = []
    for n in range(1, N_predictions):
        k = -n
        l = -n - lag
        Y_hat_.append(np.mean(Y_[:, l:k], axis = 1)[:, np.newaxis])
    return np.concatenate(Y_hat_, axis = 1)[:, ::-1]

# Covex Climatological Ferecast
def _convex_climatology_persistence(Y_cli_, Y_per_, ACF_):
    Y_cov_ = np.zeros(Y_per_.shape)
    for i in range(ACF_.shape[0]):
        Y_cov_[:, i] = Y_per_[:, i]*ACF_[i] + Y_cli_[:, i]*(1. - ACF_[i])
    return Y_cov_

__all__ = ['_get_node_info',
           '_load_data_in_chunks',
           '_structure_dataset',
           '_DST_aligning_solar_data',
           '_global_model_structure',
           '_revert_global_model_structure',
           '_compute_error_metrics',
           '_convex_climatology_persistence',
           '_climatology',
           '_persistence',
           '_ACF']
