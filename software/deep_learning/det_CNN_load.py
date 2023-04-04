import pickle, pickle5, glob, os, blosc, sys, torch

import numpy as np
import pandas as pd
import tensorflow as tf
import gpytorch as gp
import sklearn as sk

from time import sleep, time
from datetime import datetime, date, timedelta

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from gpytorch.constraints import Positive
from gpytorch.lazy import delazify
from gpytorch.kernels import Kernel

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import *

# Define system paths
path_to_pds = r"/home/gterren/caiso_power/data/datasets/"
path_to_aux = r"/home/gterren/caiso_power/data/auxiliary/"

# Get Grid Dimensions
N = 104
M = 88

# # Load the index of Demand, Solar, and Wind land in the NOAA operational forecast
# D_den_, S_den_, W_den_ = pd.read_pickle(path_to_aux + r"density_grid_0.125_(-125,-112)_(32,43).pkl")
# print(D_den_.shape, S_den_.shape, W_den_.shape)

# if pixels_selection == 0: idx_sel_ = S_den_ > 0.
# if pixels_selection == 1: idx_sel_ = (S_den_ + D_den_) > 0.
# if pixels_selection == 2: idx_sel_ = US_land_
# if pixels_selection == 3: idx_sel_ = np.ones(US_land_.shape)
#
# Load proposed data
data_ = _load_data_in_chunks([2019, 2020, 2021, 2022], path_to_pds)
print(len(data_), np.ones((N, M)).shape)

i_resource = 1
i_asset  = 1
# Define data structure for a given experiment
V_, W_, X_, Y_, Z_ = _structure_dataset(data_, i_resource, i_asset, v_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
                                                                    w_idx_ = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
                                                                    x_idx_ = [[], [1, 2, -1], []],
                                                                    y_idx_ = [[], [2, 3, -1], []],
                                                                    z_idx_ = [[0], [0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 6, 7]],
                                                                    D_idx_ = np.ones((M*N,)))
# del data_
# print(V_.shape, W_.shape, X_.shape, Y_.shape, Z_.shape)
