import sys, time, ast, pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from scipy.stats import norm, multivariate_normal

from utils import *
from scoring_utils import *
from aux_utils import *
from loading_utils import *

# Degine path to data
path_to_prc = r"/home/gterren/caiso_power/data/processed/"
path_to_raw = r"/home/gterren/caiso_power/data/dataset-2023/"
path_to_aux = r"/home/gterren/caiso_power/data/auxiliary/"

# Notes:
# SL = 0: X = {0, 1}, Y = {0}
# SL = 1: X = {1}, Y = {0}
# SL = 2: X = {0, 1}, Y = {0, 1}
# SL = 3: X = {0, 1}, Y = {0}
# DL = 0: X = {0, 1}, Y = {0}

resources_  = ['load', 'solar', 'wind']
# Resources: [{MWD, PGE, SCE, SDGE, VEA}, {NP, SP, ZP}, {NP, SP}]
i_resources_ = [0]
i_assets_    = [[1, 2, 3], [0, 1, 2], [0, 1]]

i_resources_ = [0, 1, 2]
i_assets_    = [[1], [0], [0]]
# i_resources_ = [0, 1, 2]
# i_assets_    = [[2], [1], [1]]
# i_resources_ = [0, 1]
# i_assets_    = [[3], [2]]

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

resource =  '_'.join([resources_[i_resource] for i_resource in i_resources_])
dataset  =  '_'.join(['{}-{}'.format(resources_[i_resource], '-'.join(map(str, i_assets_[i_resource]))) for i_resource in i_resources_]) + '_M{}.pkl'.format(i_mask)
print(resource, dataset)

# Loading spatial masks
M_ = _load_spatial_masks(i_resources_, path_to_aux)

# Load proposed data
data_ = _load_data_in_chunks([2019, 2020, 2021, 2022, 2023], path_to_raw)
#print(len(data_))

# Define data structure for a given experiment
Y_ac_, Y_fc_, X_ac_, X_fc_, Z_, ZZ_, g_sl_, g_dl_, assets_ = _multisource_structure_dataset(data_, i_resources_, i_assets_, M_[i_mask], tau)
#print(Y_ac_.shape, Y_fc_.shape, X_ac_.shape, X_fc_.shape, Z_.shape, g_sl_.shape, g_dl_.shape)
del data_

# Generate sparse learning dataset
X_sl_, Y_sl_, g_sl_ = _dense_learning_dataset(X_ac_, Y_ac_, Z_, g_sl_, N_lags, AR = 0,
                                                                               CS = 0,
                                                                               TM = 1)
#print(X_sl_.shape, Y_sl_.shape, g_sl_.shape)

# Generate dense learning dataset
X_dl_, Y_dl_, g_dl_ = _dense_learning_dataset(X_fc_, Y_ac_, Z_, g_dl_, N_lags, AR, CS, TM)
#print(X_dl_.shape, Y_dl_.shape, g_dl_.shape)

_generate_dataset(X_sl_, Y_sl_, g_sl_, X_dl_, Y_dl_, g_dl_, Z_, ZZ_, Y_ac_, Y_fc_, dataset, path_to_prc)
exit()

# Naive and CAISO forecasts as baselines
Y_per_fc_, Y_ca_fc_, Y_clm_fc_ = _naive_forecasts(Y_ac_, Y_fc_, N_lags)
#print(Y_per_fc_.shape, Y_ca_fc_.shape, Y_clm_fc_.shape)
del Y_ac_, Y_fc_

# Split data in training and testing
Y_per_fc_tr_, Y_per_fc_ts_ = _training_and_testing_dataset(Y_per_fc_)
Y_ca_fc_tr_, Y_ca_fc_ts_   = _training_and_testing_dataset(Y_ca_fc_)
Y_clm_fc_tr_, Y_clm_fc_ts_ = _training_and_testing_dataset(Y_clm_fc_)
print(Y_per_fc_ts_.shape, Y_ca_fc_ts_.shape, Y_clm_fc_ts_.shape)

dataset = '_'.join(['{}-{}'.format(resources_[i_resource], '-'.join(map(str, i_assets_[i_resource]))) for i_resource in i_resources_]) + '_baselines_v2.pkl'

#_save_baseline_fc(Y_per_fc_ts_, Y_ca_fc_ts_, Y_clm_fc_ts_, dataset, path_to_prc)
