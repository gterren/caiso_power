import pickle, glob, os, blosc, csv

import numpy as np
import pandas as pd

from time import sleep
from datetime import datetime, date, timedelta

from itertools import product, chain

from sklearn.preprocessing import StandardScaler
from group_lasso import GroupLasso

from scipy.stats import norm, multivariate_normal

import properscoring as ps

# # Continuous Rank Probability Score
# def _CRPS(y_, y_hat_, s_2_):
#     s_ = np.sqrt(s_2_)
#     x_ = (y_ - y_hat_)/s_
#     # Standard Normal Distribution mu = 0, std = 1
#     _N = norm(0, 1)
#     return np.mean(s_*(1/np.sqrt(np.pi) - 2.*_N.pdf(x_) - x_*(2.*_N.cdf(x_) - 1.)))

# Root Mean Squared Error
def _RMSE(y_, y_hat_):
    return np.sqrt(np.mean((y_ - y_hat_)**2, axis = 0))

# Mean Absolute Error
def _MAE(y_, y_hat_):
    return np.mean(np.absolute(y_ - y_hat_), axis = 0)

# Mean Bias Error
def _MBE(y_, y_hat_):
    return np.mean((y_ - y_hat_), axis = 0)

# # Negative Log Predictive Probability
# def _NLPP(y_, y_hat_, s_2_hat_):
#     return - np.sum([norm(y_hat_[i], np.sqrt(s_2_hat_[i])).logpdf(y_[i]) for i in range(y_hat_.shape[-1])])
#
# # Compute probabilistic scores
# def _prob_metrics(Y_, Y_hat_, S_hat_):
#     scores_ = []
#     # Samples / Tasks / Forecasting horizons
#     for tsk in range(Y_hat_.shape[1]):
#         NLPP_ = np.array([_NLPP(Y_[..., tsk, hrzn], Y_hat_[..., tsk, hrzn], S_hat_[..., tsk, hrzn]) for hrzn in range(Y_.shape[-1])])[np.newaxis, :]
#         CRPS_ = np.array([_CRPS(Y_[..., tsk, hrzn], Y_hat_[..., tsk, hrzn], S_hat_[..., tsk, hrzn]) for hrzn in range(Y_.shape[-1])])[np.newaxis, :]
#         scores_.append(np.concatenate((NLPP_, CRPS_), axis = 0)[:, np.newaxis, :])
#     return np.swapaxes(np.concatenate(scores_, axis = 1), 0, 1)

# Energy Score
def _ES(Y_, Y_hat_):
    N_observation, N_tasks, N_horizons, N_samples = Y_hat_.shape
    score_ = np.zeros((N_horizons, N_observation))
    for m in range(N_horizons):
        for i in range(N_observation):
            Y_p_   = np.tile(Y_[i, :, m], (N_samples, 1)).T
            frac1_ = np.sqrt(np.diag((Y_hat_[i, :, m, :] - Y_p_).T @ (Y_hat_[i, :, m, :] - Y_p_)))
            frac2_ = 0
            for j in range(N_samples):
                frac2_ += np.sqrt(np.sum((Y_hat_[i, :, m, :].T - Y_hat_[i, :, m, j])**2, axis = 1))
            score_[m, i] = np.sum(frac1_)/N_samples - np.sum(frac2_)/(2*(N_samples**2))
    return score_

# Variogram Score
def _VS(Y_, Y_hat_, p = .5):
    N_observation, N_tasks, N_horizons, N_samples = Y_hat_.shape
    score_ = np.zeros((N_horizons, N_observation))
    for n in range(N_horizons):
        for i in range(N_observation):
            frac1_ = np.absolute(np.subtract.outer(Y_[i, :, n], Y_[i, :, n]))**p
            frac2_ = np.zeros((N_tasks, N_tasks))
            for m in range(N_samples):
                frac2_ += np.absolute(np.subtract.outer(Y_hat_[i, :, n, m], Y_hat_[i, :, n, m]))**p
            score_[n, i] = np.sum((frac1_ - (frac2_/N_samples))**2)
    return score_

# Continuous Ranked Probability Score
def _CRPS(Y_, Y_hat_):
    N_observation, N_tasks, N_horizons, N_samples = Y_hat_.shape
    score_ = np.zeros((N_horizons, N_observation, N_tasks))
    for i in range(N_horizons):
        for j in range(N_observation):
            for k in range(N_tasks):
                score_[i, j, k] = ps.crps_ensemble(Y_[j, k, i], Y_hat_[j, k, i, :])
    return score_

# Multivaritate Normal Log Predictive Probability
def _mvNLPP(Y_, M_hat_, Cov_hat_):
    z_ = np.zeros((Y_.shape[0], ))
    for obv in range(M_hat_.shape[0]):
        z_[obv] = multivariate_normal(M_hat_[obv, ...], Cov_hat_[obv, ...], allow_singular = True).logpdf(Y_[obv, ...])
    return - np.mean(z_, axis = 0)

# Normal Log Predictive Probability
def _NLPP(Y_, M_hat_, S2_hat_):
    Z_ = np.zeros(Y_.shape)
    for obv in range(M_hat_.shape[0]):
        for tsk in range(M_hat_.shape[1]):
            Z_[obv, tsk] = norm(M_hat_[obv, tsk], S2_hat_[obv, tsk]).logpdf(Y_[obv, tsk])
    return - np.mean(Z_, axis = 0)

# # Multivaritate Continuous Rank Probability Score
# def _mvCRPS(Y_, Y_hat_):
#     mvCRPS_ = np.zeros((Y_.shape[0], Y_.shape[1]))
#     for tsk in range(Y_.shape[1]):
#         mvCRPS_[..., tsk] = ps.crps_ensemble(Y_[..., tsk], Y_hat_[..., tsk, :])
#     return np.mean(mvCRPS_, axis = 0)

# Ignorance Score
def _IS(Y_, M_hat_, Cov_hat_):
    # Samples / Tasks / Forecasting horizons
    return np.array([_mvNLPP(Y_[..., hrzn], M_hat_[..., hrzn], Cov_hat_[..., hrzn]) for hrzn in range(Y_.shape[-1])])

# Independent Ignorance Score
def _IIS(Y_, M_hat_, S2_hat_):
    # Samples / Tasks / Forecasting horizons
    return np.stack([_NLPP(Y_[..., hrzn], M_hat_[..., hrzn], S2_hat_[..., hrzn]) for hrzn in range(Y_.shape[-1])])

# # Compute CRPS probabilistuc score
# def _mv_CRPS_score(Y_, Y_hat_):
#     # Samples / Tasks / Forecasting horizons
#     return np.array([_mvCRPS(Y_[..., hrzn], Y_hat_[..., hrzn, :]) for hrzn in range(Y_.shape[-1])]).T


# # Compute deterministic scores
# def _det_metrics(Y_, Y_hat_):
#     scores_ = []
#     # Samples / Tasks / Forecasting horizons
#     for tsk in range(Y_hat_.shape[1]):
#         RMSE_ = np.array([_RMSE(Y_[..., tsk, hrzn], Y_hat_[..., tsk, hrzn]) for hrzn in range(Y_.shape[-1])])[np.newaxis, :]
#         MAE_  = np.array([ _MAE(Y_[..., tsk, hrzn], Y_hat_[..., tsk, hrzn]) for hrzn in range(Y_.shape[-1])])[np.newaxis, :]
#         MBE_  = np.array([ _MBE(Y_[..., tsk, hrzn], Y_hat_[..., tsk, hrzn]) for hrzn in range(Y_.shape[-1])])[np.newaxis, :]
#         scores_.append(np.concatenate((RMSE_, MAE_, MBE_), axis = 0)[:, np.newaxis, :])
#     return np.swapaxes(np.concatenate(scores_, axis = 1), 0, 1)

# Baselines det. error metrics
def _baseline_det_metrics(Y_, Y_hat_):
    Y_p_     = []
    Y_hat_p_ = []
    for hrzn in range(Y_hat_.shape[2]):
        Y_p_.append(Y_[..., hrzn])
        Y_hat_p_.append(Y_hat_[..., hrzn])
    Y_p_     = np.concatenate(Y_p_, axis = 0)
    Y_hat_p_ = np.concatenate(Y_hat_p_, axis = 0)
    return _sparse_det_metrics(Y_p_, Y_hat_p_)

# Compute deterministic scores for sparse model
def _sparse_det_metrics(Y_, Y_hat_):
    scores_ = []
    # Samples / Tasks / Forecasting horizons
    for tsk in range(Y_hat_.shape[1]):
        scores_.append(np.array([_RMSE(Y_[..., tsk], Y_hat_[..., tsk]),
                                  _MAE(Y_[..., tsk], Y_hat_[..., tsk]),
                                  _MBE(Y_[..., tsk], Y_hat_[..., tsk])])[..., np.newaxis])
    scores_ = np.concatenate(scores_, axis = 1).T
    return pd.DataFrame(scores_, columns = ['RMSE', 'MAE', 'MBE'],
                                 index   = ['NP15', 'SP15', 'ZP26'])


# Probabilistic forecat metrics
def _prob_metrics(Y_, M_, S2_, Y_hat_):
    # Independent Ignorance Score
    IIS_ = _IIS(Y_, M_, S2_)
    IIS_ = np.mean(IIS_, axis = 0)
    # Continus Rank Probability Score
    CRPS_ = _CRPS(Y_, Y_hat_)
    CRPS_ = np.mean(CRPS_, axis = 1)
    CRPS_ = np.mean(CRPS_, axis = 0)
    # Sample Independent Ignorance Score
    M_    = np.mean(Y_hat_, axis = -1)
    S2_   = np.std(Y_hat_, axis = -1)
    sIIS_ = _IIS(Y_, M_, S2_)
    sIIS_ = np.mean(sIIS_, axis = 0)
    return pd.DataFrame(np.stack([IIS_, CRPS_, sIIS_]).T, columns = ['IIS', 'CRPS', 'sample_ISS'],
                                                          index   = ['NP15', 'SP15', 'ZP26'])

# Probabilistic multivariate forecat metrics
def _multivariate_prob_metrics(Y_, M_, Cov_, Y_hat_):
    # Ignorance Score
    IS_ = _IS(Y_, M_, Cov_)
    IS_ = np.mean(IS_, axis = 0)
    # Energy Score
    ES_ = _ES(Y_, Y_hat_)
    ES_ = np.mean(ES_, axis = 1)
    ES_ = np.mean(ES_, axis = 0)
    # Variogram Score
    VS_ = _VS(Y_, Y_hat_)
    VS_ = np.mean(VS_, axis = 1)
    VS_ = np.mean(VS_, axis = 0)
    # Sample Ignorance Score
    M_   = np.mean(Y_hat_, axis = -1)
    Cov_ = np.zeros((M_.shape[0], M_.shape[1], M_.shape[1], M_.shape[2]))
    for obs in range(M_.shape[0]):
        for hrz in range(M_.shape[-1]):
            Cov_[obs, ..., hrz] = np.cov(Y_hat_[obs, :, hrz, ...])
    sIS_ = _IS(Y_, M_, Cov_)
    sIS_ = np.mean(sIS_, axis = 0)
    return pd.DataFrame(np.array([IS_, ES_, VS_, sIS_]), columns = [''],
                                                         index   = ['IS', 'ES', 'VS', 'sample_IS']).T

__all__ = ['_RMSE',
           '_MAE',
           '_CRPS',
           '_VS',
           '_ES',
           '_IS',
           '_IIS',
           '_sparse_det_metrics',
           '_baseline_det_metrics',
           '_prob_metrics',
           '_multivariate_prob_metrics']
