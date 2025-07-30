import pickle, glob, os, blosc, csv

import numpy as np
import pandas as pd

from time import sleep
from datetime import datetime, date, timedelta

from itertools import product, chain

from scipy.stats import norm, multivariate_normal
from scipy.ndimage.filters import gaussian_filter1d

import properscoring as ps

# Root Mean Squared Error
def _RMSE(y_, y_hat_):
    return np.sqrt(np.mean((y_ - y_hat_)**2, axis = 0))

# Mean Absolute Error
def _MAE(y_, y_hat_):
    return np.mean(np.absolute(y_ - y_hat_), axis = 0)

# Mean Bias Error
def _MBE(y_, y_hat_):
    return np.mean((y_ - y_hat_), axis = 0)

# Energy Score
def _ES_spatial(Y_, Y_hat_):
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

# Variogram Score computed across spatial dimensions
def _VS_spatial(Y_, Y_hat_, p = .5):
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

# Variogram Score computed across temporal dimensions
def _VS_temporal(Y_, Y_hat_, p = .5):
    N_observation, N_tasks, N_horizons, N_samples = Y_hat_.shape
    score_ = np.zeros((N_tasks, N_observation))
    for t in range(N_tasks):
        for n in range(N_observation):
            frac1_ = np.absolute(np.subtract.outer(Y_[n, t, :], Y_[n, t, :]))**p
            frac2_ = np.zeros((N_horizons, N_horizons))
            for m in range(N_samples):
                frac2_ += np.absolute(np.subtract.outer(Y_hat_[n, t, :, m], Y_hat_[n, t, :, m]))**p
            score_[t, n] = np.sum((frac1_ - (frac2_/N_samples))**2)
    return score_


# Energy Score across time
def _ES(Y_, Y_hat_):

    N_observation, N_tasks, N_horizons, N_scen = Y_hat_.shape

    score_ = np.zeros((N_observation))
    Y_     = np.concatenate([Y_[:, tsk, ...] for tsk in range(N_tasks)], axis = 1)
    Y_hat_ = np.concatenate([Y_hat_[:, tsk, ...] for tsk in range(N_tasks)], axis = 1)

    for n in range(N_observation):
        Y_p_   = np.tile(Y_[n, :], (N_scen, 1)).T
        frac1_ = np.sqrt(np.diag((Y_hat_[n, :, :] - Y_p_).T @ (Y_hat_[n, :, :] - Y_p_)))
        frac2_ = 0
        for m in range(N_scen):
            frac2_ += np.sqrt(np.sum((Y_hat_[n, :, :].T - Y_hat_[n, :, m])**2, axis = 1))
        score_[n] = (np.sum(frac1_)/N_scen) - (np.sum(frac2_)/(2*(N_scen**2)))
    return score_


# Variogram Score computed across temporal dimensions
def _VS(Y_, Y_hat_, p = .5):
    N_observation, N_tasks, N_horizons, N_scen = Y_hat_.shape
    score_ = np.zeros((N_observation))
    Y_     = np.concatenate([Y_[:, tsk, ...] for tsk in range(N_tasks)], axis = 1)
    Y_hat_ = np.concatenate([Y_hat_[:, tsk, ...] for tsk in range(N_tasks)], axis = 1)
    for n in range(N_observation):
        frac1_ = np.absolute(np.subtract.outer(Y_[n, ], Y_[n, :]))**p
        frac2_ = np.zeros((N_horizons*N_tasks, N_horizons*N_tasks))
        for m in range(N_scen):
            frac2_ += np.absolute(np.subtract.outer(Y_hat_[n, :, m], Y_hat_[n, :, m]))**p
        score_[n] = np.sum((frac1_ - (frac2_/N_scen))**2)
    return score_

# Energy Score across time
def _ES_temporal(Y_, Y_hat_):
    N_observation, N_tasks, N_horizons, N_samples = Y_hat_.shape
    score_ = np.zeros((N_tasks, N_observation))
    for t in range(N_tasks):
        for n in range(N_observation):
            Y_p_   = np.tile(Y_[n, t, :], (N_samples, 1)).T
            frac1_ = np.sqrt(np.diag((Y_hat_[n, t, :, :] - Y_p_).T @ (Y_hat_[n, t, :, :] - Y_p_)))
            frac2_ = 0
            for m in range(N_samples):
                frac2_ += np.sqrt(np.sum((Y_hat_[n, t, :, :].T - Y_hat_[n, t, :, m])**2, axis = 1))
            score_[t, n] = np.sum(frac1_)/N_samples - np.sum(frac2_)/(2*(N_samples**2))
    return score_

# Varify samples within a confidence interval
def _verify_confidence_intervals(Y_, M_, S2_, z = 1.96):
    print(Y_.shape, M_.shape, S2_.shape)
    N_in = 0.
    for i_zone in range(M_.shape[1]):
        for i_hour in range(M_.shape[2]):
            y_    = Y_[:, i_zone, i_hour]
            mu_   = M_[:, i_zone, i_hour]
            s2_   = np.sqrt(S2_[:, i_zone, i_hour])
            idx_  = (y_ >= (mu_ - z*s2_)) & (y_ <= (mu_ + z*s2_))
            N_in += idx_.sum()
    return N_in/(Y_.shape[0]*Y_.shape[1]*Y_.shape[2])

# Continuous Ranked Probability Score
def _CRPS(Y_, Y_hat_):
    N_observation, N_tasks, N_horizons, N_samples = Y_hat_.shape
    score_ = np.zeros((N_horizons, N_observation, N_tasks))
    for i in range(N_horizons):
        for j in range(N_observation):
            for k in range(N_tasks):
                score_[i, j, k] = ps.crps_ensemble(Y_[j, k, i], Y_hat_[j, k, i, :])
    return score_

# Normal Log Predictive Probability
def _NLPP(Y_, M_hat_, S2_hat_):
    Z_ = np.zeros(Y_.shape)
    for obv in range(M_hat_.shape[0]):
        for tsk in range(M_hat_.shape[1]):
            Z_[obv, tsk] = norm(M_hat_[obv, tsk], S2_hat_[obv, tsk]).logpdf(Y_[obv, tsk])
    return - np.mean(Z_, axis = 0)

# Independent Ignorance Score
def _IIS(Y_, M_hat_, S2_hat_):
    # Samples / Tasks / Forecasting horizons
    return np.stack([_NLPP(Y_[..., hrzn], M_hat_[..., hrzn], S2_hat_[..., hrzn]) for hrzn in range(Y_.shape[-1])])

# Probabilistic forecat metrics
def _prob_metrics(Y_, M_, S2_, Y_hat_, headers_):
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
                                                          index   = headers_)

# Robust logarithms of a multivariate normal distribution evaluation
def _robust_log_eval_multivariate_normal(m_hat_, C_hat_, Y_):
    #return multivariate_normal(m_hat_, C_hat_, allow_singular = True).logpdf(Y_)
    try:
        return multivariate_normal(m_hat_, C_hat_, allow_singular = True).logpdf(Y_)
    except:
        print('Adding jitter...')
        epsilon = np.min(np.real(np.linalg.eigvals(C_hat_)))
        return multivariate_normal(m_hat_, C_hat_ - 10*epsilon * np.eye(C_hat_.shape[0]),
                                   allow_singular = True).logpdf(Y_)

# Log Score
def _agg_LogS(Y_, M_hat_, C_hat_):
    # Multivaritate Normal Log Predictive Probability
    def _mvNLPP(Y_, M_hat_, C_hat_):
        N_samples = Y_.shape[0]
        z_ = np.zeros((N_samples,))
        for i_sample in range(N_samples):
            #z_[i_sample] = multivariate_normal(M_hat_[i_sample, ...], Cov_hat_[i_sample, ...], allow_singular = True).logpdf(Y_[i_sample, ...])
            z_[i_sample] = _robust_log_eval_multivariate_normal(M_hat_[i_sample, ...], C_hat_[i_sample, ...], Y_[i_sample, ...])
        return - z_.sum()
    # Samples / Tasks / Forecasting horizons
    N_horizons = Y_.shape[-1]
    return np.array([_mvNLPP(Y_[..., i_horizon], M_hat_[..., i_horizon], C_hat_[..., i_horizon]) for i_horizon in range(N_horizons)])

# Interval Score
def _IS(y_, alpha, lower_, upper_):
    score_  = (upper_ - lower_)
    score_ += (2./alpha)*(lower_ - y_) * (1*(y_ < lower_))
    score_ += (2./alpha)*(y_ - upper_) * (1*(y_ > upper_))
    return score_

def _interval_score(Y_, M_, S2_, z, alpha):
    # 95% CI: z = 1.96; alpha = 0.05
    # 90% CI: z = 1.645; alpha = 0.1
    N_observation, N_tasks, N_horizons = M_.shape
    Y_     = np.concatenate([Y_[:, tsk, ...] for tsk in range(N_tasks)], axis = 1)
    M_     = np.concatenate([M_[:, tsk, ...] for tsk in range(N_tasks)], axis = 1)
    S2_    = np.concatenate([S2_[:, tsk, ...] for tsk in range(N_tasks)], axis = 1)
    score_ = np.zeros((N_observation,))
    for n in range(N_observation):
        score_[n] = _IS(Y_[n, :], alpha, lower_ = M_[n, :] - z*np.sqrt(S2_[n, :]),
                                         upper_ = M_[n, :] + z*np.sqrt(S2_[n, :])).sum()
    return score_

# Sample Logarithmic Score
def _sample_agg_LogS(Y_, Y_hat_):
    M_   = np.mean(Y_hat_, axis = -1)
    Cov_ = np.zeros((M_.shape[0], M_.shape[1], M_.shape[1], M_.shape[2]))
    for obs in range(M_.shape[0]):
        for hrz in range(M_.shape[-1]):
            Cov_[obs, ..., hrz] = np.cov(Y_hat_[obs, :, hrz, ...])
    return _agg_LogS(Y_, M_, Cov_)

# Log-Score
def _LogS(Y_, M_hat_, C_hat_):
    # Multivaritate Normal Log Predictive Probability
    def _mvNLPP(Y_, M_hat_, C_hat_):
        N_samples = Y_.shape[0]
        z_ = np.zeros((N_samples,))
        for i_sample in range(N_samples):
            #z_[i_sample] = multivariate_normal(M_hat_[i_sample, ...], C_hat_[i_sample, ...], allow_singular = True).logpdf(Y_[i_sample, ...])
            z_[i_sample] = _robust_log_eval_multivariate_normal(M_hat_[i_sample, ...], C_hat_[i_sample, ...], Y_[i_sample, ...])
        return - z_
    # Samples / Tasks / Forecasting horizons
    N_horizons = Y_.shape[-1]
    return np.sum(np.array([_mvNLPP(Y_[..., i_horizon], M_hat_[..., i_horizon], C_hat_[..., i_horizon]) for i_horizon in range(N_horizons)]), axis = 0)

# Probabilistic multivariate forecat metrics
def _multivariate_prob_metrics(Y_, M_, Cov_, S2_, Y_hat_):
    N_samples  = M_.shape[0]
    N_tasks    = M_.shape[1]
    N_horizons = M_.shape[2]
    print(Y_.shape, M_.shape, Cov_.shape, S2_.shape, Y_hat_.shape)
    # Multivariate proper scoring rules
    LogS_ = _LogS(Y_, M_, Cov_).sum()/(N_samples*N_tasks*N_horizons)
    VS_   = _VS(Y_, Y_hat_).sum()/(N_samples*N_tasks*N_horizons)
    ES_   = _ES(Y_, Y_hat_).sum()/(N_samples*N_tasks*N_horizons)
    # Compute Ignorance Score at differet confidence levels
    IS975_ = _interval_score(Y_, M_, S2_, z = 2.243, alpha = 0.025).sum()/(N_samples*N_tasks*N_horizons)
    IS95_  = _interval_score(Y_, M_, S2_, z = 1.959, alpha = 0.05).sum()/(N_samples*N_tasks*N_horizons)
    IS90_  = _interval_score(Y_, M_, S2_, z = 1.645, alpha = 0.1).sum()/(N_samples*N_tasks*N_horizons)
    IS80_  = _interval_score(Y_, M_, S2_, z = 1.282, alpha = 0.2).sum()/(N_samples*N_tasks*N_horizons)
    IS60_  = _interval_score(Y_, M_, S2_, z = 0.842, alpha = 0.4).sum()/(N_samples*N_tasks*N_horizons)
    # Aggragated-Logarithmic Score
    AggLogS_ = _agg_LogS(Y_, M_, Cov_)
    AggLogS_ = np.sum(AggLogS_, axis = 0)
    # Energy Score computed in the spatial dimensions
    ES_spatial_ = _ES_spatial(Y_, Y_hat_)
    ES_spatial_ = np.mean(ES_spatial_, axis = 1)
    ES_spatial_ = np.mean(ES_spatial_, axis = 0)
    # Energy Score computed in the temporal dimensions
    ES_temporal_ = _ES_temporal(Y_, Y_hat_)
    ES_temporal_ = np.mean(ES_temporal_, axis = 1)
    ES_temporal_ = np.mean(ES_temporal_, axis = 0)
    # Variogram Score computed in the spatial dimensions
    VS_spatial_ = _VS_spatial(Y_, Y_hat_)
    VS_spatial_ = np.mean(VS_spatial_, axis = 1)
    VS_spatial_ = np.mean(VS_spatial_, axis = 0)
    # Variogram Score computed in the temporal dimensions
    VS_temporal_ = _VS_temporal(Y_, Y_hat_)
    VS_temporal_ = np.mean(VS_temporal_, axis = 1)
    VS_temporal_ = np.mean(VS_temporal_, axis = 0)
    # Compute number of samples within confidence interval
    CI975_ = _verify_confidence_intervals(Y_, M_, S2_, z = 2.243)
    CI95_  = _verify_confidence_intervals(Y_, M_, S2_, z = 1.959)
    CI90_  = _verify_confidence_intervals(Y_, M_, S2_, z = 1.645)
    CI80_  = _verify_confidence_intervals(Y_, M_, S2_, z = 1.282)
    CI60_  = _verify_confidence_intervals(Y_, M_, S2_, z = 0.842)
    # Sample Logarithmic Score
    sAggLogS_ = _sample_agg_LogS(Y_, Y_hat_)
    sAggLogS_ = np.mean(sAggLogS_, axis = 0)
    return pd.DataFrame(np.array([AggLogS_, sAggLogS_, ES_spatial_, VS_spatial_, ES_temporal_, VS_temporal_,
                                  LogS_, ES_, VS_, IS60_, IS80_, IS90_, IS95_, IS975_, CI60_, CI80_, CI90_, CI95_, CI975_]),
                        columns = [''],
                        index   = ['AggLogS', 'sAggLogS',
                                   'ES_spatial', 'VS_spatial', 'ES_temporal', 'VS_temporal',
                                   'LogS', 'ES', 'VS', 'IS60', 'IS80', 'IS90', 'IS95', 'IS975', 'CI60', 'CI80', 'CI90', 'CI95', 'CI975']).T

# Probabilistic multivariate forecat metrics
def _multiresource_prob_metrics(Y_, M_, Cov_, S2_, Y_hat_, headers_):
    
    LogS  = 0.
    VS    = 0.
    ES    = 0.
    IS975 = 0.
    IS95  = 0.
    IS90  = 0.
    IS80  = 0.
    IS60  = 0.
    CI975 = 0.
    CI95  = 0.
    CI90  = 0.
    CI80  = 0.
    CI60  = 0.

    N_samples  = M_.shape[0]
    N_tasks    = M_.shape[1]
    N_horizons = M_.shape[2]

    # Exclude solar time series from morning and evening horizons
    # morning:   0 - 7
    # afternoon: 8 - 15
    # evening:  16 - 23

    idx_solar_ = []
    idx_other_ = []
    for hearder in range(len(headers_)):
        if 'solar' in headers_[hearder]:
            idx_solar_.append(hearder)
        else:
            idx_other_.append(hearder)

    resource_idxs_ = [idx_other_, sorted(idx_solar_ + idx_other_), idx_other_]
    dayzone_idxs_  = [[0, 10], [10, 14], [14, 24]]

    for i in range(3):
        dayzone_idx_  = dayzone_idxs_[i]
        resource_idx_ = resource_idxs_[i]

        Y_p_     = Y_[:, resource_idx_, dayzone_idx_[0]:dayzone_idx_[1]].copy()
        M_p_     = M_[:, resource_idx_, dayzone_idx_[0]:dayzone_idx_[1]].copy()
        Cov_p_   = Cov_[..., resource_idx_, dayzone_idx_[0]:dayzone_idx_[1]].copy()
        Cov_p_   = Cov_p_[:, resource_idx_, ...]
        S2_p_    = S2_[:, resource_idx_, dayzone_idx_[0]:dayzone_idx_[1]].copy()
        Y_hat_p_ = Y_hat_[:, resource_idx_, dayzone_idx_[0]:dayzone_idx_[1], :].copy()

        # Multivariate proper scoring rules
        LogS += _LogS(Y_p_, M_p_, Cov_p_).sum()
        VS   += _VS(Y_p_, Y_hat_p_).sum()
        ES   += _ES(Y_p_, Y_hat_p_).sum()
        # Compute Ignorance Score at differet confidence levels
        IS975 += _interval_score(Y_p_, M_p_, S2_p_, z = 2.243, alpha = 0.025).sum()
        IS95  += _interval_score(Y_p_, M_p_, S2_p_, z = 1.959, alpha = 0.05).sum()
        IS90  += _interval_score(Y_p_, M_p_, S2_p_, z = 1.645, alpha = 0.1).sum()
        IS80  += _interval_score(Y_p_, M_p_, S2_p_, z = 1.282, alpha = 0.2).sum()
        IS60  += _interval_score(Y_p_, M_p_, S2_p_, z = 0.842, alpha = 0.4).sum()
        # Compute number of samples within confidence interval
        CI975 += _verify_confidence_intervals(Y_p_, M_p_, S2_p_, z = 2.243)*(dayzone_idx_[1] - dayzone_idx_[0])
        CI95  += _verify_confidence_intervals(Y_p_, M_p_, S2_p_, z = 1.959)*(dayzone_idx_[1] - dayzone_idx_[0])
        CI90  += _verify_confidence_intervals(Y_p_, M_p_, S2_p_, z = 1.645)*(dayzone_idx_[1] - dayzone_idx_[0])
        CI80  += _verify_confidence_intervals(Y_p_, M_p_, S2_p_, z = 1.282)*(dayzone_idx_[1] - dayzone_idx_[0])
        CI60  += _verify_confidence_intervals(Y_p_, M_p_, S2_p_, z = 0.842)*(dayzone_idx_[1] - dayzone_idx_[0])

    LogS  /= N_samples*N_tasks*N_horizons
    VS    /= N_samples*N_tasks*N_horizons
    ES    /= N_samples*N_tasks*N_horizons
    IS975 /= N_samples*N_tasks*N_horizons
    IS95  /= N_samples*N_tasks*N_horizons
    IS90  /= N_samples*N_tasks*N_horizons
    IS80  /= N_samples*N_tasks*N_horizons
    IS60  /= N_samples*N_tasks*N_horizons
    CI975 /= N_horizons
    CI95  /= N_horizons
    CI90  /= N_horizons
    CI80  /= N_horizons
    CI60  /= N_horizons

    return pd.DataFrame(np.array([LogS, ES, VS, IS60, IS80, IS90, IS95, IS975, CI60, CI80, CI90, CI95, CI975]),
                        columns = [''],
                        index   = ['LogS', 'ES', 'VS', 'IS60', 'IS80', 'IS90', 'IS95', 'IS975', 'CI60', 'CI80', 'CI90', 'CI95', 'CI975']).T


# Compute deterministic scores for sparse model
def _sparse_det_metrics(Y_, Y_hat_, hearders_):
    scores_ = []
    # Samples / Tasks / Forecasting horizons
    for tsk in range(Y_hat_.shape[1]):
        scores_.append(np.array([_RMSE(Y_[..., tsk], Y_hat_[..., tsk]),
                                  _MAE(Y_[..., tsk], Y_hat_[..., tsk]),
                                  _MBE(Y_[..., tsk], Y_hat_[..., tsk])])[..., np.newaxis])
    scores_ = np.concatenate(scores_, axis = 1).T
    print(scores_.shape)
    return pd.DataFrame(scores_, columns = ['RMSE', 'MAE', 'MBE'],
                                 index   = hearders_)

# Baselines det. error metrics
def _baseline_det_metrics(Y_, Y_hat_, hearders_):
    Y_p_     = []
    Y_hat_p_ = []
    for hrzn in range(Y_hat_.shape[2]):
        Y_p_.append(Y_[..., hrzn])
        Y_hat_p_.append(Y_hat_[..., hrzn])
    Y_p_     = np.concatenate(Y_p_, axis = 0)
    Y_hat_p_ = np.concatenate(Y_hat_p_, axis = 0)
    return _sparse_det_metrics(Y_p_, Y_hat_p_, hearders_)

# Log-Score
def _LogS(Y_, M_hat_, C_hat_):
    # Multivaritate Normal Log Predictive Probability
    def _mvNLPP(Y_, M_hat_, C_hat_):
        N_samples = Y_.shape[0]
        z_ = np.zeros((N_samples,))
        for i_sample in range(N_samples):
            #z_[i_sample] = multivariate_normal(M_hat_[i_sample, ...], Cov_hat_[i_sample, ...], allow_singular = True).logpdf(Y_[i_sample, ...])
            z_[i_sample] = _robust_log_eval_multivariate_normal(M_hat_[i_sample, ...], C_hat_[i_sample, ...], Y_[i_sample, ...])
        return - z_
    # Samples / Tasks / Forecasting horizons
    N_horizons = Y_.shape[-1]
    return np.array([_mvNLPP(Y_[..., i_horizon], M_hat_[..., i_horizon], C_hat_[..., i_horizon]) for i_horizon in range(N_horizons)])

# Probabilistic multivariate forecat metrics
def _multivariate_prob_metrics_dist(Y_, M_, Cov_, S2_, Y_hat_):

    N_samples  = M_.shape[0]
    N_tasks    = M_.shape[1]
    N_horizons = M_.shape[2]

    df_ = pd.DataFrame()
    print(_LogS(Y_, M_, Cov_).shape, _ES(Y_, Y_hat_).shape, _VS(Y_, Y_hat_).shape)
    print(_ES_spatial(Y_, Y_hat_).shape, _VS_spatial(Y_, Y_hat_).shape, _ES_temporal(Y_, Y_hat_).shape, _VS_temporal(Y_, Y_hat_).shape)
    print(_interval_score(Y_, M_, S2_, z = 1.959, alpha = 0.05).shape)
    print(_interval_score(Y_, M_, S2_, z = 1.645, alpha = 0.1).shape)

    # Ignorance Score
    df_['LogS'] = np.sum(_LogS(Y_, M_, Cov_), axis = 0)/(N_tasks*N_horizons)
    # Energy and variogram Score computed in the spatial dimensions
    df_['ES'] = _ES(Y_, Y_hat_)/(N_tasks*N_horizons)
    df_['VS'] = _VS(Y_, Y_hat_)/(N_tasks*N_horizons)
    # 95% Ignorance Score
    df_['IS95'] = _interval_score(Y_, M_, S2_, z     = 1.959,
                                               alpha = 0.05)
    # 90% Ignorance Score
    df_['IS90'] = _interval_score(Y_, M_, S2_, z     = 1.645,
                                               alpha = 0.1)
    return df_.reset_index(drop = True)

# Probabilistic multivariate forecat metrics
def _multisource_prob_metrics_dist(Y_, M_, Cov_, S2_, Y_hat_):

    N_samples  = M_.shape[0]
    N_tasks    = M_.shape[1]
    N_horizons = M_.shape[2]

    # Exclude solar time series from morning and evening horizons
    # morning:   0 - 7
    # afternoon: 8 - 15
    # evening:  16 - 23

    if N_tasks < 3:
        resource_idxs_ = [[0], [0, 1], [0]]
    else:
        resource_idxs_ = [[0, 2], [0, 1, 2], [0, 2]]
    dayzone_idxs_ = [[0, 10], [10, 14], [14, 24]]

    LogS_ = np.zeros((N_samples,))
    ES_   = np.zeros((N_samples,))
    VS_   = np.zeros((N_samples,))
    IS95_ = np.zeros((N_samples,))
    IS90_ = np.zeros((N_samples,))
    N_dim = 0
    
    for i in range(3):
        dayzone_idx_  = dayzone_idxs_[i]
        resource_idx_ = resource_idxs_[i]

        Y_p_     = Y_[:, resource_idx_, dayzone_idx_[0]:dayzone_idx_[1]].copy()
        M_p_     = M_[:, resource_idx_, dayzone_idx_[0]:dayzone_idx_[1]].copy()
        Cov_p_   = Cov_[..., resource_idx_, dayzone_idx_[0]:dayzone_idx_[1]].copy()
        Cov_p_   = Cov_p_[:, resource_idx_, ...]
        S2_p_    = S2_[:, resource_idx_, dayzone_idx_[0]:dayzone_idx_[1]].copy()
        Y_hat_p_ = Y_hat_[:, resource_idx_, dayzone_idx_[0]:dayzone_idx_[1], :].copy()

        # Number of evaluated dimension
        N_dim += len(resource_idx_)*(dayzone_idx_[1] - dayzone_idx_[0])
        # Logarithmic Score
        LogS_ += np.sum(_LogS(Y_p_, M_p_, Cov_p_), axis = 0)
        # Energy Score
        ES_ += _ES(Y_p_, Y_hat_p_)
        # Variogram Score
        VS_ += _VS(Y_p_, Y_hat_p_)
        # 95% Ignorance Score
        IS95_ += _interval_score(Y_p_, M_p_, S2_p_, z     = 1.959,
                                                    alpha = 0.05)
        # 90% Ignorance Score
        IS90_ += _interval_score(Y_p_, M_p_, S2_p_, z     = 1.645,
                                                    alpha = 0.1)

    # Expected scores
    df_         = pd.DataFrame()
    df_['LogS'] = LogS_/N_dim
    df_['ES']   = ES_/N_dim
    df_['VS']   = VS_/N_dim
    df_['IS95'] = IS95_/N_dim
    df_['IS90'] = IS90_/N_dim

    return df_.reset_index(drop = True)

# Baselines det. error metrics
def _baseline_det_metrics_dist(Y_, Y_hat_, headers_, model):
    scores_ = []
    for smpl in range(Y_hat_.shape[0]):
        for tsk in range(Y_hat_.shape[1]):
            score_ = [smpl,
                      headers_[tsk],
                      model,
                      _RMSE(Y_[smpl, tsk, ...], Y_hat_[smpl, tsk, ...]),
                      _MAE(Y_[smpl, tsk, ...], Y_hat_[smpl, tsk, ...]),
                      _MBE(Y_[smpl, tsk, ...], Y_hat_[smpl, tsk, ...])]

            scores_.append(pd.DataFrame(score_, index = ['sample', 'zone', 'model', 'RMSE', 'MAE', 'MBE']).T)

    return pd.concat(scores_, axis = 0).reset_index(drop = True)

# Gaussian kernel filtering
def _smooth_gaussian(X_, sigma):
    N_zone, N_dim, N_scen = X_.shape
    Y_ = np.zeros(X_.shape)
    for i_zone in range(N_zone):
        for i_scen in range(N_scen):
            Y_[i_zone, :, i_scen] = gaussian_filter1d(X_[i_zone, :, i_scen], sigma)
    return Y_

# Calibrate scenarios to minimize a proper scoring rule
def _calibrate_scenarios_temporal_structure_fit(Y_, Y_hat_, params_, score   = 'ES',
                                                                     verbose = False):
    # Average variograme score
    def __VS(Y_, Y_hat_):
        VS_ = _VS_temporal(Y_, Y_hat_, p = .5)
        VS_ = np.mean(VS_, axis = 1)
        VS_ = np.mean(VS_, axis = 0)
        return VS_
    # Average energy score
    def __ES(Y_, Y_hat_):
        ES_ = _ES_temporal(Y_, Y_hat_)
        ES_ = np.mean(ES_, axis = 1)
        ES_ = np.mean(ES_, axis = 0)
        return ES_

    # Run 24-hours calibration
    def __run_calibration(Y_, Y_hat_, params_, score, verbose):

        best_sigma = 0.
        best_score = np.inf
        for sigma in params_:
            Y_scens_ = np.zeros(Y_hat_.shape)
            for i_sample in range(Y_hat_.shape[0]):
                # Smooth scenarios
                Y_scens_[i_sample, ...] = _smooth_gaussian(Y_hat_[i_sample, ...], sigma = sigma)

            # Compute proper scoring rules
            ES  = __ES(Y_, Y_scens_)
            VS  = __VS(Y_, Y_scens_)
            df_ = pd.DataFrame([ES, VS], columns = ['score'],
                                         index   = ['ES', 'VS'])
            # Display interation results
            if verbose:
                print(sigma, ES, VS)

            # Stop when minimum is found
            if df_.loc[score, 'score'] < best_score:
                best_score    = df_.loc[score, 'score']
                best_sigma    = sigma
            else:
                break

        return sigma

    N_samples, N_zones, N_dim, N_scen = Y_hat_.shape
    sigma_  = np.zeros((N_zones, ))
    for i_zone in range(N_zones):
        sigma_[i_zone] = __run_calibration(Y_[:, i_zone, :][:, np.newaxis, :], Y_hat_[:, i_zone, ...][:, np.newaxis, ...], params_, score, verbose)
    return sigma_

# Calibrate scenarios given a set of sigmas
def _calibrate_scenarios_temporal_structure(Y_hat_, sigmas_):

    N_samples, N_zones, N_dim, N_scen = Y_hat_.shape
    Y_scen_ = np.zeros(Y_hat_.shape)

    for i_zone in range(N_zones):
        for i_sample in range(N_samples):
            # Smooth scenarios
            Y_scen_[i_sample, i_zone, ...] = _smooth_gaussian(Y_hat_[i_sample, i_zone, ...][np.newaxis, ...], sigma = sigmas_[i_zone])

    return Y_scen_

__all__ = ['_RMSE',
           '_MAE',
           '_MBE',
           '_CRPS',
           '_VS_spatial',
           '_VS_temporal',
           '_ES_temporal',
           '_ES_spatial',
           '_LogS',
           '_sparse_det_metrics',
           '_baseline_det_metrics',
           '_prob_metrics',
           '_multivariate_prob_metrics',
           '_multivariate_prob_metrics_dist',
           '_multisource_prob_metrics_dist',
           '_multiresource_prob_metrics',
           '_baseline_det_metrics_dist',
           '_calibrate_scenarios_temporal_structure_fit',
           '_calibrate_scenarios_temporal_structure']
