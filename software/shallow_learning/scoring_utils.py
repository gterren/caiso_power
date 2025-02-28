import pickle, glob, os, blosc, csv

import numpy as np
import pandas as pd

from time import sleep
from datetime import datetime, date, timedelta

from itertools import product, chain

from scipy.stats import norm, multivariate_normal

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
    N_in = 0.
    for i_zone in range(M_.shape[1]):
        for i_hour in range(M_.shape[2]):
            y_    = Y_[:, i_zone, i_hour]
            mu    = M_[:, i_zone, i_hour]
            std   = np.sqrt(S2_[:, i_zone, i_hour])
            idx_  = (y_ >= (mu - z*std)) & (y_ <= (mu + z*std))
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
                                                          index   = ['NP15', 'SP15', 'ZP26'][:Y_.shape[1]])

# Log-Score
def _LogS(Y_, M_hat_, Cov_hat_):
    # Multivaritate Normal Log Predictive Probability
    def _mvNLPP(Y_, M_hat_, Cov_hat_):
        z_ = np.zeros((Y_.shape[0], ))
        for obv in range(M_hat_.shape[0]):
            z_[obv] = multivariate_normal(M_hat_[obv, ...], Cov_hat_[obv, ...], allow_singular = True).logpdf(Y_[obv, ...])
        return - z_
    # Samples / Tasks / Forecasting horizons
    return np.array([_mvNLPP(Y_[..., hrzn], M_hat_[..., hrzn], Cov_hat_[..., hrzn]) for hrzn in range(Y_.shape[-1])])

# Log Score
def _agg_LogS(Y_, M_hat_, Cov_hat_):
    # Multivaritate Normal Log Predictive Probability
    def _mvNLPP(Y_, M_hat_, Cov_hat_):
        z_ = np.zeros((Y_.shape[0], ))
        for obv in range(M_hat_.shape[0]):
            z_[obv] = multivariate_normal(M_hat_[obv, ...], Cov_hat_[obv, ...], allow_singular = True).logpdf(Y_[obv, ...])
        return - z_.sum()
    # Samples / Tasks / Forecasting horizons
    return np.array([_mvNLPP(Y_[..., hrzn], M_hat_[..., hrzn], Cov_hat_[..., hrzn]) for hrzn in range(Y_.shape[-1])])

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
def _LogS(Y_, M_hat_, Cov_hat_):
    # Multivaritate Normal Log Predictive Probability
    def _mvNLPP(Y_, M_hat_, Cov_hat_):
        z_ = np.zeros((Y_.shape[0], ))
        for obv in range(M_hat_.shape[0]):
            z_[obv] = multivariate_normal(M_hat_[obv, ...], Cov_hat_[obv, ...], allow_singular = True).logpdf(Y_[obv, ...])
        return - z_
    # Samples / Tasks / Forecasting horizons
    return np.sum(np.array([_mvNLPP(Y_[..., hrzn], M_hat_[..., hrzn], Cov_hat_[..., hrzn]) for hrzn in range(Y_.shape[-1])]), axis = 0)

# Probabilistic multivariate forecat metrics
def _multivariate_prob_metrics(Y_, M_, Cov_, S2_, Y_hat_):
    # Multivariate proper scoring rules
    LogS_ = _LogS(Y_, M_, Cov_).sum()/(M_.shape[0]*M_.shape[1]*M_.shape[2])
    VS_   = _VS(Y_, Y_hat_).sum()/(M_.shape[0]*M_.shape[1]*M_.shape[2])
    ES_   = _ES(Y_, Y_hat_).sum()/(M_.shape[0]*M_.shape[1]*M_.shape[2])
    # Compute Ignorance Score at differet confidence levels
    IS975_ = _interval_score(Y_, M_, S2_, z = 2.243, alpha = 0.025).sum()/(M_.shape[0]*M_.shape[1]*M_.shape[2])
    IS95_  = _interval_score(Y_, M_, S2_, z = 1.959, alpha = 0.05).sum()/(M_.shape[0]*M_.shape[1]*M_.shape[2])
    IS90_  = _interval_score(Y_, M_, S2_, z = 1.645, alpha = 0.1).sum()/(M_.shape[0]*M_.shape[1]*M_.shape[2])
    IS80_  = _interval_score(Y_, M_, S2_, z = 1.282, alpha = 0.2).sum()/(M_.shape[0]*M_.shape[1]*M_.shape[2])
    IS60_  = _interval_score(Y_, M_, S2_, z = 0.842, alpha = 0.4).sum()/(M_.shape[0]*M_.shape[1]*M_.shape[2])
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
def _multivariate_prob_metrics_dist(Y_, M_, Cov_, S2_, Y_hat_):
    df_ = pd.DataFrame()

    # Ignorance Score
    df_['LogS'] = _LogS(Y_, M_, Cov_)/(M_.shape[1]*M_.shape[2])

    # Energy and variogram Score computed in the spatial dimensions
    df_['ES'] = _ES(Y_, Y_hat_)/(M_.shape[1]*M_.shape[2])
    df_['VS'] = _VS(Y_, Y_hat_)/(M_.shape[1]*M_.shape[2])

    # Energy and variogram Score computed in the temporal dimensions
    df_['ES_temporal'] = np.mean(_ES_temporal(Y_, Y_hat_), axis = 0)
    df_['VS_temporal'] = np.mean(_VS_temporal(Y_, Y_hat_), axis = 0)

    # Energy and variogram Score computed in the spatial dimensions
    df_['ES_spatial'] = np.mean(_ES_spatial(Y_, Y_hat_), axis = 0)
    df_['VS_spatial'] = np.mean(_VS_spatial(Y_, Y_hat_), axis = 0)
    
    # Energy and variogram Score computed in the temporal dimensions
    df_['ES_temporal'] = np.mean(_ES_temporal(Y_, Y_hat_), axis = 0)
    df_['VS_temporal'] = np.mean(_VS_temporal(Y_, Y_hat_), axis = 0)

    # Compute number of samples within confidence interval
    df_['IS95'] = np.mean(_interval_score(Y_, M_, S2_, z = 1.959, alpha = 0.05), axis = 0)
    df_['IS90'] = np.mean(_interval_score(Y_, M_, S2_, z = 1.645, alpha = 0.1), axis = 0)

    df_['sample'] = np.linspace(0, df_['VS_temporal'].shape[0] - 1, df_['VS_temporal'].shape[0], dtype = int)
    return df_.reset_index(drop = True)

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
                                 index   = ['NP15', 'SP15', 'ZP26'][:Y_.shape[1]])

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

# Baselines det. error metrics
def _baseline_det_metrics_dist(Y_, Y_hat_, model):
    zones_  = ['NP15', 'SP15', 'ZP26']
    scores_ = []
    for smpl in range(Y_hat_.shape[0]):
        for tsk in range(Y_hat_.shape[1]):
            score_ = [smpl,
                      zones_[tsk],
                      model,
                      _RMSE(Y_[smpl, tsk, ...], Y_hat_[smpl, tsk, ...]),
                      _MAE(Y_[smpl, tsk, ...], Y_hat_[smpl, tsk, ...]),
                      _MBE(Y_[smpl, tsk, ...], Y_hat_[smpl, tsk, ...])]

            scores_.append(pd.DataFrame(score_, index = ['sample', 'zone', 'model', 'RMSE', 'MAE', 'MBE']).T)

    return pd.concat(scores_, axis = 0).reset_index(drop = True)

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
           '_baseline_det_metrics_dist']
