import pickle, glob, os, blosc, csv

import numpy as np

from time import sleep
from datetime import datetime, date, timedelta
from itertools import product, chain
from group_lasso import GroupLasso
#from scipy.stats import norm, multivariate_normal
import properscoring as ps

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, OrthogonalMatchingPursuit, Ridge, ARDRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor

GroupLasso.LOG_LOSSES = True

from GP_utils import *

from ngboost import NGBRegressor
from ngboost.scores import LogScore, CRPScore
from ngboost.distns import Normal, MultivariateNormal
from scipy.stats import norm, multivariate_normal

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
def _get_cv_param(alphas_, nus_, betas_, omegas_, gammas_, etas_, lambdas_, xis_, kappas_, sl, dl):
    thetas_ = []
    # Lasso parameters
    if sl == 0:
        thetas_.append(list(alphas_))
    # Orthogonal Matching Pursuit parameters
    if sl == 1:
        thetas_.append(list(betas_))
    # Elastic Net parameters
    if sl == 2:
        thetas_.append(list(nus_))
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

# Gassuain Process for Regression
def _GaussianProcess(X_, Y_, g_, hrzn, xi, max_iter   = 250,
                                           n_init     = 5,
                                           early_stop = 10,
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

# Robust multivariate normal distribution sample generation
def _robust_sampling_from_multivariate_normal_v1(m_hat_, C_hat_, N_scenarios):
    try:
        I_ = np.linalg.pinv(C_hat_)
    except:
        print('Adding jitter...')
        epsilon = np.min(np.real(np.linalg.eigvals(C_hat_)))
        C_hat_ -= 10*epsilon * np.eye(C_hat_.shape[0])
    return np.random.multivariate_normal(m_hat_, C_hat_, N_scenarios, check_valid = "ignore")

# Robust multivariate normal distribution sample generation
def _robust_sampling_from_multivariate_normal_v2(m_hat_, C_hat_, N_scenarios):
    try:
        return multivariate_normal(m_hat_, C_hat_, allow_singular = True).rvs(N_scenarios).T
    except:
        print('Adding jitter...')
        epsilon = np.min(np.real(np.linalg.eigvals(C_hat_)))
        return multivariate_normal(m_hat_, C_hat_ - 10*epsilon * np.eye(C_hat_.shape[0]),
                                   allow_singular = True).rvs(N_scenarios).T

# Draw random samples from a multivariate normal distribution
def _sample_multivariate_normal(m_hat_, C_hat_, N_scenarios = 1):
    # Constant definition
    N_samples = m_hat_.shape[0]
    N_tasks   = m_hat_.shape[1]
    if N_scenarios == 1: Z_ = np.zeros((N_samples, N_tasks))
    else:                Z_ = np.zeros((N_samples, N_tasks, N_scenarios))
    for i_sample in range(N_samples):
        Z_[i_sample, ...] = _robust_sampling_from_multivariate_normal_v1(m_hat_[i_sample, :], C_hat_[i_sample, ...], N_scenarios)
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

# Fit sparse learning model
def _fit_sparse_learning(X_sl_tr_stnd_, X_sl_ts_stnd_, Y_sl_tr_stnd_, Y_sl_ts_, g_sl_, thetas_, sl_scaler_, SL, y_sl_stnd,
                         verbose = True):
    # Initialization prediction mean and weights
    W_hat_       = np.ones((X_sl_ts_stnd_[:, g_sl_ != g_sl_[-1]].shape[1], Y_sl_ts_.shape[1]))
    Y_sl_ts_hat_ = np.zeros(Y_sl_ts_.shape)
    if SL != 4:
        if verbose: print(' Sparse learning model training...')
        # Train independent multi-task models for each hour
        for tsk in range(Y_sl_ts_.shape[1]):
            if verbose: print('     Task', tsk, '...')
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

# Fit dense learning - Bayesian model chain
def _fit_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_, W_hat_, g_dl_, thetas_, RC, DL, key = ''):
    print(' Dense learning model training...')
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

# Fit multitask dense learning - Bayesian model chain
def _fit_multitask_dense_learning(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_, W_hat_, g_dl_, thetas_, RC, DL,
                                  key     = '',
                                  verbose = True):
    if verbose: print(' Multi-task dense learning model training...')
    # Initialization multi-task predictive mean
    Y_dl_tr_hat_ = np.zeros(Y_dl_tr_.shape)
    models_      = []
    # Train an expert models for each hour
    for hrzn in range(Y_dl_tr_hat_.shape[2]):
        if verbose: print('     Horion', hrzn, '...')
        tsk = 0
        # Define training and testing recursive dataset
        X_dl_tr_rc_, Y_dl_tr_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_tr_stnd_, Y_dl_tr_stnd_, Y_dl_tr_hat_, g_dl_, W_hat_, RC, hrzn, tsk = None)
        #print(hrzn, X_dl_tr_rc_.shape, Y_dl_tr_rc_.shape, g_dl_rc_.shape)
        # Multi-Task Gaussian Process for Regression with Hyperprior
        _DL = _MultiTaskGaussianProcess(X_dl_tr_rc_, Y_dl_tr_rc_, g_dl_rc_, xi = thetas_[tsk][-1])
        # Make prediction for multi-task recursive model
        Y_dl_tr_hat_[..., hrzn], _ = _dense_learning_multitask_predict(_DL, X_dl_tr_rc_, DL)
        models_.append(_DL)

    return models_

# Calibrate Predictive probabilistic distribution
def _calibrate_predictive_covariance_fit(M_dl_ts_hat_, C_dl_ts_hat_, Y_dl_ts_):
    # Train an expert models for each hour
    N_samples = Y_dl_ts_.shape[0]
    N_task    = Y_dl_ts_.shape[1]
    N_hours   = Y_dl_ts_.shape[2]
    L_hat_    = []
    for i_hour in range(N_hours):
        # If calibration calculate lambda
        S_hat_ = []
        L_     = []
        for i_sample in range(N_samples):
            # Calculate optimal predictive covariance matrix
            y_      = Y_dl_ts_[i_sample, :, i_hour]
            mu_hat_ = M_dl_ts_hat_[i_sample, :, i_hour]
            diff_   = (y_ - mu_hat_)[:, np.newaxis]
            # Save predictive covariance matrix
            S_hat_.append(C_dl_ts_hat_[i_sample, :, :, i_hour].flatten())
            # Save optimal predictive covariance matrix
            L_.append((diff_ @ diff_.T).flatten())
        # Unbiese predictive covariance matrix
        S_hat_ = np.stack(S_hat_)
        S_hat_ = np.concatenate([S_hat_, np.ones((S_hat_.shape[0], 1))], axis = 1)
        L_     = np.stack(L_)
        L_hat_.append(np.linalg.pinv(S_hat_.T @ S_hat_) @ S_hat_.T @ L_)

    return np.stack(L_hat_)

# Unbias predictive covariance matrix
def _unbias_covariance(l_hat_, C_hat_):
    # Constants definition
    N_task = C_hat_.shape[0]
    # Covariance matrix to vector with bias
    c_hat_ = np.concatenate([C_hat_.flatten(), np.ones((1,))], axis = 0)
    # Correct bias on the covariance
    l_bar_ = l_hat_.T @ c_hat_
    # Reshape predicted covariance to matrix shape
    L_bar_ = l_bar_.reshape(N_task, N_task)
    try:
        # Trye to invert matrix
        I_ = np.linalg.pinv(L_bar_)
    except:
        print('Singular matrix -> Restoring covariance matrix...')
        L_bar_ = C_hat_
    return L_bar_

# Calibrate the predictive covariance matrix
def _calibrate_predictive_covariance(M_dl_ts_hat_, C_dl_ts_hat_, L_hat_):
    # Contants definition
    N_samples  = C_dl_ts_hat_.shape[0]
    N_task     = C_dl_ts_hat_.shape[1]
    N_horizons = C_dl_ts_hat_.shape[-1]
    # Variables definition
    l_bar_     = np.zeros((N_samples, N_task, N_horizons))
    L_bar_all_ = []
    for i_sample in range(N_samples):
        L_bar_ = []
        for i_horizon in range(N_horizons):
            l_hat_ = L_hat_[i_horizon, ...]
            C_hat_ = C_dl_ts_hat_[i_sample, ..., i_horizon]
            # Unbias predictive covariance matrix
            L_bar_.append(_unbias_covariance(l_hat_, C_hat_))
        L_bar_all_.append(np.stack(L_bar_))
    # Rearrange dimensions order in the convariance tensor
    L_bar_ = np.moveaxis(np.stack(L_bar_all_), [1], [3])
    # Get variance from the convariance matrix
    for i_sample in range(N_samples):
        for i_horizon in range(N_horizons):
            try:
                z = multivariate_normal(M_dl_ts_hat_[i_sample, ..., i_horizon], L_bar_[i_sample, ..., i_horizon], allow_singular = True).rvs(1)
                l_bar_[i_sample, :, i_horizon]   = np.diag(L_bar_[i_sample, ..., i_horizon])
            except:
                L_bar_[i_sample, ..., i_horizon] = C_dl_ts_hat_[i_sample, ..., i_horizon]
                l_bar_[i_sample, :, i_horizon]   = np.diag(C_dl_ts_hat_[i_sample, ..., i_horizon])
    return l_bar_, L_bar_

# Predictive probabilistic distribution
def _pred_prob_dist(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd):
    #(24, 3, 3) (3, 3) | (24, 3, 24) (8, 3, 24) (3, 24) - Multitask samples
    # Constants definition
    N_samples  = Y_dl_ts_.shape[0]
    N_tasks    = Y_dl_ts_.shape[1]
    N_horizons = Y_dl_ts_.shape[2]
    # Variables definition
    Y_dl_ts_stnd_hat_ = np.zeros(Y_dl_ts_.shape)
    M_dl_ts_hat_      = np.zeros(Y_dl_ts_.shape)
    S2_dl_ts_hat_     = np.zeros(Y_dl_ts_.shape)
    C_dl_ts_hat_      = np.zeros((N_samples, N_tasks, N_tasks, N_horizons))
    # Train an expert models for each hour
    for i_horizon in range(N_horizons):
        # Train an expert models for each hour
        for i_task in range(N_tasks):
            # Define training and testing recursive dataset
            X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_                                               = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_, g_dl_, W_hat_, RC, i_horizon, i_task)
            Y_dl_ts_stnd_hat_[..., i_task, i_horizon], S2_dl_ts_hat_[..., i_task, i_horizon] = _dense_learning_predict(models_[i_horizon][i_task], X_dl_ts_rc_, DL)
        # Undo standardization from dense learning multi-task prediction
        if y_dl_stnd == 1:
            # To undo predictive covariance necessary covariance instead of variance
            M_dl_ts_hat_[..., i_horizon]  = dl_scaler_[1][i_horizon].inverse_transform(Y_dl_ts_stnd_hat_[..., i_horizon])
            S2_dl_ts_hat_[..., i_horizon] = dl_scaler_[1][i_horizon].var_*S2_dl_ts_hat_[..., i_horizon]
        for i_sample in range(N_samples):
            C_dl_ts_hat_[i_sample, ..., i_horizon] = np.diag(S2_dl_ts_hat_[i_sample, ..., i_horizon])
    return M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_

# Multitask predictive probabilistic distribution
def _multitask_pred_prob_dist(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, g_dl_, RC, DL, y_dl_stnd):
    #(24, 3, 3) (3, 3) | (24, 3, 24) (8, 3, 24) (3, 24) - Multitask samples
    # Constants definition
    N_samples  = Y_dl_ts_.shape[0]
    N_tasks    = Y_dl_ts_.shape[1]
    N_horizons = Y_dl_ts_.shape[2]
    # Variables definition
    Y_dl_ts_stnd_hat_ = np.zeros(Y_dl_ts_.shape)
    M_dl_ts_hat_      = np.zeros(Y_dl_ts_.shape)
    S2_dl_ts_hat_     = np.zeros(Y_dl_ts_.shape)
    C_dl_ts_hat_      = np.zeros((N_samples, N_tasks, N_tasks, N_horizons))
    # Train an expert models for each hour
    for i_horizon in range(N_horizons):
        # Define training and testing recursive dataset
        X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_                              = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_, g_dl_, W_hat_, RC, i_horizon, tsk = None)
        Y_dl_ts_stnd_hat_[..., i_horizon], C_dl_ts_hat_[..., i_horizon] = _dense_learning_multitask_predict(models_[i_horizon], X_dl_ts_rc_, DL)
        # Undo standardization from dense learning multi-task prediction
        if y_dl_stnd == 1:
            # To undo predictive covariance necessary covariance instead of variance
            Cov_ = np.sqrt(dl_scaler_[1][i_horizon].var_[:, np.newaxis]) @ np.sqrt(dl_scaler_[1][i_horizon].var_[:, np.newaxis].T)
            M_dl_ts_hat_[..., i_horizon] = dl_scaler_[1][i_horizon].inverse_transform(Y_dl_ts_stnd_hat_[..., i_horizon])
            C_dl_ts_hat_[..., i_horizon] = Cov_*C_dl_ts_hat_[..., i_horizon]
    # Get variance from the diagonla
    for i_sample in range(N_samples):
        for i_horizon in range(N_horizons):
            S2_dl_ts_hat_[i_sample, :, i_horizon] = np.diag(C_dl_ts_hat_[i_sample, ..., i_horizon])
    return M_dl_ts_hat_, S2_dl_ts_hat_, C_dl_ts_hat_

# Joint probabilistic predictions
def _joint_prob_prediction(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, L_hat_, g_dl_, RC, DL, y_dl_stnd,
                           N_scenarios = 100,
                           calibration = True):
    # Constants definition
    N_samples  = Y_dl_ts_.shape[0]
    N_tasks    = Y_dl_ts_.shape[1]
    N_horizons = Y_dl_ts_.shape[2]
    # Variables definition
    Y_dl_ts_stnd_hat_ = np.zeros((N_samples, N_tasks, N_horizons, N_scenarios))
    Y_dl_ts_hat_      = np.zeros((N_samples, N_tasks, N_horizons, N_scenarios))
    M_dl_ts_hat_      = np.zeros((N_samples, N_tasks))
    S2_dl_ts_hat_     = np.zeros((N_samples, N_tasks))
    # Draw samples from predictive posterior
    for i_scenario in range(N_scenarios):
        # Train an expert models for each hour
        for i_horizon in range(N_horizons):
            # Train an expert models for each hour
            for i_task in range(N_tasks):
                # Define training and testing recursive dataset
                X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_                    = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_[..., i_scenario], g_dl_, W_hat_, RC, i_horizon, i_task)
                M_dl_ts_hat_[..., i_task], S2_dl_ts_hat_[..., i_task] = _dense_learning_predict(models_[i_horizon][i_task], X_dl_ts_rc_, DL)
            # Undo standardization from dense learning multi-task prediction
            if y_dl_stnd == 1:
                # To undo predictive covariance necessary covariance instead of variance
                M_dl_ts_hat_  = dl_scaler_[1][i_horizon].inverse_transform(M_dl_ts_hat_)
                S2_dl_ts_hat_ = dl_scaler_[1][i_horizon].var_*S2_dl_ts_hat_
            # Construct covariance matrix from vector of variances
            C_dl_ts_hat_ = np.concatenate([np.diag(S2_dl_ts_hat_[i_sample, :])[np.newaxis, ...] for i_sample in range(N_samples)], axis = 0)
            # Unbias covariance matrix
            L_dl_ts_hat_ = C_dl_ts_hat_.copy()
            if calibration:
                for i_sample in range(N_samples):
                    L_dl_ts_hat_[i_sample, ...] = _unbias_covariance(L_hat_[i_horizon, ...], C_dl_ts_hat_[i_sample, ...])
            # Sample Predictive Posterior Distribution
            Y_dl_ts_hat_[..., i_horizon, i_scenario] = _sample_multivariate_normal(M_dl_ts_hat_, L_dl_ts_hat_)
            # Standardized predictors for recursive model
            if y_dl_stnd == 1:
                Y_dl_ts_stnd_hat_[..., i_horizon, i_scenario] = dl_scaler_[1][i_horizon].transform(Y_dl_ts_hat_[..., i_horizon, i_scenario])
    return Y_dl_ts_hat_

# Multitask joint prediction
def _multitask_joint_prediction(models_, dl_scaler_, X_dl_ts_stnd_, Y_dl_ts_, W_hat_, L_hat_, g_dl_, RC, DL, y_dl_stnd,
                                N_scenarios = 100,
                                calibration = True):
    # Constants definition
    N_samples  = Y_dl_ts_.shape[0]
    N_tasks    = Y_dl_ts_.shape[1]
    N_horizons = Y_dl_ts_.shape[2]
    # Variables definition
    Y_dl_ts_stnd_hat_ = np.zeros((N_samples, N_tasks, N_horizons, N_scenarios))
    Y_dl_ts_hat_      = np.zeros((N_samples, N_tasks, N_horizons, N_scenarios))
    # Draw samples from predictive posterior
    for i_scenario in range(N_scenarios):
        # Train an expert models for each hour
        for i_horizon in range(N_horizons):
            # Define training and testing recursive dataset
            X_dl_ts_rc_, Y_dl_ts_rc_, g_dl_rc_ = _dense_learning_recursive_dataset(X_dl_ts_stnd_, Y_dl_ts_, Y_dl_ts_stnd_hat_[..., i_scenario], g_dl_, W_hat_, RC, i_horizon, tsk = None)
            M_dl_ts_hat_, C_dl_ts_hat_         = _dense_learning_multitask_predict(models_[i_horizon], X_dl_ts_rc_, DL)
            # Undo standardization from dense learning multi-task prediction
            if y_dl_stnd == 1:
                # To undo predictive covariance necessary covariance instead of variance
                Cov_          = np.sqrt(dl_scaler_[1][i_horizon].var_[:, np.newaxis]) @ np.sqrt(dl_scaler_[1][i_horizon].var_[:, np.newaxis].T)
                M_dl_ts_hat_  = dl_scaler_[1][i_horizon].inverse_transform(M_dl_ts_hat_)
                C_dl_ts_hat_  = Cov_*C_dl_ts_hat_
            # Unbias covariance matrix
            L_dl_ts_hat_ = C_dl_ts_hat_.copy()
            if calibration:
                for i_sample in range(N_samples):
                    L_dl_ts_hat_[i_sample, ...] = _unbias_covariance(L_hat_[i_horizon, ...], C_dl_ts_hat_[i_sample, ...])
            # Sample Predictive Posterior Distribution
            Y_dl_ts_hat_[..., i_horizon, i_scenario] = _sample_multivariate_normal(M_dl_ts_hat_, L_dl_ts_hat_)
            # Standardized predictors for recursive model
            if y_dl_stnd == 1:
                Y_dl_ts_stnd_hat_[..., i_horizon, i_scenario] = dl_scaler_[1][i_horizon].transform(Y_dl_ts_hat_[..., i_horizon, i_scenario])
    return Y_dl_ts_hat_

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
           '_fit_dense_learning',
           '_fit_sparse_learning',
           '_fit_multitask_dense_learning',
           '_pred_prob_dist',
           '_multitask_pred_prob_dist',
           '_calibrate_predictive_covariance_fit',
           '_calibrate_predictive_covariance',
           '_joint_prob_prediction',
           '_multitask_joint_prediction']
