import gpytorch, torch

import numpy as np
import scipy as sp

from scipy.linalg import inv

# Import kernels for the Cool-MTGP model.
from lib.kernels import DotProduct, RBF, WhiteKernel, RationalQuadratic, Matern, ConstantKernel

# Import approximate Cool-MTGP (~Cool-MTGP)
from lib.A_Cool_MTGP import MultitaskGP as a_mtgp

# Import hierarchical Cool-MTGP (HCool-MTGP)
from lib.H_Cool_MTGP import MultitaskGP as h_mtgp

# import sklearn single GP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.gaussian_process.kernels import RationalQuadratic, ExpSineSquared, DotProduct


def _random(low = -2.5, high = 2.5):
    return torch.tensor(np.exp(float(np.random.uniform(low, high, size = 1)[0])))

# Fit Gaussian process using SkLearn
def _skGPR_fit(X_, y_, g_, param_):

    kernel, degree, RC, hrzn, max_training_iter, n_random_init, early_stop = param_
    # Linear kernel
    if kernel == 'linear':
       _kernel = DotProduct(sigma_0        = 1.,
                                                                                                       sigma_0_bounds = (1e-10, 1e10)) + WhiteKernel(noise_level = 1., noise_level_bounds = (1e-10, 1e10))
    # Order 2 Polynomial kernel
    if kernel == 'poly':
        _kernel = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-10, 1e10)) * DotProduct(sigma_0        = 1.,
                                                                                                        sigma_0_bounds = (1e-10, 1e10))**degree
    # Radial basis funtions kernel
    if kernel == 'RBF':
        _kernel = RBF(length_scale        = 1.,
                                                                                                 length_scale_bounds = (1e-10, 1e10)) + WhiteKernel(noise_level = 1., noise_level_bounds = (1e-10, 1e10))
    if kernel == 'RQ':
        # Rational Quadratic kernel
        _kernel = RationalQuadratic(length_scale        = 1.,
                                                                                                                 alpha               = 0.1,
                                                                                                                 length_scale_bounds = (1e-10, 1e10),
                                                                                                                 alpha_bounds        = (1e-10, 1e10)) + WhiteKernel(noise_level = 1., noise_level_bounds = (1e-10, 1e10))
    if kernel == 'matern':
        # Matern Kernel with nu hyperparameter set to 0.5
        _kernel = Matern(length_scale        = 1.0,
                                                                                                      length_scale_bounds = (1e-10, 1e10),
                                                                                                      nu                  = degree) + WhiteKernel(noise_level = 1., noise_level_bounds = (1e-10, 1e10))
    # Training Gaussian process for regression
    return GaussianProcessRegressor(kernel               = _kernel,
                                    n_restarts_optimizer = 9).fit(X_, y_)

# Gaussian Process for Regression
class _GPR(gpytorch.models.ExactGP):
    def __init__(self, X_, y_, g_, _like, kernel, degree, RC, hrzn, random_init = True, multiple_length_scales = False):
        super(_GPR, self).__init__(X_, y_, _like)
        self.mean_module = gpytorch.means.ConstantMean()
        # Random Parameters Initialization
        self.random_init            = random_init
        self.multiple_length_scales = multiple_length_scales
        # Define features index
        idx_dim_  = torch.linspace(0, g_.shape[0] - 1, g_.shape[0], dtype = int)
        # Treat features and index independently
        idx_      = idx_dim_[g_ != torch.unique(g_)[-1]]
        idx_bias_ = idx_dim_[g_ == torch.unique(g_)[-1]]

        # Define features kernel
        _K = self.__define_kernel(kernel, degree, idx_dim_ = idx_)

        # Define constant kernel for bias
        _K_bias = self.__define_kernel(kernel   = 'linear',
                                       degree   = 0,
                                       idx_dim_ = idx_bias_)
        # Multiple-kernel learning
        if (RC == 1) and (hrzn > 0):
            idx_rc_ = idx_dim_[g_ == torch.unique(g_)[-2]]
            # Define kernel for recursive predictions
            _K_chain = self.__define_kernel(kernel   = 'linear',
                                            degree   = 0,
                                            idx_dim_ = idx_rc_)
            # Combine features and bias kernels
            self.covar_module = _K + _K_chain + _K_bias
        else:
            # Combine features and bias kernels
            self.covar_module = _K + _K_bias

    # Define a kernel
    def __define_kernel(self, kernel, degree, idx_dim_ = None):
        if self.multiple_length_scales:
            dim = int(idx_dim_.shape[0])
        else:
            dim = None
        # Random Initialization Covariance Matrix
        if self.random_init:
            self.likelihood.noise_covar.raw_noise.data.fill_(self.likelihood.noise_covar.raw_noise_constraint.inverse_transform(_random()))
        if self.random_init:
            self.mean_module.constant.data.fill_(_random())
        # Linear kernel
        if kernel == 'linear':
            _K = gpytorch.kernels.LinearKernel(active_dims = idx_dim_)
            # Linear kernel parameter
            if self.random_init: _K.raw_variance.data.fill_(_K.raw_variance_constraint.inverse_transform(_random()))
            return _K
        # Radian Basis Function Kernel
        if kernel == 'RBF':
            _K = gpytorch.kernels.RBFKernel(active_dims  = idx_dim_,
                                            ard_num_dims = dim)
            # RBF Kernel parameter
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]):
                    _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        # Polynomial Expansion Kernel
        if kernel == 'poly':
            _K = gpytorch.kernels.PolynomialKernel(power       = degree,
                                                   active_dims = idx_dim_)
            # Polynomial Kernel parameter
            if self.random_init:
                _K.raw_offset.data.fill_(_K.raw_offset_constraint.inverse_transform(_random()))
        # Matern Kernel
        if kernel == 'matern':
            _K = gpytorch.kernels.MaternKernel(nu           = degree,
                                               active_dims  = idx_dim_,
                                               ard_num_dims = dim)
            # Matern Kernel parameter
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        # Rational Quadratic Kernel
        if kernel == 'RQ':
            _K = gpytorch.kernels.RQKernel(active_dims  = idx_dim_,
                                           ard_num_dims = dim)
            # RQ Kernel parameters
            if self.random_init:
                _K.raw_alpha.data.fill_(_K.raw_alpha_constraint.inverse_transform(_random()))
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]):
                    _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        _K = gpytorch.kernels.ScaleKernel(_K)
        # Amplitude coefficient parameter
        if self.random_init:
            _K.raw_outputscale.data.fill_(_K.raw_outputscale_constraint.inverse_transform(_random()))
        return _K

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Select the best model using multiple initializations
def _GPR_fit(X_, y_, g_, param_):
    # Gaussian Process Regression model fit...
    def __fit(X_, y_, g_, params_, random_init = True):
        # Optimize Kernel hyperparameters
        def __optimize(_model, _like, X_, y_, max_training_iter, early_stop):
            # Storage Variables Initialization
            nmll_ = []
            # Find optimal model hyperparameters
            _model.train()
            # Use the adam optimizer
            _optimizer = torch.optim.Adam(_model.parameters(), lr = .01)  # Includes GaussianLikelihood parameters
            # "Loss" for GPs - the marginal log likelihood
            _mll = gpytorch.mlls.ExactMarginalLogLikelihood(_like, _model)
            # Begins Iterative Optimization
            for i in range(max_training_iter):
                # Zero gradients from previous iteration
                _optimizer.zero_grad()
                # Output from model
                f_hat_ = _model(X_)
                # Calc loss and backprop gradients
                _nmll = - _mll(f_hat_, y_)
                _nmll.backward()
                _optimizer.step()
                #print(i, np.around(float(_nmll.detach().numpy()), 2))
                nmll_.append(np.around(float(_nmll.detach().numpy()), 2) )
                if np.isnan(nmll_[-1]):
                    return _model, _like, np.inf
                if i > early_stop:
                    if np.unique(nmll_[-early_stop:]).shape[0] == 1:
                        break
            return _model, _like, nmll_[-1]

        kernel, degree, RC, hrzn, max_training_iter, n_random_init, early_stop = params_
        # Add dummy feature for the bias
        X_ = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
        g_ = np.concatenate((g_, np.ones((1,))*np.unique(g_)[-1] + 1), axis = 0)
        # Numpy yo pyTorch
        X_p_ = torch.tensor(X_, dtype = torch.float)
        y_p_ = torch.tensor(y_, dtype = torch.float)
        g_p_ = torch.tensor(g_, dtype = torch.float)
        # initialize likelihood and model
        _like  = gpytorch.likelihoods.GaussianLikelihood()
        _model = _GPR(X_p_, y_p_, g_p_, _like, kernel, degree, RC, hrzn, random_init)
        return __optimize(_model, _like, X_p_, y_p_, max_training_iter, early_stop)

    kernel, degree, RC, hrzn, max_training_iter, n_random_init, early_stop = param_
    # Storage Variables Initialization
    model_ = []
    nmll_  = []
    # No Random Initialization
    _model, _like, nmll = __fit(X_, y_, g_, param_, random_init = False)
    # Get Results
    model_.append([_model, _like])
    nmll_.append(nmll)
    # Perform multiple Random Initializations
    for i in range(n_random_init):
        _model, _like, nmll = __fit(X_, y_, g_, param_, random_init = True)
        # Get Results
        model_.append([_model, _like])
        nmll_.append(nmll)
    # Best Results of all different Initialization
    _model, _like = model_[np.argmin(nmll_)]
    nmll          = nmll_[np.argmin(nmll_)]
    return [_model, _like, nmll]

# Calculating prediction for new sample
def _GPR_predict(GP_, X_):
    _model, _like, nmll = GP_
    X_   = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
    X_p_ = torch.tensor(X_, dtype = torch.float)
    _model.eval()
    _like.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        _f_hat = _like(_model(X_p_))
        return _f_hat.mean.numpy(), np.sqrt(_f_hat.variance.numpy()), np.sqrt(_like.noise.numpy())

# Gaussian Process for Regression
class _MT_GPR(gpytorch.models.ExactGP):
    def __init__(self, X_, Y_, g_, _mvlike, kernel, degree, RC, hrzn, random_init = True, multiple_length_scales = False):
        super(_MT_GPR, self).__init__(X_, Y_, _mvlike)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks = Y_.shape[1])
        # Random Parameters Initialization
        self.random_init            = random_init
        self.multiple_length_scales = multiple_length_scales
        # Define features index
        idx_dim_  = torch.linspace(0, g_.shape[0] - 1, g_.shape[0], dtype = int)
        # Treat features and index independently
        idx_      = idx_dim_[g_ != torch.unique(g_)[-1]]
        idx_bias_ = idx_dim_[g_ == torch.unique(g_)[-1]]
        #self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.LinearKernel(), num_tasks = Y_.shape[1], rank = Y_.shape[1])
        self.covar_module = gpytorch.kernels.MultitaskKernel(self.__define_kernel(kernel, degree), num_tasks = Y_.shape[1], rank = Y_.shape[1])

        # Define features kernel
        _K = self.__define_kernel(kernel, degree, idx_dim_ = idx_)

        # Define constant kernel for bias
        _K_bias = self.__define_kernel(kernel   = 'linear',
                                       degree   = 0,
                                       idx_dim_ = idx_bias_)

        #self.covar_module = gpytorch.kernels.MultitaskKernel(_K + _K_bias, num_tasks = Y_.shape[1], rank = Y_.shape[1])

        # Multiple-kernel learning
        if (RC == 1) and (hrzn > 0):
            idx_rc_ = idx_dim_[g_ == torch.unique(g_)[-2]]
            # Define kernel for recursive predictions
            _K_chain = self.__define_kernel(kernel   = 'linear',
                                            degree   = 0,
                                            idx_dim_ = idx_rc_)
            # Combine features and bias kernels
            self.covar_module = gpytorch.kernels.MultitaskKernel(_K + _K_chain + _K_bias, num_tasks = Y_.shape[1], rank = Y_.shape[1])
        else:
            # Combine features and bias kernels
            self.covar_module = gpytorch.kernels.MultitaskKernel(_K + _K_bias, num_tasks = Y_.shape[1], rank = Y_.shape[1])

    # Define a kernel
    def __define_kernel(self, kernel, degree, idx_dim_ = None):
        if self.multiple_length_scales:
            dim = int(idx_dim_.shape[0])
        else:
            dim = None
        # Constraint name: likelihood.raw_task_noises_constraint                   constraint = GreaterThan(1.000E-04)
        # Constraint name: covar_module.task_covar_module.raw_var_constraint       constraint = Positive()
        # Random Initialization Covariance Matrix
        if self.random_init:
            self.likelihood.raw_task_noises.data.fill_(self.likelihood.raw_task_noises_constraint.inverse_transform(_random()))
        # Linear kernel
        if kernel == 'linear':
            _K = gpytorch.kernels.LinearKernel(active_dims = idx_dim_)
            # Linear kernel parameter
            if self.random_init: _K.raw_variance.data.fill_(_K.raw_variance_constraint.inverse_transform(_random()))
            return _K
        # Radian Basis Function Kernel
        if kernel == 'RBF':
            _K = gpytorch.kernels.RBFKernel(active_dims  = idx_dim_,
                                            ard_num_dims = dim)
            # RBF Kernel parameter
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]):
                    _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        # Polynomial Expansion Kernel
        if kernel == 'poly':
            _K = gpytorch.kernels.PolynomialKernel(power       = degree,
                                                   active_dims = idx_dim_)
            # Polynomial Kernel parameter
            if self.random_init:
                _K.raw_offset.data.fill_(_K.raw_offset_constraint.inverse_transform(_random()))
        # Matern Kernel
        if kernel == 'matern':
            _K = gpytorch.kernels.MaternKernel(nu           = degree,
                                               active_dims  = idx_dim_,
                                               ard_num_dims = dim)
            # Matern Kernel parameter
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        # Rational Quadratic Kernel
        if kernel == 'RQ':
            _K = gpytorch.kernels.RQKernel(active_dims  = idx_dim_,
                                           ard_num_dims = dim)
            # RQ Kernel parameters
            if self.random_init:
                _K.raw_alpha.data.fill_(_K.raw_alpha_constraint.inverse_transform(_random()))
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]):
                    _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        _K = gpytorch.kernels.ScaleKernel(_K)
        # Amplitude coefficient parameter
        if self.random_init:
            _K.raw_outputscale.data.fill_(_K.raw_outputscale_constraint.inverse_transform(_random()))
        return _K

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# Select the best model using multiple initializations
def _MTGPR_fit(X_, Y_, g_, param_):
    # Gaussian Process Regression model fit...
    def __fit(X_, Y_, g_, params_, random_init = True):
        # Optimize Kernel hyperparameters
        def __optimize(_model, _mvlike, X_, Y_, max_training_iter, early_stop):
            # Storage Variables Initialization
            nmll_ = []
            # Find optimal model hyperparameters
            _model.train()
            # Use the adam optimizer
            _optimizer = torch.optim.Adam(_model.parameters(), lr = .1)  # Includes GaussianLikelihood parameters
            # "Loss" for GPs - the marginal log likelihood
            _mll = gpytorch.mlls.ExactMarginalLogLikelihood(_mvlike, _model)
            # Begins Iterative Optimization
            for i in range(max_training_iter):
                # Zero gradients from previous iteration
                _optimizer.zero_grad()
                # Output from model
                F_hat_ = _model(X_)
                # Calc loss and backprop gradients
                _nmll = - _mll(F_hat_, Y_)
                _nmll.backward()
                _optimizer.step()
                #print(i, np.around(float(_error.detach().numpy()), 2))
                nmll_.append(np.around(float(_nmll.detach().numpy()), 2) )
                if np.isnan(nmll_[-1]):
                    return _model, _mvlike, np.inf
                if i > early_stop:
                    if np.unique(nmll_[-early_stop:]).shape[0] == 1:
                        break
            return _model, _mvlike, nmll_[-1]

        kernel, degree, RC, hrzn, max_training_iter, n_random_init, early_stop = params_
        # Add dummy feature for the bias
        X_ = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
        g_ = np.concatenate((g_, np.ones((1,))*np.unique(g_)[-1] + 1), axis = 0)
        # Numpy yo pyTorch
        X_p_ = torch.tensor(X_, dtype = torch.float)
        Y_p_ = torch.tensor(Y_, dtype = torch.float)
        g_p_ = torch.tensor(g_, dtype = torch.float)
        # initialize likelihood and model
        _mvlike = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = Y_p_.shape[1])
        _model  = _MT_GPR(X_p_, Y_p_, g_p_, _mvlike, kernel, degree, RC, hrzn, random_init)
        # for constraint_name, constraint in _model.named_constraints():
        #    print(f'Constraint name: {constraint_name:55} constraint = {constraint}')
        return __optimize(_model, _mvlike, X_p_, Y_p_, max_training_iter, early_stop)

    kernel, degree, RC, hrzn, max_training_iter, n_random_init, early_stop = param_
    # Storage Variables Initialization
    model_ = []
    nmll_  = []
    # No Random Initialization
    _MTGPR, _mvlike, nmll = __fit(X_, Y_, g_, param_, random_init = False)
    # Get Results
    model_.append([_MTGPR, _mvlike])
    nmll_.append(nmll)
    # Perform multiple Random Initializations
    for i in range(n_random_init):
        _MTGPR, _mvlike, nmll = __fit(X_, Y_, g_, param_, random_init = True)
        # Get Results
        model_.append([_MTGPR, _mvlike])
        nmll_.append(nmll)
    # Best Results of all different Initialization
    _MTGPR, _mvlike = model_[np.argmin(nmll_)]
    nmll            = nmll_[np.argmin(nmll_)]
    return [_MTGPR, _mvlike, nmll]

# Calculating prediction for new sample
def _MTGPR_predict(MTGP_, X_):
    _model, _mvlike, nmll = MTGP_
    X_   = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
    X_p_ = torch.tensor(X_, dtype = torch.float)
    _model.eval()
    _mvlike.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        _F_hat = _mvlike(_model(X_p_))
        #print(np.sqrt(_mvlike.noise.numpy()).shape, np.sqrt(_mvlike.task_noises.numpy()).shape)
        #print(np.diag(_mvlike.task_noises.numpy()).shape)
        #print(_F_hat.mean.numpy().shape, np.sqrt(_F_hat.variance.numpy()).shape, np.sqrt(_F_hat.covariance_matrix.numpy()).shape)
        return _F_hat.mean.numpy(), np.sqrt(_F_hat.variance.numpy()), np.sqrt(_mvlike.task_noises.numpy())

# Calculating prediction for new sample
def _cMTGPR_predict(cMTGPR_, X_ts_):
    _cMTGPR, X_tr_ = cMTGPR_
    Y_hat_ = _cMTGPR.predict(X_ts_, conf_intervals = False, compute_C_star = False)
    # Compute per-task covariances
    G_hat_   = _cMTGPR._PriorW
    S_noise_ = _cMTGPR._SigmaTT
    #S_hat_   = _cMTGPR.C_star_
    #K_ts_    = _cMTGPR.C_K_test_
    #print(y_hat_.shape, S_noise_.shape, S_hat_.shape)
    K_tr_    = _cMTGPR.compute_kernel(X_tr_, X_tr_)
    K_tr_ts_ = _cMTGPR.compute_kernel(X_tr_, X_ts_)
    K_ts_    = _cMTGPR.compute_kernel(X_ts_, X_ts_)
    # S_hat_   = np.kron(G_hat_, K_ts_) - np.kron(G_hat_, K_tr_ts_).T @ inv(np.kron(G_hat_, K_tr_) + np.kron(S_noise_, np.eye(X_tr_.shape[0]))) @ np.kron(G_hat_, K_tr_ts_) \
    #            + np.kron(S_noise_, np.eye(X_ts_.shape[0]))
    N_samples, N_tasks = Y_hat_.shape
    S_hat_   = np.kron(G_hat_, K_ts_) - np.kron(G_hat_, K_tr_ts_).T @ inv(np.kron(G_hat_, K_tr_) + np.kron(S_noise_, np.eye(X_tr_.shape[0]))) @ np.kron(G_hat_, K_tr_ts_)
    s_hat_   = np.diagonal(S_hat_)
    S_hat_   = np.stack([s_hat_[N_samples*i_task:N_samples*(i_task + 1)] for i_task in range(N_tasks)]).T
    S_noise_ = np.diagonal(_cMTGPR._SigmaTT)
    return Y_hat_, np.sqrt(S_hat_), np.sqrt(S_noise_)

# Select the best model using multiple initializations
def _cMTGPR_fit(X_, Y_, g_, param_):
    #DotProduct, RBF, WhiteKernel, RationalQuadratic, Matern, ConstantKernel
    #kernels_ = ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']
    #degrees_ = [0., 0., 2., 3., 1./2., 3./2., 5./2., 0.]
    if param_[0] == 'linear':
        kernel_ = DotProduct()
    if param_[0] == 'poly':
        kernel_ = DotProduct()**param_[1]
    if param_[0] == 'RBF':
        kernel_ = ConstantKernel()*RBF() + ConstantKernel()
    if param_[0] == 'matern':
        kernel_ = ConstantKernel()*Matern(nu = param_[1]) + ConstantKernel()
    if param_[0] == 'RQ':
        kernel_ = ConstantKernel()*RationalQuadratic() + ConstantKernel()
    _cMTGPR = h_mtgp(kernel = kernel_, kernel_noise = WhiteKernel(), n_restarts_optimizer = param_[-1])
    _cMTGPR.fit(X_, Y_, alpha_method = 'largeT')
    return [_cMTGPR, X_]


__all__ = ['_GPR_predict',
           '_GPR_fit',
           '_skGPR_fit',
           '_MTGPR_fit',
           '_MTGPR_predict',
           '_cMTGPR_fit',
           '_cMTGPR_predict']
