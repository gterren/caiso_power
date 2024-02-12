import gpytorch, torch, copy

import numpy as np
import pandas as pd

import scipy as sp

from scipy.linalg import inv

# Import kernels for the Cool-MTGP model.
from lib.kernels import DotProduct, RBF, WhiteKernel, RationalQuadratic, Matern, ConstantKernel

# Import approximate Cool-MTGP (~Cool-MTGP)
from lib.A_Cool_MTGP import MultitaskGP as a_mtgp

# Import hierarchical Cool-MTGP (HCool-MTGP)
from lib.H_Cool_MTGP import MultitaskGP as h_mtgp

from lib.coolmt_gptorch import Cool_MTGP

# import sklearn single GP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.gaussian_process.kernels import RationalQuadratic, ExpSineSquared, DotProduct


path_to_params = r'/home/gterren/caiso_power/GP_param/'

def _get_param_names(_model):
    return [param_name for param_name, param in _model.named_parameters()]

def _get_param_values(_model):
    return [param.item() for param_name, param in _model.named_parameters()]

# Fit Gaussian process using SkLearn
def _skGPR_fit(X_, y_, g_, param_):

    kernel, degree, RC, hrzn, max_training_iter, n_random_init, early_stop = param_
    # Linear kernel
    if kernel == 'linear':
       _kernel = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-10, 1e10)) * DotProduct(sigma_0        = 1.,
                                                                                                       sigma_0_bounds = (1e-10, 1e10)) + WhiteKernel(noise_level = 1., noise_level_bounds = (1e-10, 1e10))
    # Order 2 Polynomial kernel
    if kernel == 'poly':
        _kernel = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-10, 1e10)) * DotProduct(sigma_0        = 1.,
                                                                                                        sigma_0_bounds = (1e-10, 1e10))**degree
    # Radial basis funtions kernel
    if kernel == 'RBF':
        _kernel = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-10, 1e10)) * RBF(length_scale        = 1.,
                                                                                                 length_scale_bounds = (1e-10, 1e10)) + WhiteKernel(noise_level = 1., noise_level_bounds = (1e-10, 1e10)) + ConstantKernel(constant_value = 1., constant_value_bounds = (1e-10, 1e10))
    if kernel == 'RQ':
        # Rational Quadratic kernel
        _kernel = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-10, 1e10)) * RationalQuadratic(length_scale        = 1.,
                                                                                                                 alpha               = 0.1,
                                                                                                                 length_scale_bounds = (1e-10, 1e10),
                                                                                                                 alpha_bounds        = (1e-10, 1e10)) + WhiteKernel(noise_level = 1., noise_level_bounds = (1e-10, 1e10)) + ConstantKernel(constant_value = 1., constant_value_bounds = (1e-10, 1e10))
    if kernel == 'matern':
        # Matern Kernel with nu hyperparameter set to 0.5
        _kernel = ConstantKernel(constant_value = 1., constant_value_bounds = (1e-10, 1e10)) * Matern(length_scale        = 1.0,
                                                                                                      length_scale_bounds = (1e-10, 1e10),
                                                                                                      nu                  = degree) + WhiteKernel(noise_level = 1., noise_level_bounds = (1e-10, 1e10)) + ConstantKernel(constant_value = 1., constant_value_bounds = (1e-10, 1e10))
    # Training Gaussian process for regression
    return GaussianProcessRegressor(kernel               = _kernel,
                                    n_restarts_optimizer = param_[-2]).fit(X_, y_)

# Gaussian Process for Regression
class _GPR(gpytorch.models.ExactGP):
    def __init__(self, X_, y_, g_, _like, kernel, degree, hrzn, multiple_length_scales = False):
        super(_GPR, self).__init__(X_, y_, _like)
        self.mean_module = gpytorch.means.ConstantMean()
        # Random Parameters Initialization
        self.multiple_length_scales = multiple_length_scales
        # Define features index
        idx_dim_ = torch.linspace(0, g_.shape[0] - 1, g_.shape[0], dtype = int)
        # Treat features and index independently
        # idx_others_  = idx_dim_[(g_ != 0) & (g_ != 1) & (g_ != 2)]
        # idx_ft_      = idx_dim_[(g_ == 0) | (g_ == 1) | (g_ == 2)]
        idx_dim_all_ = [idx_dim_]
        # Define features kernel
        self.covar_module = self.__define_kernel(kernel, degree, idx_dim_ = idx_dim_all_)
        # Define constant kernel for bias
        # _K_bias = self.__define_kernel(kernel   = 'linear',
        #                                degree   = 0,
        #                                idx_dim_ = idx_bias_)
        # # self.covar_module = _K + _K_bias
        # print(hrzn)
        # # Multiple-kernel learning
        # if hrzn > 0:
        #     idx_rc_ = idx_dim_[g_ == torch.unique(g_)[-2]]
        #
        #     # Define kernel for recursive predictions
        #     _K_chain = self.__define_kernel(kernel   = 'linear',
        #                                     degree   = 0,
        #                                     idx_dim_ = idx_rc_)
        #     # Combine features and bias kernels
        #     self.covar_module = _K + _K_chain + _K_bias
        # else:
        #     # Combine features and bias kernels
        #     self.covar_module = _K + _K_bias

    # Define a kernel
    def __define_kernel(self, kernel, degree, idx_dim_ = None):
        # if self.multiple_length_scales:
        #     dim = int(idx_dim_.shape[0])
        # else:
        dim = None
        #_prior = gpytorch.priors.SmoothedBoxPrior(1e-5, 2.5)
        #_prior = gpytorch.priors.GammaPrior(.4, 4.)
        _prior = gpytorch.priors.GammaPrior(1., 10.)
        #_prior = gpytorch.priors.HalfCauchyPrior(1.)
        #_prior = gpytorch.priors.NormalPrior(0, 1)
        # Random Initialization noise variance
        self.likelihood.noise = _prior.sample()
        # Random Initialization Constant Mean
        self.mean_module.constant = _prior.sample()
        # Linear kernel
        if kernel == 'linear':
            _K = gpytorch.kernels.LinearKernel(active_dims = idx_dim_[0])
            # Linear kernel parameter initialization
            _K.variance = _prior.sample()
        # Radian Basis Function Kernel
        if kernel == 'RBF':
            _K_1 = gpytorch.kernels.RBFKernel(active_dims  = idx_dim_[0],
                                              ard_num_dims = dim)
            # RBF Kernel parameter initialization
            _K_1.lengthscale = _prior.sample()
            _K               = gpytorch.kernels.ScaleKernel(_K_1)
            _K.outputscale   = _prior.sample()
        # Polynomial Expansion Kernel
        if kernel == 'poly':
            _K_1 = gpytorch.kernels.PolynomialKernel(power       = degree,
                                                     active_dims = idx_dim_[0])
            # Polynomial Kernel parameter initialization
            _K_1.offset    = _prior.sample()
            _K             = gpytorch.kernels.ScaleKernel(_K_1)
            _K.outputscale = _prior.sample()
        # Matern Kernel
        if kernel == 'matern':
            _K_1 = gpytorch.kernels.MaternKernel(nu           = degree,
                                                 active_dims  = idx_dim_[0],
                                                 ard_num_dims = dim)
            # Matern Kernel parameter initialization
            _K_1.lengthscale = _prior.sample()
            _K               = gpytorch.kernels.ScaleKernel(_K_1)
            _K.outputscale   = _prior.sample()
        # Rational Quadratic Kernel
        if kernel == 'RQ':
            _K_1 = gpytorch.kernels.RQKernel(active_dims  = idx_dim_[0],
                                           ard_num_dims = dim)
            # RQ Kernel parameters initialization
            _K_1.lengthscale = _prior.sample()
            _K_1.alpha       = _prior.sample()
            _K               = gpytorch.kernels.ScaleKernel(_K_1)
            _K.outputscale   = _prior.sample()
        # Piecewise Polynomial Kernel
        if kernel == 'PW':
            _K = gpytorch.kernels.PiecewisePolynomialKernel(q            = degree,
                                                            active_dims  = idx_dim_[0],
                                                            ard_num_dims = dim)
            # PW Kernel parameters initialization
            _K.lengthscale = _prior.sample()
        # Stationary and non-stationary Kernel
        if kernel == 'linear_exp_rbf':
            _K_0 = gpytorch.kernels.LinearKernel(active_dims = idx_dim_[1])
            _K_2 = gpytorch.kernels.RBFKernel(active_dims    = idx_dim_[2],
                                              ard_num_dims   = dim)
            _K_1 = gpytorch.kernels.ScaleKernel(_K_0)
            _K_3 = gpytorch.kernels.ScaleKernel(_K_2)

            # Kernel parameter initialization
            _K_0.variance    = _prior.sample()
            _K_1.outputscale = _prior.sample()
            _K_2.lengthscale = _prior.sample()
            _K_3.outputscale = _prior.sample()
            # Define multiple kernel
            _K = _K_1 + _K_2
        if kernel == 'linear_exp_matern':
            _K_0 = gpytorch.kernels.LinearKernel(active_dims  = idx_dim_[1])
            _K_2 = gpytorch.kernels.MaternKernel(nu           = degree,
                                                 active_dims  = idx_dim_[2],
                                                 ard_num_dims = dim)
            _K_1 = gpytorch.kernels.ScaleKernel(_K_0)
            _K_3 = gpytorch.kernels.ScaleKernel(_K_2)
            # Kernel parameter initialization
            _K_0.variance    = _prior.sample()
            _K_1.outputscale = _prior.sample()
            _K_2.lengthscale = _prior.sample()
            _K_3.outputscale = _prior.sample()
            # Define multiple kernel
            _K = _K_1 + _K_2
        if kernel == 'linear_exp_rq':
            _K_0 = gpytorch.kernels.LinearKernel(active_dims = idx_dim_[1])
            _K_2 = gpytorch.kernels.RQKernel(active_dims  = idx_dim_[2],
                                             ard_num_dims = dim)
            _K_1 = gpytorch.kernels.ScaleKernel(_K_0)
            _K_3 = gpytorch.kernels.ScaleKernel(_K_2)
            # Kernel parameter initialization
            _K_0.variance    = _prior.sample()
            _K_1.outputscale = _prior.sample()
            _K_2.lengthscale = _prior.sample()
            _K_2.alpha       = _prior.sample()
            _K_3.outputscale = _prior.sample()
            # Define multiple kernel
            _K = _K_1 + _K_2
        if kernel == 'linear_period':

            _K_0 = gpytorch.kernels.LinearKernel(active_dims = idx_dim_[1])
            _K_2 = gpytorch.kernels.PeriodicKernel(active_dims  = idx_dim_[2],
                                                   ard_num_dims = dim)

            _K_1 = gpytorch.kernels.ScaleKernel(_K_0)
            _K_3 = gpytorch.kernels.ScaleKernel(_K_3)
            _K_0.variance      = _prior.sample()
            _K_2.lengthscale   = _prior.sample()
            _K_2.period_length =  _prior.sample()
            _K_1.outputscale   = _prior.sample()
            _K_3.outputscale   = _prior.sample()


        return _K

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Select the best model using multiple initializations
def _GPR_fit(X_, y_, g_, param_):
    # Gaussian Process Regression model fit...
    def __fit(X_, y_, g_, params_):
        # Optimize Kernel hyperparameters
        def __optimize(_model, _like, X_, y_, max_iter, early_stop):
            # Storage Variables Initialization
            nmll_ = []
            # Find optimal model hyperparameters
            _model.train()
            # Use the adam optimizer
            _optimizer = torch.optim.Adam(_model.parameters(), lr = .1)  # Includes GaussianLikelihood parameters
            # "Loss" for GPs - the marginal log likelihood
            _mll             = gpytorch.mlls.ExactMarginalLogLikelihood(_like, _model)
            best_nmll        = np.inf
            early_stop_count = 0
            try:
                # Begins Iterative Optimization
                for i in range(max_iter):
                    # Zero gradients from previous iteration
                    _optimizer.zero_grad()
                    # Output from model
                    f_hat_ = _model(X_)
                    # Calc loss and backprop gradients
                    _nmll = - _mll(f_hat_, y_)
                    _nmll.backward()
                    _optimizer.step()
                    new_nmll = np.around(_nmll.detach().numpy(), 2)
                    # Abort if nan is found
                    if np.isnan(new_nmll):
                        return _model, _like, best_nmll
                    # Save last minima nmll and models
                    if best_nmll > new_nmll:
                        best_nmll   = new_nmll
                        _best_model = copy.deepcopy(_model)
                        _best_like  = copy.deepcopy(_like)
                        early_stop_count = 0
                    else:
                        # Keep track of the early counting
                        early_stop_count += 1
                        # Enforce early stopping
                        if early_stop_count == early_stop:
                            break
                return _best_model, _best_like, best_nmll
            except:
                return _model, _like, best_nmll

        kernel, degree, hrzn, max_iter, n_init, early_stop, key = params_
        # Add dummy feature for the bias
        X_ = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
        g_ = np.concatenate((g_, np.ones((1,))*np.unique(g_)[-1] + 1), axis = 0)
        # Numpy yo pyTorch
        X_p_ = torch.tensor(X_, dtype = torch.float)
        y_p_ = torch.tensor(y_, dtype = torch.float)
        g_p_ = torch.tensor(g_, dtype = torch.float)
        # initialize likelihood and model
        _like  = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = gpytorch.constraints.GreaterThan(1e-10))
        _model = _GPR(X_p_, y_p_, g_p_, _like, kernel, degree, hrzn)

        # init_params_ = _get_param_values(_model)

        _model, _like, end_nmll = __optimize(_model, _like, X_p_, y_p_, max_iter, early_stop)

        # name_params_  = _get_param_names(_model)
        # name_params_ += ['NMLL']
        # name_params_  = [name_param + '_init' for name_param in name_params_] + name_params_
        # end_params_   = _get_param_values(_model)
        # init_params_ += [init_nmll]
        # end_params_  += [end_nmll]
        # h_            = np.array(init_params_ + end_params_)

        return _model, _like, end_nmll

    kernel, degree, hrzn, max_iter, n_init, early_stop, key = param_
    print(kernel, degree, hrzn, max_iter, n_init, early_stop, key)
    # Storage Variables Initialization
    model_ = []
    nmll_  = []
    #H_     = []
    # Perform multiple Random Initializations
    for i in range(n_init):
        _model, _like, nmll = __fit(X_, y_, g_, param_)
        # Get Results
        model_.append([_model, _like])
        nmll_.append(nmll)
        #H_.append(h_)

    #pd.DataFrame(np.stack(H_).T, index = name_params_).to_csv(path_to_params + key + '_{}'.format(hrzn) + '.csv')
    # Best Results of all different Initialization
    _model, _like = model_[np.argmin(nmll_)]
    nmll          = nmll_[np.argmin(nmll_)]
    return [_model, _like, nmll]

# Calculating prediction for new sample
def _GPR_predict(GP_, X_, return_var = False):
    _model, _like, nmll = GP_
    X_   = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
    X_p_ = torch.tensor(X_, dtype = torch.float)
    _model.eval()
    _like.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        _f_hat = _like(_model(X_p_))
        if return_var: return _f_hat.mean.numpy(), _f_hat.variance.numpy() + _like.noise.numpy()
        else:          return _f_hat.mean.numpy()

# Gaussian Process for Regression
class _MT_GPR(gpytorch.models.ExactGP):
    def __init__(self, X_, Y_, g_, _mvlike, kernel, degree, multiple_length_scales = False):
        super(_MT_GPR, self).__init__(X_, Y_, _mvlike)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks = Y_.shape[1])
        # Random Parameters Initialization
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

        self.covar_module = gpytorch.kernels.MultitaskKernel(_K + _K_bias, num_tasks = Y_.shape[1], rank = Y_.shape[1])

        # # Multiple-kernel learning
        # if (RC == 1) and (hrzn > 0):
        #     idx_rc_ = idx_dim_[g_ == torch.unique(g_)[-2]]
        #     # Define kernel for recursive predictions
        #     _K_chain = self.__define_kernel(kernel   = 'linear',
        #                                     degree   = 0,
        #                                     idx_dim_ = idx_rc_)
        #     # Combine features and bias kernels
        #     self.covar_module = gpytorch.kernels.MultitaskKernel(_K + _K_chain + _K_bias, num_tasks = Y_.shape[1], rank = Y_.shape[1])
        # else:
        #     # Combine features and bias kernels
        #     self.covar_module = gpytorch.kernels.MultitaskKernel(_K + _K_bias, num_tasks = Y_.shape[1], rank = Y_.shape[1])

    # Define a kernel
    def __define_kernel(self, kernel, degree, idx_dim_ = None):
        if self.multiple_length_scales:
            dim = int(idx_dim_.shape[0])
        else:
            dim = None
        # Linear kernel
        if kernel == 'linear':
            _K = gpytorch.kernels.LinearKernel(variance_prior = gpytorch.priors.GammaPrior(1., 10.),
                                               active_dims    = idx_dim_)
            # Linear kernel parameter initialization
            _K.variance = gpytorch.priors.GammaPrior(1., 10.).sample()
            return _K
        # Radian Basis Function Kernel
        if kernel == 'RBF':
            _K = gpytorch.kernels.RBFKernel(lengthscale_prior = gpytorch.priors.GammaPrior(1., 10.),
                                            active_dims       = idx_dim_,
                                            ard_num_dims      = dim)
            # RBF Kernel parameter initialization
            for i in range(_K.lengthscale.shape[1]): _K.lengthscale[0, i] = gpytorch.priors.GammaPrior(1., 10.).sample()
        # Polynomial Expansion Kernel
        if kernel == 'poly':
            _K = gpytorch.kernels.PolynomialKernel(power        = degree,
                                                   offset_prior = gpytorch.priors.GammaPrior(1., 10.),
                                                   active_dims  = idx_dim_)
            # Polynomial Kernel parameter initialization
            _K.offset = gpytorch.priors.GammaPrior(1., 10.).sample()
        # Matern Kernel
        if kernel == 'matern':
            _K = gpytorch.kernels.MaternKernel(nu                = degree,
                                               lengthscale_prior = gpytorch.priors.GammaPrior(1., 10.),
                                               active_dims       = idx_dim_,
                                               ard_num_dims      = dim)
            # Matern Kernel parameter initialization
            for i in range(_K.lengthscale.shape[1]): _K.lengthscale[0, i] = gpytorch.priors.GammaPrior(1., 10.).sample()
        # Rational Quadratic Kernel
        if kernel == 'RQ':
            _K = gpytorch.kernels.RQKernel(lengthscale_prior = gpytorch.priors.GammaPrior(1., 10.),
                                           active_dims       = idx_dim_,
                                           ard_num_dims      = dim)
            # RQ Kernel parameters initialization
            for i in range(_K.lengthscale.shape[1]): _K.lengthscale[0, i] = gpytorch.priors.GammaPrior(1., 10.).sample()
        # Piecewise Polynomial Kernel
        if kernel == 'PW':
            _K = gpytorch.kernels.PiecewisePolynomialKernel(q                 = degree,
                                                            lengthscale_prior = gpytorch.priors.GammaPrior(1., 10.),
                                                            active_dims       = idx_dim_,
                                                            ard_num_dims      = dim)
            # PW Kernel parameters initialization
            for i in range(_K.raw_lengthscale.shape[1]): _K.lengthscale[0, i] = gpytorch.priors.GammaPrior(1., 10.).sample()
      # Stationary and non-stationary Kernel
        if kernel == 'S-NS':
            _K_1 = gpytorch.kernels.LinearKernel(variance_prior = gpytorch.priors.GammaPrior(1., 10.),
                                               active_dims    = idx_dim_)
            # Linear kernel parameter initialization
            _K_1.variance = gpytorch.priors.GammaPrior(1., 10.).sample()

            _K_2 = gpytorch.kernels.RBFKernel(lengthscale_prior = gpytorch.priors.GammaPrior(1., 10.),
                                               active_dims       = idx_dim_,
                                               ard_num_dims      = dim)
            # RBF Kernel parameter initialization
            for i in range(_K_2.lengthscale.shape[1]): _K_2.lengthscale[0, i] = gpytorch.priors.GammaPrior(1., 10.).sample()
            return _K_1 * _K_2
        # Amplitude coefficient
        _K = gpytorch.kernels.ScaleKernel(_K, outputscale_prior = gpytorch.priors.GammaPrior(1., 10.))
        # Amplitude coefficient parameter initialization
        _K.outputscale = gpytorch.priors.GammaPrior(1., 10.).sample()
        return _K

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# Select the best model using multiple initializations
def _MTGPR_fit(X_, Y_, g_, param_):
    # Gaussian Process Regression model fit...
    def __fit(X_, Y_, g_, params_):
        # Optimize Kernel hyperparameters
        def __optimize(_model, _mvlike, X_, Y_, max_iter, early_stop):
            # Storage Variables Initialization
            nmll_ = []
            # Find optimal model hyperparameters
            _model.train()
            # Use the adam optimizer
            _optimizer = torch.optim.Adam(_model.parameters(), lr = 0.1)  # Includes GaussianLikelihood parameters
            # "Loss" for GPs - the marginal log likelihood
            _mll = gpytorch.mlls.ExactMarginalLogLikelihood(_mvlike, _model)
            # Begins Iterative Optimization
            for i in range(max_iter):
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

        kernel, degree, max_iter, n_init, early_stop = params_
        # Add dummy feature for the bias
        X_ = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
        g_ = np.concatenate((g_, np.ones((1,))*np.unique(g_)[-1] + 1), axis = 0)
        # Numpy yo pyTorch
        X_p_ = torch.tensor(X_, dtype = torch.float)
        Y_p_ = torch.tensor(Y_, dtype = torch.float)
        g_p_ = torch.tensor(g_, dtype = torch.float)
        # initialize likelihood and model
        _mvlike = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = Y_p_.shape[1])
        _model  = _MT_GPR(X_p_, Y_p_, g_p_, _mvlike, kernel, degree)
        # for constraint_name, constraint in _model.named_constraints():
        #    print(f'Constraint name: {constraint_name:55} constraint = {constraint}')
        return __optimize(_model, _mvlike, X_p_, Y_p_, max_iter, early_stop)

    kernel, degree, max_iter, n_init, early_stop = param_
    # Storage Variables Initialization
    model_ = []
    nmll_  = []
    # Perform multiple Random Initializations
    for i in range(n_init):
        _MTGPR, _mvlike, nmll = __fit(X_, Y_, g_, param_)
        # Get Results
        model_.append([_MTGPR, _mvlike])
        nmll_.append(nmll)
    # Best Results of all different Initialization
    _MTGPR, _mvlike = model_[np.argmin(nmll_)]
    nmll            = nmll_[np.argmin(nmll_)]
    return [_MTGPR, _mvlike, nmll]

# Calculating prediction for new sample
def _MTGPR_predict(MTGP_, X_, return_var = False):
    _model, _mvlike, nmll = MTGP_
    X_   = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
    X_p_ = torch.tensor(X_, dtype = torch.float)
    _model.eval()
    _mvlike.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        _F_hat = _mvlike(_model(X_p_))
        N_samples, N_tsks = _F_hat.mean.numpy().shape
        S2_hat_   = _F_hat.covariance_matrix.numpy()
        S2_hat_p_ = np.stack([S2_hat_[i*N_tsks:(i + 1)*N_tsks, i*N_tsks:(i + 1)*N_tsks] for i in range(N_samples)])
        #S2_hat_p_ = np.stack([np.diag(_F_hat.variance.numpy()[i, :]) for i in range(N_samples)])
        if return_var: return _F_hat.mean.numpy(), S2_hat_p_ + np.diag(_mvlike.task_noises.numpy())
        else:          return _F_hat.mean.numpy()

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
    return Y_hat_, S_hat_ + S_noise_

# Select the best model using multiple initializations
def _cMTGPR_fit(X_, Y_, g_, params_):
    kernel, degree, RC, hrzn, max_iter, n_init, early_stop = params_
    #DotProduct, RBF, WhiteKernel, RationalQuadratic, Matern, ConstantKernel
    #kernels_ = ['linear', 'RBF', 'poly', 'poly', 'matern', 'matern', 'matern', 'RQ']
    #degrees_ = [0., 0., 2., 3., 1./2., 3./2., 5./2., 0.]
    if kernel == 'linear':
        kernel_ = DotProduct()
    if kernel == 'poly':
        kernel_ = DotProduct()**degree
    if kernel == 'RBF':
        kernel_ = ConstantKernel()*RBF() + ConstantKernel()
    if kernel == 'matern':
        kernel_ = ConstantKernel()*Matern(nu = degree) + ConstantKernel()
    if kernel == 'RQ':
        kernel_ = ConstantKernel()*RationalQuadratic() + ConstantKernel()
    _cMTGPR = h_mtgp(kernel = kernel_, kernel_noise = WhiteKernel(), n_restarts_optimizer = n_init)
    _cMTGPR.fit(X_, Y_, alpha_method = 'largeT')
    return [_cMTGPR, X_]


# Fit Gaussian process using SkLearn
def _tcMTGPR_fit(X_, Y_, g_, params_):
    kernel, degree, max_iter, n_init, early_stop = params_
   # Add dummy feature for the bias
    X_ = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
    g_ = np.concatenate((g_, np.ones((1,))*np.unique(g_)[-1] + 1), axis = 0)
    # Numpy yo pyTorch
    X_p_ = torch.tensor(X_, dtype = torch.float)
    Y_p_ = torch.tensor(Y_, dtype = torch.float)
    g_p_ = torch.tensor(g_, dtype = torch.float)
    _Cool_MTGPR = Cool_MTGP(kernel = kernel,
                            degree = degree)
    _Cool_MTGPR.train(X_p_, Y_p_, g_p_, training_iter = max_iter,
                                        n_init        = n_init,
                                        early_stop    = early_stop,
                                        verbose       = False)
    return _Cool_MTGPR

# Calculating prediction for new sample
def _tcMTGPR_predict(_tcMTGPR, X_, return_cov = False):
    X_   = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
    X_p_ = torch.tensor(X_, dtype = torch.float)
    # Make prediction
    Y_hat_, covariance_data_ = _tcMTGPR.predict(X_p_, conf_intervals = False,
                                                      compute_C_star = True)
    if return_cov:
        S2_hat_           = covariance_data_['C_star']
        N_samples, N_tsks = Y_hat_.shape
        return Y_hat_, np.stack([S2_hat_[i*N_tsks:(i + 1)*N_tsks, i*N_tsks:(i + 1)*N_tsks] for i in range(N_samples)])
    else:
        return Y_hat_

__all__ = ['_GPR_predict',
           '_GPR_fit',
           '_skGPR_fit',
           '_MTGPR_fit',
           '_MTGPR_predict',
           '_cMTGPR_fit',
           '_cMTGPR_predict',
           '_tcMTGPR_fit',
           '_tcMTGPR_predict']
