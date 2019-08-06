import numpy as np
import george
from robo.priors.default_priors import DefaultPrior
from robo.models.gaussian_process import GaussianProcess
from robo.models.base_model import BaseModel


class MetaGP(BaseModel):
    def __init__(self, meta_data, lower, upper, rng):
        self.lower = lower
        self.upper = upper
        self.rng = rng

        self.base_models = {}
        self.target_model = self.get_gp()
        self.weights = None

        self.mus = []
        self.vars =[]

        for dataset_name, (X, y) in meta_data.items():

            if len(y) > 0 and len(np.unique(y, axis=0)) > 3:

                gp = self.get_gp()
                y = (y - np.mean(y)) / np.std(y)
                gp.train(X.copy(), y.copy())
                self.base_models[dataset_name] = gp

        self.y_mean = np.inf
        self.y_std = np.inf
        self.X = None
        self.y = None
        self.y_normalized = None

    @BaseModel._check_shapes_train
    def train(self, X, y, do_optimize=True):
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        y_normalized = (y - self.y_mean) / self.y_std
        self.y_normalized = y_normalized
        self.X = X
        self.y = y
        self.target_model.train(X, y_normalized, do_optimize=do_optimize)
        self.compute_weights(X, y_normalized)

    @BaseModel._check_shapes_predict
    def predict(self, X, full_cov=False, **kwargs):
        means = []
        variances = []
        for w, (_, model) in zip(
                self.weights,
                self.base_models.items()
        ):
            if w == 0:
                means.append(np.zeros(X.shape[0]))
                variances.append(np.zeros(X.shape[0]))
            else:
                m, v = model.predict(X, full_cov, **kwargs)
                means.append(m.flatten())
                variances.append(v.flatten())
        m, v = self.target_model.predict(X, full_cov, **kwargs)
        means.append(m.flatten())
        means = np.array(means)
        variances.append(v.flatten())
        variances = np.array(variances)
        mu = np.average(means, weights=self.weights, axis=0) * self.y_std + self.y_mean
        var = np.sum([
            v * (w ** 2)
            for v, w in zip(variances, self.weights)
        ], axis=0)
        var = var * (self.y_std ** 2)

        self.vars.append(var)
        self.mus.append(mu)

        return mu, var

    def compute_weights(self, X, y):
        # Dummy implementation, just the average
        weights = np.ones(len(self.base_models) + 1)
        self.weights = weights / np.sum(weights)

    def get_gp(self):
        lower_ = self.lower
        upper_ = self.upper

        cov_amp = 2
        n_dims = self.lower.shape[0]

        initial_ls = np.ones([n_dims])
        exp_kernel = george.kernels.Matern52Kernel(
            initial_ls, ndim=n_dims,
        )
        kernel = cov_amp * exp_kernel

        prior = DefaultPrior(len(kernel) + 1)

        model = GaussianProcess(
            kernel,
            prior=prior,
            rng=self.rng,
            normalize_output=True,
            normalize_input=True,
            lower=lower_,
            upper=upper_,
            noise=1e-6,
        )
        return model