import numpy as np
import george
from robo.priors.default_priors import DefaultPrior
from robo.models.gaussian_process import GaussianProcess
from robo.initial_design.init_latin_hypercube_sampling import init_latin_hypercube_sampling
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from robo.acquisition_functions.log_ei import LogEI
from robo.solver.bayesian_optimization import BayesianOptimization
from scipy.optimize import differential_evolution
from lib.objective_function.Alpine1d import Alpine1d
from lib.objective_function.Alpine2d import Alpine2d
from lib.objective_function.Alpine1dShifted import Alpine1dShifted
from lib.objective_function.Function import Function
from lib.objective_function.Ginuta import Ginuta
from lib.objective_function.Griewank import Griewank
from lib.models.RLGP import RLGP
from lib.models.RGPE import RGPE
from lib.BO.BayesianOptimizationSurrogateModelEnsemble import BayesianOptimizationSurrogateModelEnsemble
import json

def benchmark_function(
        function,
        seed,
        n_eval=20,
        n_initial_points=5,
        model_class=None,
        model_kwargs=None,
):
    lower = np.array([-10])
    upper = np.array([10])
    rng1 = np.random.RandomState(seed)
    rng2 = np.random.RandomState(seed)

    cov_amp = 2
    n_dims = lower.shape[0]

    initial_ls = np.ones([n_dims])
    exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                               ndim=n_dims)
    kernel = cov_amp * exp_kernel

    prior = DefaultPrior(len(kernel) + 1)

    if model_class is None:
        model = GaussianProcess(
            kernel,
            prior=prior,
            rng=rng1,
            normalize_output=True,
            normalize_input=True,
            lower=lower,
            upper=upper,
            noise=1e-3,
        )
    else:
        model = model_class(rng=rng1, **model_kwargs)

    acq = LogEI(model)
    max_func = SciPyOptimizer(acq, lower, upper, n_restarts=50, rng=rng2)

    bo = BayesianOptimization(
        objective_func=function,
        lower=np.array([-10]),
        upper=np.array([10]),
        acquisition_func=acq,
        model=model,
        initial_points=n_initial_points,
        initial_design=init_latin_hypercube_sampling,
        rng=rng2,
        maximize_func=max_func
    )

    bo.run(n_eval)
    rval = np.minimum.accumulate(bo.y)

    return rval


def benchmark_function_model_selection(
        target,
        seed,
        n_eval=20,
        n_initial_points=5,
        model_class=None
):
    lower = np.array([-10])
    upper = np.array([10])
    rng1 = np.random.RandomState(seed)
    rng2 = np.random.RandomState(seed)

    # Build models for all algorithms
    models = []
    acqs = []
    max_funcs = []
    targets = []

    for obj_function_name, obj_function in objective_functions.items():

        meta_data = {}
        base = {}
        for model_index in range(0, len_meta_data + 1):
            if obj_function != model_index:
                base[model_index] = obj_function(model_index)

        for i, (key, obj_function_) in enumerate(base.items()):
            rs = np.random.RandomState(i)
            X = rs.rand(20, 1) * 20 - 10
            y = obj_function_(X)
            meta_data[i] = (X, y)
        model_kwargs = {
            'lower': np.array([-10]),
            'upper': np.array([10]),
            'meta_data': meta_data,
        }

        target = obj_function(target_index)
        targets.append(target)

        model = model_class(rng=rng1, **model_kwargs)
        models.append(model)
        acq = LogEI(model)
        acqs.append(acq)
        max_func = SciPyOptimizer(acq, lower, upper, n_restarts=50, rng=rng2)
        max_funcs.append(max_func)

    print("Benchmark for model selection...")
    bo = BayesianOptimizationSurrogateModelEnsemble(
        objective_funcs=targets,
        lower=np.array([-10]),
        upper=np.array([10]),
        acquisition_funcs=acqs,
        models=models,
        initial_points=n_initial_points,
        initial_design=init_latin_hypercube_sampling,
        rng=rng2,
        maximize_funcs=max_funcs
    )

    bo.run(n_eval)
    rval = np.minimum.accumulate(bo.ys[bo.target_index])

    return rval


def get_trajectory_for_model(model_class, model_kwargs, n_runs, n_initial_points, n_eval, target):
    if model_class is not RLGP:
        results = []
        for seed in range(1, n_runs + 1):
            results.append(
                benchmark_function(
                    target,
                    seed,
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    n_initial_points=n_initial_points,
                    n_eval=n_eval,
                )
            )
        results = np.array(results)
        trajectory = np.mean(results, axis=0)
        stderr = np.std(results, axis=0) / np.sqrt(results.shape[1])

    else:
        results = []
        for seed in range(1, n_runs + 1):
            result= benchmark_function_model_selection(
                    None,
                    seed,
                    model_class=model_class,
                    n_initial_points=n_initial_points,
                    n_eval=n_eval,
                )
            results.append(result)
        results = np.array(results)
        trajectory = np.mean(results, axis=0)
        stderr = np.std(results, axis=0) / np.sqrt(results.shape[1])

    return trajectory.tolist(), stderr.tolist(), minimum


def benchmark(objective_function, objective_function_name):
    trajectories = {}
    meta_data = {}
    base = {}
    target = objective_function(target_index)

    for model_index in range(0, len_meta_data + 1):
        if target_index != model_index:
            base[model_index] = objective_function(model_index)

    for i, (key, obj_function) in enumerate(base.items()):
        rs = np.random.RandomState(i)
        X = rs.rand(20, 1) * 20 - 10
        y = obj_function(X)
        meta_data[i] = (X, y)

    for i, (method_name, method) in enumerate(methods.items()):
        if method is not None:
            if method is not RGPE:
                print("Call model selection...")
                trajectories[method_name] = get_trajectory_for_model(
                    model_class=method,
                    model_kwargs=None,
                    n_runs=n_runs,
                    n_initial_points=3,
                    n_eval=n_eval,
                    target=None
                )

            else:
                # RGPE
                trajectory_warmstarting_warm_staring = get_trajectory_for_model(
                    model_class=method,
                    model_kwargs={
                        'lower': np.array([-10]),
                        'upper': np.array([10]),
                        'meta_data': meta_data,
                    }, n_runs=n_runs,
                    n_initial_points=3,
                    n_eval=n_eval,
                    target=target
                )
                trajectories[method_name] = trajectory_warmstarting_warm_staring
        else:
            # GP
            trajectory_gp = get_trajectory_for_model(
                model_class=None, model_kwargs=None, n_runs=n_runs,
                n_initial_points=3, n_eval=n_eval, target=target
            )
            trajectories['GP'] = trajectory_gp

    trajectory_mean[objective_function_name] = trajectories

    # Write results in json
    with open('model_selection.json', 'w') as json_file:
        json.dump(trajectory_mean, json_file)


n_runs = 20
n_eval = 20
target_index = 0
len_meta_data = 5
methods = {
    'RGPE': RGPE,
    'GP': None,
    'RLGP': RLGP
}

objective_functions = {
    'BoTorch': Function,
    'Alpine1d': Alpine1d,
    'Alpine2d': Alpine2d,
    'Ginuta': Ginuta,
    'Griewank': Griewank,
    'Alpine1d(Shifted)': Alpine1dShifted,
}

# Find global minimum
minimums = []
for obj_function_name, obj_function in objective_functions.items():
    target_ = obj_function(0)
    minimums.append(differential_evolution(target_, [(-10, 10)], maxiter=5000).fun)

minimum = np.min(minimums)

trajectory_mean = {}

for obj_function_name, obj_function in objective_functions.items():
    print("Optimize model for {}".format(obj_function_name))
    benchmark(obj_function, obj_function_name)
