import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from lib.models.RLGP import RLGP
from lib.models.RGPE import RGPE
import json

n_runs = 20
n_eval = 20
mean_trajectories = {}

methods = {
    'RGPE': RGPE,
    'GP': None,
    'RLGP': RLGP
}

objective_function_algorithms = {
    'BoTorch',
    'Alpine1d',
    'Alpine2d',
    'Ginuta',
    'Griewank',
    'Alpine1d(Shifted)'
}

with open('model_selection.json', 'r') as fp:
    mean_trajectories = json.load(fp)


def plot(mean_trajectories, n_eval):

    plt.rcParams["figure.figsize"] = (23, 8)
    fig = plt.figure()
    timeline = np.arange(1, n_eval + 0.01)

    plt.subplot(1, 1, 1)

    for method_name, method in methods.items():

        t_mean = []
        s_mean = []
        m_ = []

        for objective_function_algorithm in objective_function_algorithms:

            for objective_function_algorithm_, trajectories_ in mean_trajectories.items():

                if objective_function_algorithm_ == objective_function_algorithm:
                    for method_n, (t, s, m) in trajectories_.items():
                        if method_name == method_n:
                            t_mean.append(t)
                            s_mean.append(s)
                            m_.append(m)

        t_means = np.mean(t_mean, axis=0)
        s_means = np.mean(s_mean, axis=0)
        ms_ = np.mean(m_, axis=0)

        plt.errorbar(timeline, t_means - ms_, yerr=s_means, label=method_name)

    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Average regret', fontsize=20)
    plt.legend(loc="upper right", prop={'size': 20}, numpoints=1)
    plt.xticks(np.arange(0, 20.1, step=5), fontsize=16)
    plt.yticks(fontsize=16)

    now = datetime.now()  # current date and time
    time_ = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = 'model_selecion_{}_GP_RGPE_RLGP_mean.png'.format(time_)
    fig.savefig(file_name, bbox_inches='tight', pad_inches=0)


plot(mean_trajectories, n_eval)