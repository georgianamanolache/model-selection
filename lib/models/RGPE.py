import numpy as np
from lib.models.MetaGP import MetaGP


class RGPE(MetaGP):
    def compute_weights(self, X, y):
        N_SAMPLES = 500

        true_rankings = (y[:, None] < y).flatten()
        # true_rankings = tr2

        # Sample rankings of base models
        sampled_rankings = []
        for _, base_model in self.base_models.items():
            sampled_rankings_m = []
            samples = base_model.sample_functions(X, n_funcs=N_SAMPLES)
            for sample in samples:
                rankings = (sample[:, None] < sample).flatten()
                sampled_rankings_m.append(rankings)
            sampled_rankings.append(sampled_rankings_m)

        # Sample rankings of target model
        loo_samples = []
        for i in range(X.shape[0]):
            X_tmp = list(X)
            x_loo = X_tmp[i]
            del X_tmp[i]
            X_tmp = np.array(X_tmp)
            y_tmp = list(y)
            del y_tmp[i]
            y_tmp = np.array(y_tmp)
            self.target_model.train(X_tmp, y_tmp, do_optimize=False)
            m = self.target_model.sample_functions(np.array([x_loo]), n_funcs=N_SAMPLES).flatten()
            loo_samples.append(m)
        loo_samples = np.array(loo_samples).transpose()
        sampled_rankings_m = []
        for sample in loo_samples:
            rankings = (sample[:, None] < y).flatten()
            sampled_rankings_m.append(rankings)
        sampled_rankings_m = np.array(sampled_rankings_m)
        sampled_rankings.append(sampled_rankings_m)

        sampled_rankings = np.array(sampled_rankings)

        # Compute scores
        weights = np.zeros(len(self.base_models) + 1)
        scores = np.zeros((N_SAMPLES, len(self.base_models) + 1))
        for i in range(len(self.base_models) + 1):
            for sample_idx in range(N_SAMPLES):
                scores[sample_idx][i] = np.sum(true_rankings != sampled_rankings[i][sample_idx])

        # Perform model pruning
        medians = np.median(scores[:, :-1], axis=0)
        percentile = np.percentile(scores[:, -1], 95)
        for i in range(len(self.base_models)):
            if medians[i] > percentile:
                scores[:, i] = np.inf

        # Compute weights
        for sample_idx in range(N_SAMPLES):
            minimum = np.min(scores[sample_idx])
            minima = np.where(scores[sample_idx] == minimum)[0]
            if np.sum(minima) == 1:
                weights[minima[0]] += 1
            elif len(self.base_models) in minima:
                weights[-1] += 1
            else:
                weights[self.rng.choice(minima)] += 1

        self.weights = weights / np.sum(weights)

        self.target_model.train(X, y)
        if np.sum(self.weights) == 0:
            super().compute_weights(X, y)