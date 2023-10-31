import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sksurv.metrics import brier_score


def _pfi(
        model, 
        var_name, 
        data_x: pd.DataFrame, 
        data_y, 
        n_permutations,
        score_per_timestep,
        timesteps_data = None):
    
    if timesteps_data is None:
        timesteps_data = np.array([e[1] for e in data_y])
    else:
        assert len(timesteps_data.shape) == 1
    timesteps = np.array([np.quantile(timesteps_data, v) for v in np.arange(0., 0.98, 0.02)])
    timesteps = timesteps_data[abs(timesteps[None, :] - timesteps_data[:, None]).argmin(axis=0)]
    timesteps = np.unique(timesteps)
    if timesteps[0] == timesteps_data.min():
        timesteps[0] += 1
    if timesteps[-1] == timesteps_data.max():
        timesteps[-1] -= 1
    surv_fs = model.predict_survival_function(data_x)
    data_x_perm = data_x.copy(deep = True)
    surv_fs_per_perm = [None] * n_permutations
    for iter in range(n_permutations):
        data_x_perm[var_name] = np.random.permutation(data_x_perm[var_name])
        surv_f_perm = model.predict_survival_function(data_x_perm.values)
        surv_fs_per_perm[iter] = surv_f_perm
    min_data_timesteps = min([e[1] for e in data_y])
    max_data_timesteps = max([e[1] for e in data_y])
    timesteps = timesteps[timesteps > min_data_timesteps]
    timesteps = timesteps[timesteps < max_data_timesteps]
    preds = np.array([f(timesteps) for f in surv_fs])
    preds_perm = np.array([[f(timesteps) for f in fs] for fs in surv_fs_per_perm])
    _, scores = score_per_timestep(data_y, data_y, preds, timesteps)
    scores_perm = [score_per_timestep(data_y, data_y, p, timesteps)[1] for p in preds_perm]
    scores_perm_mean = np.stack(scores_perm).mean(axis = 0)
    return scores, scores_perm_mean, timesteps


def _pfi_single_plot_values(
        model, 
        var_name, 
        data_x: pd.DataFrame, 
        data_y, 
        n_permutations, 
        score_per_timestep,
        method,
        timesteps_data=None):
    brier_scores, brier_scores_perm_mean, timesteps = _pfi(
        model, var_name, data_x, data_y, n_permutations, score_per_timestep, timesteps_data)
    if method == 'divide':
        values = brier_scores_perm_mean / brier_scores
    elif method == 'subtract':
        values = brier_scores_perm_mean - brier_scores
    elif method == 'clean':
        values = brier_scores_perm_mean
    else:
        raise ValueError(f'Method {method} not recognized')
    return timesteps, values


def plot_pfi(
        model, 
        data_x, 
        data_y, 
        n_permutations=25, 
        score_per_timestep=brier_score, 
        method='divide',
        figure_kwargs={}):
    results = {}
    for var_name in data_x.columns:
        results[var_name] = _pfi_single_plot_values(
            model, 
            var_name, 
            data_x, 
            data_y, 
            n_permutations, 
            score_per_timestep, 
            method)
    fig = plt.figure(**figure_kwargs)
    for var_name, (timesteps, values) in results.items():
        plt.plot(timesteps, values, label = var_name)
        plt.ylabel(f'Importance with {method}')
        plt.xlabel(f'Timestep')
    plt.title('PFI plot')
    plt.xlim(timesteps[1], timesteps[-1])
    plt.legend()
    plt.grid()
    plt.show()