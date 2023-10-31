import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pdp(model, var_name, var_values, data_x: pd.DataFrame, timesteps, reduce: bool =True):
    var_idx = data_x.columns.tolist().index(var_name)
    n_obs = data_x.shape[0]
    n_values = len(var_values)
    data_x_mod = np.repeat(data_x.values, [n_values] * n_obs, axis = 0)
    data_x_mod[:, var_idx] = np.tile(var_values, n_obs)
    surv_fs = model.predict_survival_function(data_x_mod)
    ids_per_val = np.tile(np.identity(n_values, dtype = bool), n_obs)
    val_to_curve = {}
    for ids, val in zip(ids_per_val, var_values):
        surv_fs_sub = np.stack([f(timesteps) for f in surv_fs[ids]])
        if reduce:
            val_to_curve[val] = surv_fs_sub.mean(0)
        else:
            val_to_curve[val] = surv_fs_sub
    return val_to_curve


def plot_pdp(model, var_name, var_values, data_x: pd.DataFrame, timesteps=np.arange(1, 2000)):
    val_to_curve = _pdp(model, var_name, var_values, data_x, timesteps)
    for var_name_, curve in val_to_curve.items():
        plt.plot(timesteps, curve, label = var_name_)
    plt.ylabel('Expected prediction')
    plt.xlabel('Timestep')
    plt.title(f'PDP plot for variable {var_name}')
    plt.ylim(0, 1.05)
    plt.xlim(timesteps[0], timesteps[-1])
    plt.legend()
    plt.grid()
    plt.show()