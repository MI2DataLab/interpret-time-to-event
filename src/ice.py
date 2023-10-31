# TODO: recode similarly to pdp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ice(model, var_name, var_values, data_x: pd.DataFrame):
    var_names = data_x.index.tolist()
    var_idx = var_names.index(var_name)
    x_v = data_x.values.reshape(1, -1)
    x_all = x_v.repeat(len(var_values), axis = 0)
    x_all[:, var_idx] = var_values
    surv_fs = model.predict_survival_function(x_all)
    return surv_fs


def plot_ice(model, var_name, var_values, data_x: pd.DataFrame, timesteps=np.arange(1, 2000)):
    surv_fs = _ice(model, var_name, var_values, data_x)
    for surv_f, var_value in zip(surv_fs, var_values):
        plt.plot(timesteps, surv_f(timesteps), label = var_value)
    plt.ylabel('Prediction')
    plt.xlabel('Timestep')
    plt.title(f'ICE plot for variable {var_name}')
    plt.ylim(0, 1.05)
    plt.xlim(timesteps[0], timesteps[-1])
    plt.legend()
    plt.grid()
    plt.show()