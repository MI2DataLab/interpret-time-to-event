import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from argparse import ArgumentParser
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import brier_score
from tqdm import tqdm

from pfi import _pfi_single_plot_values

import logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

def set_seed(args):
    log.info(f'Seed: {args.seed}')
    np.random.seed(args.seed)

def make_splits_from_path(path):
    data = pd.read_csv(path)
    data_x, data_y = data.iloc[:, 2:], data.iloc[:, :2]
    data_y = np.array(
        [(bool(r[0]), float(r[1])) for r in data_y.values[:, [1, 0]]],
        dtype = [('Status', '?'), ('Survival_in_days', '<f8')])
    return data_x, data_y

def load_dataset(args):
    data_train_x, data_train_y = make_splits_from_path(args.path_train_data)
    data_val_x, data_val_y = make_splits_from_path(args.path_val_data)
    info = OrderedDict()
    if args.path_json is not None:
        assert args.split_id is not None
        data_train_x = pd.concat([data_train_x, data_val_x], ignore_index = True)
        data_val_x = data_train_x.copy()
        data_train_y = np.concatenate([data_train_y, data_val_y])
        data_val_y = np.copy(data_train_y)
        info['timesteps_data'] = np.array([e[1] for e in data_train_y])
        with open(args.path_json, 'r') as f:
            splits = json.load(f)
        split_val_ids = splits[f'Fold0{args.split_id}' if args.split_id != 10 else 'Fold10']
        split_val_ids = [v - 1 for v in split_val_ids]
        split_val_ids = [idx in split_val_ids for idx in data_train_x.index]
        split_train_ids = [not v for v in split_val_ids]
        data_val_x = data_train_x.loc[split_val_ids]
        data_train_x = data_train_x.loc[split_train_ids]
        data_val_y = data_train_y[split_val_ids]
        data_train_y = data_train_y[split_train_ids]
    else:
        info['timesteps_data'] = None
    col_names = np.array([name.split('_')[-1] for name in data_train_x.columns])
    group_names = np.unique(col_names)
    info['groups'] = [(group_name, col_names == group_name) for group_name in group_names]
    return data_train_x, data_train_y, data_val_x, data_val_y, info

def make_model(args):
    rsf = RandomSurvivalForest(
        n_estimators = 200, 
        min_samples_split = 10, 
        min_samples_leaf = 15, 
        n_jobs = -1, 
        random_state = args.seed)
    return rsf

def compute_pfi(args, model, data_x, data_y, group_name, group_ids, n_permutations, score_per_timestep, method, timesteps_data):
    results = {}
    if args.permute_groups:
        var_names = data_x.columns[group_ids].tolist()
        timesteps, values = _pfi_single_plot_values(
            model, 
            var_names, 
            data_x, 
            data_y, 
            n_permutations, 
            score_per_timestep, 
            method,
            timesteps_data)
        results[group_name] = values
    else:
        for var_name in data_x.columns[group_ids]:    
            log.info(f'Variable name: {var_name}')
            timesteps, values = _pfi_single_plot_values(
                model, 
                var_name, 
                data_x, 
                data_y, 
                n_permutations, 
                score_per_timestep, 
                method)
            results[var_name] = values
    results['timesteps'] = timesteps
    return results

def get_full_model_scores(model, data_x, data_y, timesteps):
    surv_fs = model.predict_survival_function(data_x)
    preds = [f(timesteps) for f in surv_fs]
    _, scores = brier_score(data_y, data_y, preds, timesteps)
    return scores

def save_results(args, results):
    pd.DataFrame(results).to_csv(args.path_save)

def main(args):
    log.info('Running')
    log.info(f'args: {args}')
    set_seed(args)
    data_train_x, data_train_y, data_val_x, data_val_y, info = load_dataset(args)
    model = make_model(args)
    log.info('Training the model')
    model.fit(data_train_x, data_train_y)
    log.info('Training finished')
    log.info('Calculating PFIs')
    results = OrderedDict()
    for group_name, group_ids in tqdm(info['groups']):
        log.info(f'Group: {group_name}')
        group_results = compute_pfi(
            args,
            model, 
            data_val_x, 
            data_val_y,
            group_name,
            group_ids, 
            n_permutations = args.n_permutations, 
            score_per_timestep = brier_score, 
            method = 'clean',
            timesteps_data = info['timesteps_data'])
        results.update(group_results)
    log.info('Calculating full model scores')
    results['full_model'] = get_full_model_scores(model, data_val_x, data_val_y, results['timesteps'])
    results.move_to_end('full_model', last = False)
    results.move_to_end('timesteps', last = False)
    log.info('Saving results')
    save_results(args, results)
    log.info('Finished')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path-train-data', type = str, help = 'Path to train part of the dataset')
    parser.add_argument('--path-val-data', type = str, help = 'Path to validation part of the dataset')
    parser.add_argument('--path-json', type = str, help = 'Path to json with crossvalidation splits')
    parser.add_argument('--split-id', type = int, help = 'Split id for crossvalidation')
    parser.add_argument('--path-save', type = str, help = 'Path where results will be saved, must finish with .csv')
    parser.add_argument('--n-permutations', default = 25, type = int, help = 'Seed')
    parser.add_argument('--seed', type = int, help = 'Seed')
    parser.add_argument('--permute-groups', action = 'store_true')
    args = parser.parse_args()
    main(args)