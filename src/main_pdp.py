import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from argparse import ArgumentParser
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score

from pdp import _pdp

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
    if args.merge:
        data_train_x = pd.concat([data_train_x, data_val_x], ignore_index = True)
        data_val_x = data_train_x.copy()
        data_train_y = np.concatenate([data_train_y, data_val_y])
        data_val_y = np.copy(data_train_y)
    if args.path_json is not None:
        assert args.merge
        assert args.split_id is not None
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
    col_names = np.array([name.split('_')[-1] for name in data_train_x.columns])
    group_names = np.unique(col_names)
    info = OrderedDict()
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

def compute_pdp(args, model, data_x, timesteps):
    results = OrderedDict()
    clinical_colnames = get_binary_clinical_colnames(data_x)
    var_values = [0, 1]
    for name in clinical_colnames:
        results_pdp = _pdp(model, name, var_values, data_x, timesteps, args.reduction)
        for key, value in results_pdp.items():
            results[f'{name}={key}'] = value
    if not args.reduction:
        # NOTE: If not aggregated with mean, we aggregate 
        #       by computing variance of the difference
        results_var = OrderedDict()
        for name in clinical_colnames:
            results_var[name] = (results[f'{name}=1'] - results[f'{name}=0']).var(0)
        results = results_var
    return results

def get_binary_clinical_colnames(data_x):
    clinical_colnames = [name for name in data_x.columns if 'clinical' in name]
    binary_values = np.array([0, 1])
    for name in clinical_colnames:
        var = data_x[name]
        uniq_values = np.unique(var)
        uniq_values.sort()
        if not np.array_equal(uniq_values, binary_values):
            clinical_colnames.remove(name)
    return clinical_colnames

def get_full_model_curve(model, data_x, timesteps):
    surv_fs = model.predict_survival_function(data_x)
    curve = np.array([f(timesteps) for f in surv_fs]).mean(0)
    return curve

def get_ibs(data_x, data_y, model):
    surv_fs = model.predict_survival_function(data_x)
    timesteps_data = [e[1] for e in data_y]
    timesteps = np.arange(min(timesteps_data), max(timesteps_data))
    preds = np.asarray([f(timesteps) for f in surv_fs])
    score = integrated_brier_score(data_y, data_y, preds, timesteps)
    return score

def save_ibs(args, data_val_x, data_val_y, model):
    score = get_ibs(data_val_x, data_val_y, model)
    if args.merge and args.path_json is None:
        full_train = True
    elif args.merge and args.path_json is not None:
        full_train = False
    else:
        full_train = False
    df = pd.DataFrame.from_dict({
        'model_id': args.model_id, 
        'dataset_id': f'{args.data_id}_diff-var={not args.reduction}_full-train={full_train}_split={args.split_id}', 
        'ibs': score}, 
        orient = 'index')
    df = df.transpose()
    df.columns = [None] * 3
    df.to_csv(args.path_perf_csv, mode = 'a', header = False, index = False)

def get_timesteps(args, data_y):
    timesteps_data = np.array([e[1] for e in data_y])
    timesteps = np.array([np.quantile(timesteps_data, v) for v in np.linspace(0., 0.98, args.n_timesteps)])
    timesteps = timesteps_data[abs(timesteps[None, :] - timesteps_data[:, None]).argmin(axis=0)]
    timesteps = np.unique(timesteps)
    if timesteps[0] == timesteps_data.min():
        timesteps[0] += 1
    if timesteps[-1] == timesteps_data.max():
        timesteps[-1] -= 1
    return timesteps

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
    if args.ibs_only:
        save_ibs(args, data_val_x, data_val_y, model)
    else:
        log.info(f'Calculating PDPs')
        results = OrderedDict()
        timesteps = get_timesteps(args, data_val_y)
        results['timesteps'] = timesteps
        results_pdp = compute_pdp(args, model, data_val_x, timesteps)
        results.update(results_pdp)
        log.info('Calculating full model curve')
        results['full_model'] = get_full_model_curve(model, data_val_x, timesteps)
        results.move_to_end('full_model', last = False)
        results.move_to_end('timesteps', last = False)
        log.info('Saving results')
        save_results(args, results)
        save_ibs(args, data_val_x, data_val_y, model)
        log.info('Finished')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path-train-data', type = str, help = 'Path to train part of the dataset')
    parser.add_argument('--path-val-data', type = str, help = 'Path to validation part of the dataset')
    parser.add_argument('--path-json', type = str, help = 'Path to json specifying crossvalidation splits')
    parser.add_argument('--path-save', type = str, help = 'Path where results will be saved, must finish with .csv')
    parser.add_argument('--path-perf-csv', type = str, help = "Path to csv for storing model's performance on each dataset")
    parser.add_argument('--split-id', type = int, help = 'Split number for crossvalidation')
    parser.add_argument('--model-id', type = str, default = 'rsf', help = 'Model ID used in performance csv')
    parser.add_argument('--data-id', type = str, help = 'Dataset ID used in performance csv')
    parser.add_argument('--n-timesteps', type = int, default = 50, help = 'Number of timesteps to evaluate the survival function on')
    parser.add_argument('--reduction', type = bool, default = True, help = 'Whether to apply mean over survival functions for each unique value (false) or use variance of difference (true)')
    parser.add_argument('--merge', type = bool, default = False, help = 'Whether to merge training and validation parts or not')
    parser.add_argument('--ibs-only', action = 'store_true', help = 'Whether to calculate IBS only and skip PDP')
    parser.add_argument('--seed', type = int, help = 'Seed')
    args = parser.parse_args()
    main(args)