"""
The "train -> validate -> predict" pipeline.
"""

import argparse
import numpy as np
import pandas as pd
import importlib
import json
import itertools
import os
import sys
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import time

import dankypipe.constants as c
from dankypipe import pipe


def fetch_data(job_name, **kwargs):
    config = pipe.download_config(job_name, **kwargs)
    config.update(pipe.build_feature_set(config['features'], **kwargs))
    config['train_all'] = {
        'x': pd.concat([config['train']['x'], config['validate']['x']], axis=0, sort=False),
        'y': pd.concat([config['train']['y'], config['validate']['y']], axis=0, sort=False)
    }

    return config


def metrics(y, yhat):
    return dict(
        auc=roc_auc_score(y, yhat),
        accuracy=accuracy_score(y, yhat),
        confusion_matrix=confusion_matrix(y, yhat),
        classification_report=classification_report(y, yhat)
    )


def validate(config, parameters, kwargs):
    train = config['train']
    val = config['validate']

    model = load_model(config)(parameters, kwargs)
    model.train(**train)
    yhat = model.predict(val['x'])

    return metrics(val['y'], yhat)


def predict(config, parameters):
    test = config['test']
    train_full = config['train_full']

    model = load_model(config)(parameters)
    model.train(**train_full)
    yhat = model.predict(test['x'])
    return yhat


def run_task(config):
    """runs the task specified in the config
    Check config schema for a list of valid tasks

    Args:
        config: dict

    Returns:
        None
    """
    task = config['task']
    job = config['job_name']
    params = config['model']['parameters']
    kwargs = config['model']['kwargs']

    if task == 'validate':
        results = validate(config, params, kwargs)
        pipe.upload_results(job, str(results), None)

    elif task == 'predict':
        predictions = predict(config, params)
        pipe.upload_results(job, None, predictions)

    elif task == 'validate_predict':
        results = validate(config, params, kwargs)
        predictions = predict(config, params)
        pipe.upload_results(job, str(results), predictions)

    elif task == 'tune' or task == 'tune_predict':
        if config['tuning']['search_type'] == 'grid':
            best_params, results = tune_grid(config)

        elif config['tuning']['search_type'] == 'stage_wise':
            best_params, results = tune_stage_wise(config)

        else:
            raise KeyError('Search type not defined')

        predictions = predict(config, best_params) if task == 'tune' else None
        pipe.upload_results(job, str(results) + str(best_params), predictions)


def _update_dict(obj, path, value):
    """updates(mutates) a nested value in a dict

    >>> obj = {"a": {"b": {"c": 10}, "d": 5}}
    >>> _update_dict(obj, 'a.b.c', 10)
    >>> obj
    out: {"a": {"b": {"c": 10}, "d": 5}}

    Args:
        obj: the dict to mutate
        path: the path to the value to mutate
        value: replacement value

    Returns:
        None
    """
    keys = path.split('.')
    for i in range(0, len(keys) - 1):
        obj = obj[keys[i]]
    obj[keys[len(keys) - 1]] = value


def tune_stage_wise(config):
    """"runs a stage-wise search on the validation set

    Args:
        config: dict
    Returns:
        (best_params, [(params, metrics)])
    """
    parameters = copy.deepcopy(config['model']['parameters'])
    updates = config['tuning']['parameters']
    metric = config['tuning']['metric']
    kwargs = config['model']['kwargs']

    # initialize all params to the first value in their list
    for path, values in updates.items():
        _update_dict(parameters, path, values[0])

    results = []
    for path, values in updates.items():
        candidate_parameters = copy.deepcopy(parameters)
        results = []
        for value in values:
            _update_dict(candidate_parameters, path, value)
            res = validate(config, candidate_parameters, kwargs)
            results.append((candidate_parameters, res))

        parameters = max(results, key=lambda x: x[1][metric])[0]

    return parameters, results


def tune_grid(config):
    """runs a gridsearch on the validation set

    Args:
        config: dict
    Returns:
        (best_params, [(params, metrics)])
    """
    parameters = config['model']['parameters']
    kwargs = config['model']['kwargs']
    updates = config['tuning']['parameters']
    metric = config['tuning']['metric']
    job = config['job_name']

    candidate_updates = itertools.product(*[
        [
            {
                updates[i]['name']: j
            } for j in updates[i]['values']
        ] for i, k in enumerate(updates)
    ])
    results = []

    tasks = list(candidate_updates)
    task_count = len(tasks)
    for i, c in enumerate(tasks):
        log(job, f'fitting task {i+1} of {task_count}')

        candidate_parameters = copy.deepcopy(parameters)
        candidate_parameters['metric'] = metric
        for d in c:
            for k, v in d.items():
                candidate_parameters[k] = v

        candidate_parameters = dict(params=candidate_parameters)
        res = validate(config, candidate_parameters, kwargs)

        log(job, res)
        results.append((candidate_parameters, res))

    best_parameters = max(results, key=lambda x: x[1][metric])[0]

    log(job, 'tuning completed')
    log(job, best_parameters)
    return best_parameters, results


def title(*args):
    """returns a title/header formatted string
    TODO: move to utils?

    >>> title('Hello')
    out: -----
         Hello
         -----

    Args:
      *args: [any]
    Returns:
        None
    """
    out = ' '.join(map(str, args))
    line = ('-'*len(out))
    return '\n'.join([line, out, line])


def load_model(config):
    """loads a model.

    Looks for a python file with the name
    'config.model.name' in the 'models' folder.

    The file should export a class called Model
    with the methods 'train' and 'predict'.

    Args:
        config: (dict)
    Returns:
        Model
    """
    model_path = 'dankypipe.models.' + config['model']['name']
    model = importlib.import_module(model_path)

    model_str = ""
    with open(model.__file__) as f:
        model_str += title('Model Source') + '\n'

        for line in f:
            if len(line) == 1:
                model_str += '\n'
            else:
                model_str += line.rstrip() + '\n'

    model_str += '\n------------end model\n\n'
    log(config['job_name'], model_str)

    return model.Model


def log(job, message):
    if isinstance(message, dict):
        message = json.dumps(message)
    elif not isinstance(message, str):
        message = str(message)

    with open(f'{job}_log.txt', 'w') as f:
        f.write(f'{c.now()}:  {message}\n')

    f.close()
    del f


def main():
    parser = argparse.ArgumentParser(description='--')
    parser.add_argument('job', type=str, help='HELP!')
    job = parser.parse_args().job

    a = time.time()
    log(job, 'building dataset')
    config = fetch_data(job)

    a, n = np.round(time.time()-a, 2), len(config["features"])
    log(job, f'downloaded {n} in {a} seconds ({np.round(a/n, 2)} feat/s)')

    log(job, 'building model')
    run_task(config)


if __name__ == '__main__':
    main()
