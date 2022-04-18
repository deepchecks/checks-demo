import numpy as np
import pandas as pd
from deepchecks import BaseCheck, TrainTestBaseCheck


def insert_categorical_drift(column: pd.Series, percent, category):
    column = column.to_numpy()
    categories = np.delete(np.unique(column), [category])
    ratio = percent / 100
    category_count = np.count_nonzero(column == category)
    category_ratio = category_count / column.shape[0]
    amount_to_replace = int(column.shape[0] * (ratio - category_ratio))
    if amount_to_replace == 0:
        return column

    possible_indices = np.where(column != category) if amount_to_replace > 0 else np.where(column == category)
    indices_to_replace = np.random.choice(possible_indices[0], abs(amount_to_replace), replace=False)
    for index in indices_to_replace:
        column[index] = category if amount_to_replace > 0 else np.random.choice(categories)

    return column


def insert_numerical_drift(column: pd.Series, mean, std):
    return column + np.random.normal(mean, std, size=(column.shape[0])) * np.average(column)


def build_snippet(check: BaseCheck,
                  properties: dict = None,
                  model: bool = False,
                  condition_name: str = None,
                  condition_params: dict = None):
    check_name = check.__class__.__name__
    arguments = f'train=train_dataset, test=test_dataset' if isinstance(check, TrainTestBaseCheck) else 'dataset'
    if model:
        arguments += f', model=model'
    properties_string = ', '.join([f'{k}={v}' for k, v in properties.items()]) if properties else ''
    condition_params_string = ', '.join([f'{k}={v}' for k, v in condition_params.items()]) if condition_params else ''
    condition_string = f'.{condition_name}({condition_params_string})' if condition_name else ''
    snippet = (f'from deepchecks.tabular.checks import {check_name}\n'
               f'check = {check_name}({properties_string}){condition_string}\n'
               f'result = check.run({arguments})\n'
               'result.show()')

    return snippet
