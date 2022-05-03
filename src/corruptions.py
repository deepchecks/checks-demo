import random

import numpy as np
import pandas as pd
from deepchecks.tabular import Dataset


def insert_numerical_drift(column: pd.Series, mean: float, std: float):
    return column + np.random.normal(mean, std, size=(column.shape[0]))


def insert_categorical_drift(column: pd.Series, percent: int, category: str):
    column = column.to_numpy()
    categories = list(set(np.unique(column)) - {category})
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


def insert_duplicates(data: pd.DataFrame, rows_to_duplicate_num: int, percent: int):
    np_data = data.to_numpy()
    indices_to_duplicate = np.random.choice(len(data), size=rows_to_duplicate_num, replace=False)
    probabilities = np.random.default_rng().random(rows_to_duplicate_num)
    probabilities = probabilities / np.sum(probabilities)
    amount_to_replace = min(int(len(data) * percent / 100), len(data))
    rows_to_put = np_data[np.random.choice(indices_to_duplicate, size=amount_to_replace, p=probabilities)]
    indices_to_replace = np.random.choice(len(data), size=amount_to_replace, replace=False)
    np_data[indices_to_replace] = rows_to_put
    return pd.DataFrame(np_data, columns=data.columns)


def relate_column_to_label(dataset: Dataset, column: pd.Series, label_power: float):
    return column + (dataset.data[dataset.label_name] * column.mean() * label_power)


def insert_variants(column: pd.Series, percent: int):
    def flip_case(x):
        return x.upper() if x.islower() else x.lower()

    functions = [
        lambda x: flip_case(x[0]) + x[1:],  # Switch upper lower first letter
        lambda x: x.replace(' ', '-'),  # Replace space with -
        lambda x: x + '.',  # Add . at the end
        lambda x: x.upper(),  # Upper case all letters
        lambda x: x.lower(),  # Lower case all letters
        lambda x: ' ' + x,  # Add space at the beginning
    ]

    column = column.to_numpy()
    value = np.random.choice(column)
    variants = [f(value) for f in functions if f(value) != value]
    size = min(int(len(column) * percent / 100), len(column))
    indices_to_replace = np.random.choice(len(column), size=size, replace=False)
    for index in indices_to_replace:
        column[index] = random.choice(variants)
    return column

