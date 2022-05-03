import numpy as np
import pandas as pd


def insert_numerical_drift(column: pd.Series, mean, std):
    return column + np.random.normal(mean, std, size=(column.shape[0]))


def insert_categorical_drift(column: pd.Series, percent, category):
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
