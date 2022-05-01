import numpy as np
import pandas as pd
import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates

from datasets import DatasetOption
from utils import build_snippet


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    dataset: Dataset = dataset_option['train']
    new_data = dataset.data.copy()

    with check_param_col:
        st.text('No parameters to control')
    with manipulate_col:
        if st.checkbox('Insert duplicates', value=True):
            new_data = insert_duplicates(new_data)

    check = DataDuplicates().add_condition_ratio_not_greater_than(0.05)
    snippet = build_snippet(check, dataset_option, condition_name='add_condition_ratio_not_greater_than(0.05)')
    dataset = dataset.copy(new_data)
    return check.run(dataset), snippet, (dataset,)


def insert_duplicates(new_data):
    rows_to_duplicate_num = 5
    rows_to_duplicate = new_data.sample(rows_to_duplicate_num).to_numpy()
    probabilities = np.random.default_rng().random(rows_to_duplicate_num)
    probabilities = probabilities / np.sum(probabilities)
    amount_to_duplicate = int(0.1 * len(new_data))
    indices_to_duplicate = np.random.choice(rows_to_duplicate_num, size=amount_to_duplicate, p=probabilities)
    added_data = rows_to_duplicate[indices_to_duplicate]

    return pd.concat([new_data, pd.DataFrame(added_data, columns=new_data.columns)])
