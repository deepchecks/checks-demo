
import numpy as np
import pandas as pd
import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates

from datasets import DatasetOption


def run(dataset_option: DatasetOption):
    dataset: Dataset = dataset_option['train']
    new_data = dataset.data.copy()

    st.sidebar.subheader(f'Manipulate data')

    if st.sidebar.checkbox('Insert duplicates', value=True):
        new_data = insert_duplicates(new_data)

    check = DataDuplicates().add_condition_ratio_not_greater_than(0.1)
    return check.run(dataset.copy(new_data))


def insert_duplicates(new_data):
    rows_to_duplicate_num = 5
    rows_to_duplicate = new_data.sample(rows_to_duplicate_num).to_numpy()
    probabilities = np.random.default_rng().random(rows_to_duplicate_num)
    probabilities = probabilities / np.sum(probabilities)
    amount_to_duplicate = int(0.1 * len(new_data))
    indices_to_duplicate = np.random.choice(rows_to_duplicate_num, size=amount_to_duplicate, p=probabilities)
    added_data = rows_to_duplicate[indices_to_duplicate]

    return pd.concat([new_data, pd.DataFrame(added_data, columns=new_data.columns)])
