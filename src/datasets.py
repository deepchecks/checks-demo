from typing import TypedDict, Any, Dict

import pandas as pd
import streamlit as st

from deepchecks.tabular import Dataset
from deepchecks.tabular.datasets.regression import avocado
from deepchecks.tabular.datasets.classification import iris

__all__ = ['get_dataset_options', 'DatasetOption']


class DatasetOption(TypedDict):
    train: Dataset
    test: Dataset
    model: Any


@st.cache(show_spinner=False, hash_funcs={dict: lambda _: id})
def get_dataset_options():
    iris_data = iris.load_data(as_train_test=True)
    avocado_data = avocado.load_data(as_train_test=True)

    return {
        'iris': DatasetOption(train=iris_data[0], test=iris_data[1], model=iris.load_fitted_model()),
        'avocado': DatasetOption(train=avocado_data[0], test=avocado_data[1], model=avocado.load_fitted_model()),
    }


