from typing import TypedDict, Any, Optional

import pandas as pd
import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.datasets.classification import iris
from deepchecks.tabular.datasets.regression import avocado

__all__ = ['get_dataset_options', 'DatasetOption']


class DatasetOption(TypedDict):
    train: Dataset
    test: Dataset
    model: Any
    features_importance: Optional[pd.Series]


# The avocado model doesn't have FI and calculating it takes a long time and memory. so hard-coding it here.
AVOCADO_FI = pd.Series({
    'Total Volume': 0.073976,
    '4046': 0.170529,
    '4225': 0.131923,
    '4770': 0.000000,
    'Total Bags': 0.039963,
    'Small Bags': 0.057997,
    'Large Bags': 0.224358,
    'XLarge Bags': 0.025718,
    'type': 0.176636,
    'year': 0.000000,
    'region': 0.098899
})


@st.cache(show_spinner=False, hash_funcs={dict: lambda _: id})
def get_dataset_options():
    sample_size = 1000
    iris_data = iris.load_data(as_train_test=True)
    avocado_data = avocado.load_data(as_train_test=True)

    return {
        'iris': DatasetOption(train=iris_data[0].sample(sample_size), test=iris_data[1].sample(sample_size),
                              model=iris.load_fitted_model(), features_importance=None),
        'avocado': DatasetOption(train=avocado_data[0].sample(sample_size), test=avocado_data[1].sample(sample_size),
                                 model=avocado.load_fitted_model(), features_importance=AVOCADO_FI),
    }
