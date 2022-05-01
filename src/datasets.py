from typing import TypedDict, Any, Optional

import pandas as pd
import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.datasets.classification import iris, breast_cancer, adult
from deepchecks.tabular.datasets.regression import avocado

__all__ = ['get_dataset_options', 'DatasetOption']


class DatasetOption(TypedDict):
    train: Dataset
    test: Dataset
    model: Any
    features_importance: Optional[pd.Series]
    dataset_params: dict
    model_snippet: str
    contain_categorical_columns: bool


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
    with st.spinner('Loading datasets...'):
        sample_size = 1000
        iris_data = iris.load_data(as_train_test=True)
        avocado_data = avocado.load_data(as_train_test=True)
        breast_cancer_data = breast_cancer.load_data(as_train_test=True)
        # adult_data = adult.load_data(as_train_test=True)

        return {
            'avocado (regression)': DatasetOption(train=avocado_data[0].sample(sample_size),
                                     test=avocado_data[1].sample(sample_size),
                                     model=avocado.load_fitted_model(),
                                     features_importance=AVOCADO_FI,
                                     dataset_params=dict(label='AveragePrice', cat_features=['region', 'type'],
                                                         datetime_name='Date'),
                                     model_snippet=('from deepchecks.tabular.datasets.regression import avocado\n\n'
                                                    'model = avocado.load_fitted_model()'),
                                     contain_categorical_columns=True),
            'iris (classification)': DatasetOption(train=iris_data[0].sample(sample_size),
                                  test=iris_data[1].sample(sample_size),
                                  model=iris.load_fitted_model(),
                                  features_importance=None,
                                  dataset_params=dict(label='target', cat_features=[], label_type='classification_label'),
                                  model_snippet=('from deepchecks.tabular.datasets.classification import iris\n\n'
                                                 'model = iris.load_fitted_model()'),
                                  contain_categorical_columns=False),
            'breast_cancer (classification)': DatasetOption(train=breast_cancer_data[0].sample(sample_size),
                                           test=breast_cancer_data[1].sample(sample_size),
                                           model=breast_cancer.load_fitted_model(),
                                           features_importance=None,
                                           dataset_params=dict(label='target', cat_features=[],
                                                               label_type='classification_label'),
                                           model_snippet=('from deepchecks.tabular.datasets.classification import breast_cancer\n\n'
                                                          'model = breast_cancer.load_fitted_model()'),
                                           contain_categorical_columns=False),
            # 'adult (classification)': DatasetOption(train=adult_data[0].sample(sample_size),
            #                        test=adult_data[1].sample(sample_size),
            #                        model=adult.load_fitted_model(),
            #                        features_importance=None,
            #                        dataset_params=dict(label='income', cat_features=['workclass', 'education', 'marital-status',
            #                                            'occupation', 'relationship', 'race', 'sex', 'native-country'],
            #                                            label_type='classification_label'),
            #                        model_snippet=('from deepchecks.tabular.datasets.classification import adult\n\n'
            #                                       'model = adult.load_fitted_model()')),

        }
