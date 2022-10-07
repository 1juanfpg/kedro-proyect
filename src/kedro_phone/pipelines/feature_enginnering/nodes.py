"""
This is a boilerplate pipeline 'feature_enginnering'
generated using Kedro 0.18.3
"""
import logging
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
import numpy as np

# Assemble pipeline(s)
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def pre_processing(x: pd.DataFrame,
                   y:pd.Series,
                   parameters: Dict[str, Any]) -> pd.DataFrame:
    """data processing only in the train data but not in the test data

    Args:
        data: Data train frame containing features.
    Returns:
        data: Processed data for training .
    """
    data = pd.concat([x,y],axis=1)

    data = (data
            .drop(['Unnamed: 0', 'index'], axis=1)
    )
                                       
    mlflow.set_experiment(parameters['target_column'])
    mlflow.log_param('pre-processing', "to_numeric, to_categorical")
    
    parameters['features'].remove('Unnamed: 0')
    parameters['features'].remove('hgv')
    parameters['features'].remove('index') 
    
    x_out = data[parameters['features']]
    y_out = data[parameters['target_column']]

    logger.info(f"Shape = {x_out.shape} pparameters['categorical_cols']re_processing")

    return x_out, y_out


def first_processing(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    create pipeline of General transformations to the data like creating new features.

    Args:
        data: train data after splitting
        parameters: list of the general transforms to apply to all the data

    Returns:
        pd.DataFrame: transformed data

    """
    logger.info(f"Shape = {data.shape} first_processing")
    parameters['numerical_cols'].remove('hgv')
    data = (data
            .pipe(to_categorical, categorical_cols=parameters['categorical_cols'])
            .pipe(to_numeric, numerical_cols=parameters['numerical_cols'])
            # .pipe(to_categorical_y, target_column=parameters['target_column'])
    )

    mlflow.set_experiment('readmission')
    mlflow.log_param('first-processing', 'to_categorical, to_numeric, to_categorical_y')

    return data,('first_processing', "hola")

# seccion data_split

# def data_type_split(data: pd.DataFrame, parameters: Dict[str, Any])-> pd.DataFrame:

#     if parameters['numerical_cols'] and parameters['categorical_cols']:
#         numerical_cols = parameters['numerical_cols']
#         categorical_cols = parameters['categorical_cols']
#     else:
#         numerical_cols = make_column_selector(dtype_include=np.number)(data)
#         categorical_cols = make_column_selector(dtype_exclude=np.number)(data)

#     mlflow.set_experiment(parameters['target_column'])    
#     mlflow.log_param('num_cols', numerical_cols)
#     mlflow.log_param('cat_cols', categorical_cols)

#     return data

# Post processing

def post_processing(x_in: np.ndarray, y_train: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
    """
    General processing to transformed data, like remove duplicates
    important after transformation the data types are numpy ndarray

    Args:
        x_in: x data after transformations
        y_train: y_train

    Returns:

    """
    methods = ["remove duplicates"]
    mlflow.set_experiment(parameters['target_column'])
    mlflow.log_param('post-processing', methods)

    y = y_train[parameters['target_column']].to_numpy().reshape(-1, 1)

    data = np.concatenate([x_in, y], axis=1)

    # remove duplicates
    data = np.unique(data, axis=0)
    y_out = data[:, -1]
    x_out = data[:, :-1]
    mlflow.log_param('shape post-processing', x_out.shape)
    return x_out, y_out

# --- help functions ---

def to_categorical(data: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    for x in categorical_cols:
        data[x] = data[x].astype('category')
    return data  

def to_numeric(data: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    for x in numerical_cols:
        data[x] = data[x].astype('int')
    return data  

def to_categorical_y(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    data[target_column] = data[target_column].astype('category')
    return data  