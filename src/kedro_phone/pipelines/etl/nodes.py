"""
This is a boilerplate pipeline 'etl'
generated using Kedro 0.18.3
"""

import importlib
import logging
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from deepchecks.tabular.suites import train_test_validation

import great_expectations as ge
from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult

from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures

logger = logging.getLogger(__name__)

def load_data(parameters: Dict[str, Any]) -> pd.DataFrame:
    """Load data from path.
    Args:
        parameters['path']: path to data from.
    Returns: dataframe containing data.
    """
    data = pd.read_csv(parameters['path'])

    print("total data->",data.shape)

    return data

def etl_processing(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    General transformations to the data like removing columns with
    the same constant value, duplicated columns., duplicate values

    Args:
        data: raw data after extract
        parameters: list of the general transforms to apply to all the data

    Returns:
        pd.DataFrame: transformed data

    """
    # print (data.info())
    mlflow.set_experiment(parameters['target_column'])
    mlflow.log_param("shape raw_data", data.shape)

    data = (data
            .pipe(clean_data)
            .dropna(axis=0)
            .pipe(drop_duplicates, drop_cols=['index'])
    )

    # convert this step as a scikit-learn transformer
    # this steps is only useful until at the end in the SLQ query or
    # data load method, only query the specific columns
    # the specific columns
    columns = parameters['features']
    # convert target column (str) to list
    target = parameters['target_column']
    columns.append(target)


    pipe_functions = [
        # ('drop_constant_values', DropConstantFeatures(tol=1, missing_values= 'ignore')),
        ('drop_duplicates', DropDuplicateFeatures(missing_values= 'ignore'))                
    ]

    # get methods name for experimentation tracking
    methods = []
    for name, _ in pipe_functions:
        methods.append(name)

    mlflow.set_experiment(parameters['target_column'])
    mlflow.log_param('etl_transforms', methods)

    print(methods)

    pipeline_train_data = Pipeline(steps=pipe_functions)
    # print(pipeline_train_data)
    #apply transformation to data
    data_transformed = pipeline_train_data.fit_transform(data)
    print("----------------------------------------------")
    print(data.shape, data.info())

    mlflow.log_param("shape data etl", data_transformed.shape)

    return data_transformed

# --- Node ---

# integridad de los datos 
def data_integrity_validation(data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:

    categorical_features = parameters['categorical_cols']
    label = parameters['target_column']

    dataset = Dataset(data,
                 label=label,
                 cat_features=categorical_features)

    # Run Suite:
    integ_suite = data_integrity()
    suite_result = integ_suite.run(dataset)

    mlflow.set_experiment(parameters['target_column'])
    mlflow.log_param(f"data integrity validation", str(suite_result.passed()))
    
    if not suite_result.passed():
        # save report in data/08_reporting
        suite_result.save_as_html('data/08_reporting/data_integrity_check.html')
        logger.error("data integrity not pass validation tests")
        #raise Exception("data integrity not pass validation tests")
    return data

# separación de los datos en test y train
def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.
    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_processing.yml.
    Returns:
        Split data.
    """
    mlflow.set_experiment('price_range')
    mlflow.log_param("split random_state", parameters['split']['random_state'])
    mlflow.log_param("split test_size", parameters['split']['test_size'])

    #remove rows without target information
    data = data.dropna(subset=[parameters['target_column']])
    del parameters['features'][15]
    print(parameters['features'])
    # 
    x_features = data[parameters['features']]
    y_target = data[parameters['target_column']]

    x_train, x_test, y_train, y_test = train_test_split(
        x_features,
        y_target,
        test_size=parameters['split']['test_size'],
        random_state=parameters['split']['random_state']
    )

    mlflow.log_param(f"shape train", x_train.shape)
    mlflow.log_param(f"shape test", x_test.shape)

    return x_train, x_test, y_train, y_test

# Validación de los datos de salida 
def train_test_validation_dataset(x_train,
                                  x_test,
                                  y_train,
                                  y_test,
                                  parameters: Dict) -> Tuple:
    categorical_features = parameters['categorical_cols']
    label = parameters['target_column']

    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)

    train_ds = Dataset(train_df,
                       label=label,
                       cat_features=categorical_features
                       )
    test_ds = Dataset(test_df,
                      label=label,
                      cat_features=categorical_features
                      )
    validation_suite = train_test_validation()
    suite_result = validation_suite.run(train_ds, test_ds)
    
    mlflow.set_experiment(parameters['target_column'])
    mlflow.log_param("train_test validation", str(suite_result.passed()))
    
    if not suite_result.passed():
        # save report in data/08_reporting
        suite_result.save_as_html('data/08_reporting/train_test_check.html')
        logger.error("Train / Test Dataset not pass validation tests")
    return x_train, x_test, y_train, y_test

# --- help functions -------

# funcion para le limpiado de datos
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """ Replace target column values """
    data.replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data.replace('??????', np.nan, inplace=True)
    data.replace('nan', np.nan, inplace=True)
    data.replace('-948961565145.0', np.nan, inplace=True)
    data.replace('5285988458456.0', np.nan, inplace=True)
    print("total value clean_data ->",data.shape)
    return data

# remove duplicates from data based on a column
def drop_duplicates(data: pd.DataFrame,
                    drop_cols: list) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    data = data.drop_duplicates(subset=drop_cols, keep='first')
    print("Delete data duplicate")
    print("total value drop_duplicate->",data.shape)
    return data