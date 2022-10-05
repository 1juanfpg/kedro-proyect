"""
This is a boilerplate pipeline 'etl'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


from .nodes import (split_data,
                    load_data,
                    etl_processing, validation_data,
                    train_test_validation_dataset, data_integrity_validation
                    )