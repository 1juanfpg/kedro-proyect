"""
This is a boilerplate pipeline 'feature_enginnering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    pre_processing,
    first_processing,
    # data_type_split,
    post_processing
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   node(
                func=pre_processing,
                inputs=["x_train","y_train", "parameters"],
                outputs=["x_train_out", "y_train_out"],
                name="pre_processing",
            ),
            node(
                func=first_processing,
                inputs=["x_train_out", "parameters"],
                outputs=["x_train_transformed","first_processing_pipline"],
                name="first_processing",
            ),
            # node(
            #     func=data_type_split,
            #     inputs=["new_data_first", "parameters"],
            #     outputs=["x_train_transformed"],
            #     name="data_type_split",
            # ),
            node(
                func=post_processing,
                inputs=["x_train_transformed",
                        "y_train_out",
                        "parameters"],
                outputs=["x_train_model_input",
                         "y_train_model_input"],
                name="post_processing",
            )
        ]
    )
