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