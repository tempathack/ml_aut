from typing import Type
from abc import ABCMeta

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sklearn.pipeline import Pipeline
from sktime.base import BaseEstimator
from sklearn.manifold import TSNE,LocallyLinearEmbedding
from umap import UMAP
import numpy as np
from sktime.datasets import load_unit_test
from sktime.transformations.panel.dwt import DWTTransformer
from sktime.transformations.series.acf import AutoCorrelationTransformer,PartialAutoCorrelationTransformer
import pandas as pd


class CustomDWTTransformer(DWTTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_transform(self, X, y=None):
        # Call the base class's fit_transform method

        max_idx = X.iloc[0].iloc[0].size

        super().fit(X, y)
        X = super().transform(X, y)
        # Add extra manipulations using self.extra_param
        # Additional processing here...
        return self.revert_structure(X, max_idx)

    def transform(self, X, y=None):
        # Call the base class's transform method
        X = super().transform(X)

    def revert_structure(self, X, max_idx):
        idx = X.iloc[:, 0].size
        for col in X.columns:
            for i in range(idx):
                X.loc[:, col].iloc[i] = self.fill_series_with_zeros(X.loc[:, col].iloc[i], max_idx)

        return X

    @staticmethod
    def fill_series_with_zeros(series, max_index):
        # Create a new index from 0 to max_index
        new_index = pd.RangeIndex(start=0, stop=max_index)

        # Reindex the series with the new index and fill NaN values with 0
        filled_series = series.reindex(new_index, fill_value=0)

        return filled_series

class CustomAutoCorrelationTransformer(AutoCorrelationTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_transform(self, X, y=None):
        # Call the base class's fit_transform method

        max_idx = X.iloc[0].iloc[0].size

        super().fit(X, y)
        X = super().transform(X, y)
        # Add extra manipulations using self.extra_param
        # Additional processing here...
        return self.revert_structure(X, max_idx)

    def transform(self, X, y=None):
        # Call the base class's transform method
        X = super().transform(X)

    def revert_structure(self, X, max_idx):
        idx = X.iloc[:, 0].size
        for col in X.columns:
            for i in range(idx):
                X.loc[:, col].iloc[i] = self.fill_series_with_zeros(X.loc[:, col].iloc[i], max_idx)

        return X

    @staticmethod
    def fill_series_with_zeros(series, max_index):
        # Create a new index from 0 to max_index
        new_index = pd.RangeIndex(start=0, stop=max_index)

        # Reindex the series with the new index and fill NaN values with 0
        filled_series = series.reindex(new_index, fill_value=0)

        return filled_series
class CustomPartialAutoCorrelationTransformer(PartialAutoCorrelationTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_transform(self, X, y=None):
        # Call the base class's fit_transform method

        max_idx = X.iloc[0].iloc[0].size


        super().fit(X, y)
        X = super().transform(X, y)
        # Add extra manipulations using self.extra_param
        # Additional processing here...
        return self.revert_structure(X, max_idx)
    def transform(self, X, y=None):
        # Call the base class's transform method
        X = super().transform(X)

    def revert_structure(self, X, max_idx):
        idx = X.iloc[:, 0].size
        for col in X.columns:
            for i in range(idx):
                X.loc[:, col].iloc[i] = self.fill_series_with_zeros(X.loc[:, col].iloc[i], max_idx)

        return X

    @staticmethod
    def fill_series_with_zeros(series, max_index):
        # Create a new index from 0 to max_index
        new_index = pd.RangeIndex(start=0, stop=max_index)

        # Reindex the series with the new index and fill NaN values with 0
        filled_series = series.reindex(new_index, fill_value=0)

        return filled_series
