from typing import Type

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sktime.datatypes._panel._convert import from_nested_to_2d_array


class TimeSeriesToPanelData(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=14):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        panel_data = self.process_to_panel(X, window_l=self.window_size)
        return panel_data

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def process_to_panel(series, window_l=14):
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        return pd.Series([pd.Series(window if window.shape[0] == window_l else
                                    pd.concat([window] + [window.tail(1) for _ in range(window_l - window.shape[0])])

                                    , name=None).reset_index(drop=True) for window in
                          series.rolling(window_l, min_periods=window_l)]).to_frame(series.name).set_index(series.index)
class FlattenedTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, single_dim=True):
        self.single_dim = single_dim
        if self.single_dim:
            self.pipe = Pipeline([('scaler', StandardScaler()),
                                  ('pca', PCA(n_components=1))])
        else:
            self.pipe = Pipeline([('scaler', StandardScaler())])
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_arr = np.array(from_nested_to_2d_array(X).T).reshape(-1, X.shape[1])
        if self.single_dim:
            X_arr = pd.DataFrame(self.pipe.transform(X_arr), columns=['single_dim'])
        else:
            X_arr = pd.DataFrame(self.pipe.transform(X_arr), columns=X.columns)

        X_arr=self._re_panel(X_arr,window_size=14)
        return X_arr
    def fit_transform(self, X, y=None):
        X_arr = np.array(from_nested_to_2d_array(X).T).reshape(-1, X.shape[1])
        if self.single_dim:
            X_arr = pd.DataFrame(self.pipe.fit_transform(X_arr), columns=['single_dim'])
        else:
            X_arr = pd.DataFrame(self.pipe.fit_transform(X_arr), columns=X.columns)

        X_arr=self._re_panel(X_arr,window_size=14)
        return X_arr
    @staticmethod
    def _re_panel(obj, window_size=None) -> pd.DataFrame:
        if window_size is None or obj.shape[0] < window_size:
            raise AttributeError(f"window can not be None or bigger than df.shape[0]")

        # obtain numeric columns

        numeric_cols = obj.select_dtypes(include=['float', 'integer']).columns.tolist()

        if len(numeric_cols) == 0:
            raise AttributeError('expecting numeric columns')

        df_ = pd.concat([pd.Series(
            [pd.Series(obj.loc[:, col].iloc[idx:idx + window_size]) for idx in range(0, obj.shape[0], window_size)])
            for col in obj.columns], axis=1)
        df_.columns =obj.columns

        return df_

