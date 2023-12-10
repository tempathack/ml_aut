from collections import defaultdict
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
import pandas as pd
from scipy.sparse import csr_matrix
from ML_CONFIGS_UTILS.ML_CONFIGS import Config_Utils
from LOGGER.LOGGING import WrapStack

class Transformers(Config_Utils):
    def __init__(self, transform, *args, **kwargs):
        super(Transformers, self).__init__()

        if not transform in self.checked_in_transforms:
            raise ValueError(f"Transform is not supported use one of {self.configs['transforms'].keys()}")

        self.transform = transform
        self.args = args
        if self._empty_dict(kwargs):
            self.kwargs = self.configs['transforms'][transform]['default_kwargs']

        else:
            self.kwargs = kwargs

    def get_transform(self):

        if self.transform in self.checked_in_transforms:
            method = self.configs['transforms'][self.transform]['object']
            return method(*self.args, **self.kwargs)
        else:
            raise ValueError("Transform is not supported")


class Ml_Process(Config_Utils):
    def __init__(self, X):
        super().__init__()
        self._validate_obj(X)
        self.X = X
        self.is_ts = self.TS_check(self.X)
        self.dimensions = X.shape[0]
        self.global_transform_track = defaultdict(list)

    @property
    def possible_transforms(self) -> dict:
        return self.configs['transforms'].keys()
    @property
    def possible_imputation(self) -> dict:
        return self.configs['imputers'].keys()
    @staticmethod
    def _handle_cat(obj, handle_cat) -> pd.DataFrame:
        cat_cols = obj.select_dtypes(exclude=['float', 'integer']).columns
        if handle_cat:

            mode_value = obj.loc[:, cat_cols].mode().to_dict()

            # Fill the missing values with mode
            obj.loc[:, cat_cols].fillna(value=mode_value, inplace=True)

            obj.loc[:, cat_cols] = obj.loc[:, cat_cols].astype('category').apply(lambda x: x.cat.codes)
            return obj
        else:
            return obj.drop(columns=cat_cols)

    @WrapStack.FUNCTION_SCREEN
    def main_transform(self, transform, handle_cat=True, *args, **kwargs) -> pd.DataFrame:
        transformer = Transformers(transform, *args, **kwargs).get_transform()

        is_3d = self._validate_3d(self.X)
        contains_nulls = self._validate_null(self.X)
        contains_categorical = self._validate_categorical(self.X)
        is_ts = self.TS_check(self.X)
        X = self.X

        if contains_categorical:
            X = self._handle_cat(X, handle_cat, *args, **kwargs)

        if contains_nulls:
            X = self._impute(X, is_3d, *args, **kwargs)

        if not is_3d and self.configs['transforms'][transform]['req_3d'] and is_ts:
            X = self.to_panel(X, *args, **kwargs, window_size=14)

        if is_3d and transform in self.configs['is_sk_transform'] and is_ts:
            transformer = TabularToSeriesAdaptor(transformer)
            transformed_df = transformer.fit_transform(X)
        else:  # tabular transformations
            transformed_df = transformer.fit_transform(X)

        if isinstance(transformed_df, csr_matrix):

            X = pd.DataFrame(transformed_df.todense(), index=X.index).add_suffix(
                '_' + transform)
        elif not isinstance(transformed_df, pd.DataFrame):
            X = pd.DataFrame(transformed_df, index=X.index).add_suffix('_' + transform)

        else:
            X = transformed_df.add_suffix('_' + transform)

        self.global_transform_track[transform].append(transformer)

        self.X = X
        return self.X