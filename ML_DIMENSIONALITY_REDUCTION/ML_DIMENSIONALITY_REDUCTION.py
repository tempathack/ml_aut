import pandas as pd
from sklearn.preprocessing import StandardScaler
from LOGGER.LOGGING import WrapStack
from umap import UMAP


from ML_CONFIGS_UTILS.ML_CONFIGS import Config_Utils


class Reducers(Config_Utils):
    def __init__(self, dim_reducer, *args, **kwargs):
        super(Reducers, self).__init__()

        if not dim_reducer in self.configs['dim_reduction'].keys():
            raise KeyError(f"Transform is not supported use one of {self.configs['dim_reduction'].keys()}")

        self.dim_reducer = dim_reducer
        self.args = args
        if self._empty_dict(kwargs):
            self.kwargs = self.configs['dim_reduction'][self.dim_reducer]['default_kwargs']

        else:
            self.kwargs = kwargs

    def get_dim_reducer(self):
        method = self.configs['dim_reduction'][self.dim_reducer]['object']

        return method(*self.args, **self.kwargs)





class Ml_Reduce(Config_Utils):
    'Main Class to reduce trainings data '
    def __init__(self, X:pd.DataFrame,y:pd.DataFrame, *args, **kwargs):
        super().__init__()
        self.X = X
        self.y = self.eval_df(y)
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def _reduce(X, y=None,method=None, *args, **kwargs):

        dim_reducer=Reducers(method,*args,**kwargs).get_dim_reducer()

        if method=='LDA' :
            X_reduced = dim_reducer.fit_transform(X,y)
        else:
            X_reduced = dim_reducer.fit_transform(X)

        if not isinstance(X_reduced, pd.DataFrame):
            return pd.DataFrame(X_reduced,index=X.index).add_prefix(method)

        else:
            return X_reduced

    @WrapStack.FUNCTION_SCREEN
    def dimensionality_reduction(self,method=None,upper_limit_dim:int=20,*args,**kwargs):
        '''
        Main function to reduce dimensionality
        :param method: string representing of dimensionality reduction
        :param upper_limit: int  max dims before dim reduction
        :param args:
        :param kwargs:
        :return: pd.DataFrame of dimensionality reduction
        '''
        if not method in self.configs['dim_reduction'].keys() :
            raise KeyError("specify valid method first")


        if  self._validate_3d(self.X):
            raise AttributeError('Data can not be in 3D Shape for the purpose of reduction')

        if self.X.shape[1]>upper_limit_dim:
            pca_kwargs=kwargs.copy()
            pca_kwargs.update({'n_components': upper_limit_dim})
            self.X=self._reduce(self.X, self.y,method='PCA',*args,**pca_kwargs)

        scaler=StandardScaler()
        self.X=scaler.fit_transform(self.X)
        self.X=self._is_df(self.X,prefix='StandardScaler')

        return self._reduce(self.X, self.y,method=method,*args,**kwargs)