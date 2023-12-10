import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import SpectralEmbedding
from LOGGER.LOGGING import WrapStack
from umap import UMAP


from ML_CONFIGS_UTILS.ML_CONFIGS import Config_Utils


class Ml_Reduce(Config_Utils):
    def __init__(self, X,y, *args, **kwargs):
        super().__init__()
        self.X = self.eval_df(X)
        self.y = self.eval_df(y)
        self.args = args
        self.kwargs = kwargs
    @staticmethod
    def _perform_pca(X, y=None, *args, **kwargs):

        trans=PCA(**kwargs)

        X_reduced=trans.fit_transform(X)

        if not isinstance(X_reduced,pd.DataFrame):
            return pd.DataFrame(X_reduced,index=X.index).add_prefix('PCA')

        else:
            return X_reduced
    @staticmethod
    def _perform_LDA(X, y=None, *args, **kwargs):

        trans =LDA(**kwargs)

        X_reduced = trans.fit_transform(X,y)

        if not isinstance(X_reduced, pd.DataFrame):
            return pd.DataFrame(X_reduced,index=X.index).add_prefix('LDA')

        else:
            return X_reduced
    @staticmethod
    def _perform_tsne(X, y=None, *args, **kwargs):

        trans = TSNE(**kwargs)

        X_reduced = trans.fit_transform(X)

        if not isinstance(X_reduced, pd.DataFrame):
            return pd.DataFrame(X_reduced,index=X.index).add_prefix('TSNE')

        else:
            return X_reduced
    @staticmethod
    def _perform_isomap(X, y=None, *args, **kwargs):

        trans = Isomap(**kwargs)

        X_reduced = trans.fit_transform(X)

        if not isinstance(X_reduced, pd.DataFrame):
            return pd.DataFrame(X_reduced,index=X.index).add_prefix('ISOMAP')

        else:
            return X_reduced
    @staticmethod
    def _perform_lle(X, y=None, *args, **kwargs):

        trans = LocallyLinearEmbedding(**kwargs)

        X_reduced = trans.fit_transform(X)

        if not isinstance(X_reduced, pd.DataFrame):
            return pd.DataFrame(X_reduced,index=X.index).add_prefix('LLE')

        else:
            return X_reduced
    @staticmethod
    def _perform_mds(X, y=None, *args, **kwargs):

        trans = MDS(**kwargs)

        X_reduced = trans.fit_transform(X)

        if not isinstance(X_reduced, pd.DataFrame):
            return pd.DataFrame(X_reduced,index=X.index).add_prefix('MDS')

        else:
            return X_reduced
    @staticmethod
    def _perform_se(X, y=None, *args, **kwargs):

        trans = SpectralEmbedding(**kwargs)

        X_reduced = trans.fit_transform(X)

        if not isinstance(X_reduced, pd.DataFrame):
            return pd.DataFrame(X_reduced,index=X.index).add_prefix('SE')

        else:
            return X_reduced
    @staticmethod
    def _perform_umap(X, y=None, *args, **kwargs):

        trans = UMAP(**kwargs)

        X_reduced = trans.fit_transform(X)

        if not isinstance(X_reduced, pd.DataFrame):
            return pd.DataFrame(X_reduced,index=X.index).add_prefix('UMAP')

        else:
            return X_reduced
    @WrapStack.FUNCTION_SCREEN
    def dimensionality_reduction(self,method=None,upper_limit=20,*args,**kwargs):

        if method is None :
            raise ValueError("specify method first")

        assert method in self.configs['dim_reduction'],f'Method {method} is not supported'

        self.is_2d(self.X),'Data can not be in 3D Shape for the purpose of reduction'

        if self.X.shape[1]>upper_limit:
            self.X=self._perform_pca(self.X, self.y, *args, **kwargs.update({'n_components':upper_limit}))


        if method=='PCA':
            res=self._perform_pca(self.X,self.y,*args,**kwargs)
        elif method=='LDA':
            res=self._perform_LDA(self.X, self.y, *args, **kwargs)
        elif method=='SE':
            res=self._perform_se(self.X, self.y, *args, **kwargs)
        elif method=='UMAP':
            res=self._perform_umap(self.X, self.y, *args, **kwargs)
        elif method=='LLE':
            res=self._perform_lle(self.X, self.y, *args, **kwargs)
        elif method=='MDS':
            res=self._perform_mds(self.X, self.y, *args, **kwargs)
        elif method=='TSNE':
            res = self._perform_tsne(self.X, self.y, *args, **kwargs)
        elif method=='ISOMAP':
            res = self._perform_isomap(self.X, self.y, *args, **kwargs)

        return res