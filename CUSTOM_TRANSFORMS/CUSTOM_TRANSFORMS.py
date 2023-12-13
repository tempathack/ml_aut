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


class UniToMultivariateWrapper(BaseEstimator):
   def __init__(self, model: Type[BaseEstimator],var_thres=0.8, n_components: int = 1, *args, **kwargs):
       if not issubclass(model, BaseEstimator):
           raise AttributeError('This is not a Model')
       super(UniToMultivariateWrapper, self).__init__()
       self.model=model
       self.pipe = TabularToSeriesAdaptor(
           Pipeline([('scaler', StandardScaler()),
                     ('pca', PCA(n_components=var_thres)),
                     ('umap',UMAP(n_components=n_components))]))
       self._is_fitted = False
   def __call__(self, *args, **kwargs):
       self.model = self.model(*args, **kwargs)
       return self
   def fit(self, X, y):
       y = np.asarray(y)
       X_new = self.pca_transform(X)
       self.model.fit(X_new, y)
       self._is_fitted = True
       return self

   def predict(self, X):
       if not self.is_fitted:
           raise AssertionError("Model needs to be fitted first")
       X_new = self.pca_transform(X)
       return self.model.predict(X_new)

   def pca_transform(self, X):
       X_new = self.pipe.fit_transform(X)
       return X_new