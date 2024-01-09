from sktime.base import BaseEstimator
from CUSTOM_MODELS.CUSTOM_MODELS import FlattenedTransforms
from typing import Type
import numpy as np



class UniToMultivariateWrapper(BaseEstimator):
   def __init__(self, model: Type[BaseEstimator],var_thres=0.8, *args, **kwargs):
       if not issubclass(model, BaseEstimator):
           raise AttributeError('This is not a Model')
       super(UniToMultivariateWrapper, self).__init__()
       self.model=model
       self.n_components = 1
       self.pipe = FlattenedTransforms()
       self._is_fitted = False

   def __call__(self, *args, **kwargs):
       self.model = self.model(*args, **kwargs)
       return self
   def fit(self, X, y):
       y = np.asarray(y)
       X_new = self._pca_fit_transform(X)
       self.model.fit(X_new, y)
       self._is_fitted = True
       return self
   def predict(self, X):
       if not  self._is_fitted:
           raise AssertionError("Model needs to be fitted first")
       X_new = self._pca_transform(X)
       return self.model.predict(X_new)
   def predict_proba(self, X):
       if not  self._is_fitted:
           raise AssertionError("Model needs to be fitted first")
       X_new = self._pca_transform(X)
       return self.model.predict_proba(X_new)
   def _pca_fit_transform(self, X):
       X_new = self.pipe.fit_transform(X)
       return X_new
   def _pca_transform(self, X):
       X_new = self.pipe.transform(X)
       return X_new