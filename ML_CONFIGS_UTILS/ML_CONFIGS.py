from sktime.transformations.all  import (SummaryTransformer,WindowSummarizer,RandomIntervalFeatureExtractor,Rocket,MiniRocket,MiniRocketMultivariate,TSFreshRelevantFeatureExtractor,BoxCoxTransformer,LogTransformer,SqrtTransformer,Detrender,Deseasonalizer)
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures,StandardScaler,RobustScaler,PowerTransformer,QuantileTransformer,KBinsDiscretizer,KernelCenterer
from sktime.transformations.panel.dwt import DWTTransformer
from sktime.transformations.panel.dictionary_based import PAA,SFA
from sktime.transformations.panel.hog1d import HOG1DTransformer
from sktime.transformations.series.lag import  Lag
from sktime.transformations.series.acf import PartialAutoCorrelationTransformer,AutoCorrelationTransformer
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,log_loss
from sklearn.metrics import f1_score, precision_score,log_loss,accuracy_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression,LogisticRegression,ElasticNet,Ridge
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor,ExtraTreesClassifier,ExtraTreesRegressor,HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.kernel_approximation import RBFSampler,Nystroem
from sklearn.svm import SVC,SVR
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor,XGBClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from catboost import CatBoostClassifier,CatBoostRegressor
from sktime.classification.ensemble import BaggingClassifier
from sktime.classification.sklearn import RotationForest
from sktime.classification.deep_learning.cnn import CNNClassifier
from optuna import trial
from sktime.classification.deep_learning.fcn import FCNClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier,ElasticEnsemble
from sktime.performance_metrics.forecasting import MeanAbsoluteError,MeanAbsolutePercentageError,MedianSquaredError
from sktime.classification.hybrid import HIVECOTEV1,HIVECOTEV2
from sktime.classification.kernel_based import RocketClassifier,TimeSeriesSVC,Arsenal
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.dictionary_based import BOSSEnsemble,IndividualBOSS,WEASEL,MUSE,ContractableBOSS
from sktime.classification.interval_based import SupervisedTimeSeriesForest,TimeSeriesForestClassifier
from sktime.regression.kernel_based import RocketRegressor
from feature_engine.discretisation import DecisionTreeDiscretiser,ArbitraryDiscretiser,EqualWidthDiscretiser,GeometricWidthDiscretiser
from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sktime.regression.deep_learning import CNNRegressor,TapNetRegressor
from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from feature_engine.discretisation import DecisionTreeDiscretiser,ArbitraryDiscretiser,EqualWidthDiscretiser,GeometricWidthDiscretiser
from sktime.forecasting.compose import DirectTabularRegressionForecaster
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor,KNeighborsTransformer
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from CUSTOM_MODELS.CUSTOM_MODELS import  TimeSeriesToPanelData
from CUSTOM_MODELS.MODEL_UTILS import UniToMultivariateWrapper
from CUSTOM_TRANSFORMS.CUSTOM_TRANSFORMS import CustomDWTTransformer,CustomMathTransformer,CustomPartialAutoCorrelationTransformer,CustomAutoCorrelationTransformer
import time
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.preprocessing import label_binarize
from umap import UMAP
from sklearn.manifold import SpectralEmbedding
import threading
from collections import defaultdict
from sktime.datatypes import check_is_scitype
from typing import Optional,Dict,List,Literal,Set,Tuple,Union,Callable,Any

def unpack_args(func):
    from functools import wraps
    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper
def multiclass_auc(y_true, y_pred, average="macro"):
    """
    Compute multiclass AUC.

    Parameters:
    y_true (array-like): True class labels.
    y_pred (array-like): Predicted probabilities for each class.
    average (str): 'macro', 'weighted', or 'micro'. Determines the type of averaging to be performed.

    Returns:
    float: Multiclass AUC score.
    """
    # Binarize the true labels to use them in roc_auc_score
    classes = np.unique(y_true)
    y_true_binarized = label_binarize(y_true, classes=classes)

    # Check if y_pred is a probability distribution; if not, raise an error
    if y_pred.shape[1] != len(classes):
        raise ValueError("y_pred should be a matrix of probabilities, one for each class.")

    # Calculate the AUC
    auc_score = roc_auc_score(y_true_binarized, y_pred, average=average, multi_class='ovr')
    return auc_score

class FunctionTimer:
    def __init__(self, func, timeout=5):
        self.func = func
        self.timeout = timeout
        self.result = None
        self.exception = None
        self.elapsed_time = None
        self.thread = None


    def run(self, *args, **kwargs):
        def target():
            start_time = time.time()
            try:
                self.result = self.func(*args, **kwargs)
            except Exception as e:
                self.exception = e
            finally:
                self.elapsed_time = time.time() - start_time

        self.thread = threading.Thread(target=target)
        self.thread.start()

        self.thread.join(self.timeout)
        if self.thread.is_alive():
            # Function or thread took too long, terminate it
            return None

        if self.exception:
            raise self.exception

        return self.result




class Config_Utils():
    def __init__(self):
        'check in section for all function parameters and so on'
        self.configs={}

        self.configs['transforms']={'SummaryTransformer':{'object':SummaryTransformer,'ts_only':True,'req_3d':True,'default_kwargs':{}},
                                    'WindowSummarizer': {'object': WindowSummarizer, 'ts_only': True,
                                                            'req_3d': True, 'default_kwargs': {}},
                                    'RandomIntervalFeatureExtractor': {'object':RandomIntervalFeatureExtractor, 'ts_only': True,
                                                          'req_3d': True, 'default_kwargs': {}},
                                    'DWTTransformer': {'object': CustomDWTTransformer,
                                                                       'ts_only': True,
                                                                       'req_3d': True, 'default_kwargs': {}},
                                    'PAA': {'object': PAA,
                                                       'ts_only': True,
                                                       'req_3d': True, 'default_kwargs': {}},
                                    'Rocket': {'object': Rocket,'ts_only': True, 'req_3d': True, 'default_kwargs': {}},
                                    'MiniRocketMultivariate': {'object': MiniRocketMultivariate, 'ts_only': True, 'req_3d': True,
                                                    'default_kwargs': {}},
                                    'TSFreshFeatureExtractor': {'object': TSFreshFeatureExtractor, 'ts_only': True,
                                                                'req_3d': True,
                                                                'default_kwargs': {}},
                                    'HOG1DTransformer': {'object': HOG1DTransformer, 'ts_only': True,
                                                                'req_3d': True,
                                                                'default_kwargs': {}},
                                    'BoxCoxTransformer': {'object': BoxCoxTransformer,
                                                                         'ts_only': True,
                                                                         'req_3d': False,
                                                                         'default_kwargs': {}},
                                    'Lag': {'object': Lag,
                                                          'ts_only': True,
                                                          'req_3d': False,
                                                          'default_kwargs': {'lags':[1,3,5,10],
                                                                            'index_out':'original'}},
                                    'StandardScaler': {'object': StandardScaler,
                                                        'ts_only':False,
                                                        'req_3d': False,
                                                        'default_kwargs': {}},
                                    'PowerTransformer': {'object': PowerTransformer,
                                                        'ts_only': False,
                                                        'req_3d': False,
                                                        'default_kwargs': {}},
                                    'QuantileTransformer': {'object': QuantileTransformer,
                                                        'ts_only': False,
                                                        'req_3d': False,
                                                        'default_kwargs': {}},
                                    'CustomMathTransformer': {'object': CustomMathTransformer,
                                                            'ts_only': False,
                                                            'req_3d': False,
                                                            'default_kwargs': {}},
                                    'EqualWidthDiscretiser': {'object': EqualWidthDiscretiser,
                                                                  'ts_only': False,
                                                                  'req_3d': False,
                                                                  'default_kwargs': {}},
                                    'RobustScaler': {'object': RobustScaler,
                                                             'ts_only': False,
                                                             'req_3d': False,
                                                             'default_kwargs': {}},
                                    'MinMaxScaler': {'object': MinMaxScaler,
                                                          'ts_only': False,
                                                          'req_3d': False,
                                                          'default_kwargs': {}},
                                    'PolynomialFeatures':{'object': PolynomialFeatures,
                                                          'ts_only': False,
                                                          'req_3d': False,
                                                          'default_kwargs': {}},
                                    'RBFSampler':{'object': RBFSampler,
                                                           'ts_only': False,
                                                           'req_3d': False,
                                                           'default_kwargs': {}},
                                    'Nystroem': {'object': Nystroem,
                                                   'ts_only': False,
                                                   'req_3d': False,
                                                   'default_kwargs': {}},
                                    }

        self.configs['is_sk_transform']=['RobustScaler', 'MinMaxScaler', 'QuantileTransformer',
                           'StandardScaler', 'PowerTransformer']

        self.configs['imputers']={'KNNImputer': KNNImputer,
                    'SimpleImputer': SimpleImputer}

        self.configs['feat_selections']={'Regression':['correlation', 'f_regression', 'DecisionTreeRegressor', 'RandomForestRegressor', 'mutual_info_regression', 'LassoCV','all'],
                              'Classification':['permutation importance', 'f_classif', 'chi2', 'DecisionTreeClassifier', 'RandomForestClassifier', 'mutual_info_classif', 'LogisticRegressionCV','all']}

        self.configs['dim_reduction']={'LDA': {'object':LDA,'default_kwargs':{}},
                                       'TSNE':{'object':TSNE,'default_kwargs':{}}, 'PCA': {'object':PCA,'default_kwargs':{}}, 'SE':{'object':SpectralEmbedding,'default_kwargs':{}},
                                       'UMAP': {'object':UMAP,'default_kwargs':{}}, 'LLE': {'object':LocallyLinearEmbedding,'default_kwargs':{'eigen_solver':'dense'}},
                                       'MDS': {'object':MDS,'default_kwargs':{}}, 'ISOMAP': {'object':Isomap,'default_kwargs':{}}, None: {'object':None,'default_kwargs':{}}}

        self.configs['metrics']={'tab':{'Regression':{'mean_squared_error':(mean_squared_error,{}),
                                               'mean_absolute_error':(mean_absolute_error,{}),
                                               'r2_score':(r2_score,{}),'explained_variance_score':(explained_variance_score,{}),
                                               'mean_absolute_percentage_error':(mean_absolute_percentage_error,{})},
                                 'Classification':{'binary':{'log_loss':(log_loss,{}),'accuracy':(accuracy_score,{}),
                                                               'f1':(f1_score,{}),'auc':(roc_auc_score,{}),'percision':(precision_score,{})}
                                                   ,'multiclass':{'log_loss':(log_loss,{}),'accuracy':(accuracy_score,{}),
                                                                  'auc':(multiclass_auc,{})}}},
                                'ts' :{'Regression': {'mean_squared_error': (MedianSquaredError, {}),
                                                 'mean_absolute_error': (MeanAbsoluteError, {}),
                                                 'mean_absolute_percentage_error': (
                                                 MeanAbsolutePercentageError, {})},
                                  'Classification': {'binary':{'log_loss': (log_loss, {}), 'accuracy': (accuracy_score, {}),
                                                     'f1': (f1_score, {}), 'auc': (roc_auc_score, {}),
                                                     'percision': (precision_score, {})},'multiclass':{}}}}

        self.configs['models']={'Regression':{'LinearRegression':{'object':LinearRegression,'ts_only':False,'req_3d':False,'is_sklearn':True,'default_kwargs':{}},
                                              'ElasticNet':{'object':ElasticNet,'ts_only':False,'req_3d':False,'is_sklearn':True,'default_kwargs':{}},
                                              'HistGradientBoostingRegressor': {'object':HistGradientBoostingRegressor , 'ts_only': False, 'req_3d': False,
                                                             'is_sklearn': True, 'default_kwargs': {}},
                                              'Ridge':{'object':Ridge,'ts_only':False,'req_3d':False,'is_sklearn':True,'default_kwargs':{}},
                                              'XGBRegressor':{'object':XGBRegressor,'ts_only':False,'req_3d':False,'is_sklearn':True,'default_kwargs':{}},
                                              'CatBoostRegressor': {'object': CatBoostRegressor, 'ts_only': False,
                                                               'req_3d': False, 'is_sklearn': True,
                                                               'default_kwargs': {}},
                                              'LGBMRegressor': {'object': LGBMRegressor, 'ts_only': False,
                                                               'req_3d': False, 'is_sklearn': True,
                                                               'default_kwargs': {}},
                                              'ExtraTreesRegressor': {'object': ExtraTreesRegressor, 'ts_only': False,
                                                                'req_3d': False, 'is_sklearn': True,
                                                                'default_kwargs': {}},
                                              'KNeighborsRegressor': {'object': KNeighborsRegressor, 'ts_only': False,
                                                                      'req_3d': False, 'is_sklearn': True,
                                                                      'default_kwargs': {}},
                                              'RandomForestRegressor': {'object': RandomForestRegressor, 'ts_only': False,
                                                                'req_3d': False, 'is_sklearn': True,
                                                                'default_kwargs': {}},
                                              'AdaBoostRegressor': {'object': AdaBoostRegressor,
                                                                        'ts_only': False,
                                                                        'req_3d': False, 'is_sklearn': True,
                                                                        'default_kwargs': {}},
                                              'MLPRegressor': {'object': MLPRegressor,
                                                                    'ts_only': False,
                                                                    'req_3d': False, 'is_sklearn': True,
                                                                    'default_kwargs': {}},
                                              'RocketRegressor': {'object': RocketRegressor,
                                                               'ts_only': True,
                                                               'req_3d': True, 'is_sklearn': False,
                                                               'default_kwargs': {}},
                                              'KNeighborsTimeSeriesRegressor': {'object': KNeighborsTimeSeriesRegressor,
                                                                  'ts_only':True,
                                                                  'req_3d': True, 'is_sklearn': False,
                                                                  'default_kwargs': {}},
                                              'CNNRegressor': {'object': CNNRegressor,
                                                                                'ts_only': True,
                                                                                'req_3d': True, 'is_sklearn': False,
                                                                                'default_kwargs': {}},
                                              'TapNetRegressor': {'object': TapNetRegressor,
                                                               'ts_only': True,
                                                               'req_3d': True, 'is_sklearn': False,
                                                               'default_kwargs': {}},

                                              },
                                'Classification':{'LogisticRegression':{'object':LogisticRegression,'ts_only':False,'req_3d':False,'is_sklearn':True,'binary_only':False,'default_kwargs':{}},
                                                'XGBClassifier':{'object':XGBClassifier,'ts_only':False,'req_3d':False,'binary_only':False,'is_sklearn':True,'default_kwargs':{'verbosity':0}},
                                'CatBoostClassifier': {'object': CatBoostClassifier, 'ts_only': False, 'req_3d': False,'binary_only':False,'is_sklearn': True, 'default_kwargs': {'silent':True}},
                                'LGBMClassifier': {'object': LGBMClassifier, 'ts_only': False,'binary_only':False, 'req_3d': False,
                                                  'is_sklearn': True, 'default_kwargs': {'verbosity':-100}},
                                                  'HistGradientBoostingClassifier': {
                                                      'object': HistGradientBoostingClassifier, 'ts_only': False,
                                                      'req_3d': False,
                                                      'is_sklearn': True, 'default_kwargs': {}},
                                                  'ExtraTreesClassifier': {'object': ExtraTreesClassifier, 'ts_only': False,'binary_only':False,
                                                                     'req_3d': False,
                                                                     'is_sklearn': True, 'default_kwargs': {}},
                                'RandomForestClassifier': {'object': RandomForestClassifier, 'ts_only': False,'binary_only':False, 'req_3d': False,
                                        'is_sklearn': True, 'default_kwargs': {}},
                                                  'KNeighborsClassifier': {'object':KNeighborsClassifier,'binary_only':False,
                                                                             'ts_only': False, 'req_3d': False,
                                                                             'is_sklearn': True, 'default_kwargs': {}},
                                'DecisionTreeClassifier': {'object': DecisionTreeClassifier, 'ts_only': False,'binary_only':False,
                                                           'req_3d': False,
                                                           'is_sklearn': True, 'default_kwargs': {}},
                                'AdaBoostClassifier': {'object': AdaBoostClassifier, 'ts_only': False,'binary_only':False,
                                                       'req_3d': False,
                                                       'is_sklearn': True, 'default_kwargs': {}},
                                'MLPClassifier': {'object': MLPClassifier,
                                                 'ts_only': False,
                                                 'req_3d': False, 'is_sklearn': True,'binary_only':False,
                                                 'default_kwargs': {}},
                                'CNNClassifier': {'object': CNNClassifier,
                                                      'ts_only': True,
                                                      'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                                      'default_kwargs': {}},
                                'FCNClassifier': {'object': FCNClassifier,
                                                  'ts_only': True,
                                                  'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                                  'default_kwargs': {}},
                                'KNeighborsTimeSeriesClassifier': {'object': KNeighborsTimeSeriesClassifier,
                                                 'ts_only': True,
                                                 'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                                 'default_kwargs': {}},
                                'HIVECOTEV2': {'object': HIVECOTEV2,
                                               'ts_only': True,
                                               'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                               'default_kwargs': {}},
                                                  'HIVECOTEV1': {'object': UniToMultivariateWrapper(HIVECOTEV1),
                                                                 'ts_only': True,
                                                                 'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                                                 'default_kwargs': {}},
                                                  'MUSE': {'object': MUSE,
                                                                 'ts_only': True,
                                                                 'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                                                 'default_kwargs': {'support_probabilities':True}},
                                                  'ContractableBOSS': {'object': UniToMultivariateWrapper(ContractableBOSS),
                                                           'ts_only': True,
                                                           'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                                           'default_kwargs': {}},
                                'RocketClassifier': {'object': RocketClassifier,
                                               'ts_only': True,
                                               'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                               'default_kwargs': {}},
                                                  'ElasticEnsemble': {'object': UniToMultivariateWrapper(ElasticEnsemble),
                                                                       'ts_only': True,
                                                                       'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                                                       'default_kwargs': {'proportion_of_param_options':0.1,
                                                                                          'proportion_train_for_test':0.1,
                                                                                          'distance_measures':['dtw','ddtw'],
                                                                                          'majority_vote':True}},
                                'TimeSeriesSVC': {'object': TimeSeriesSVC,'binary_only':False,
                                               'ts_only': True,
                                               'req_3d':True, 'is_sklearn': False,
                                               'default_kwargs': {'probability':True}},
                                'ShapeletTransformClassifier': {'object': UniToMultivariateWrapper(ShapeletTransformClassifier),
                                               'ts_only': True,
                                               'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                               'default_kwargs': {}},
                                                  'BOSSEnsemble': {
                                                      'object': UniToMultivariateWrapper(BOSSEnsemble),
                                                      'ts_only': True,
                                                      'req_3d': True, 'is_sklearn': False,
                                                      'default_kwargs': {}},
                                                  'IndividualBOSS': {
                                                      'object': UniToMultivariateWrapper(IndividualBOSS),
                                                      'ts_only': True,
                                                      'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                                      'default_kwargs': {}},
                            'WEASEL': {'object':UniToMultivariateWrapper(WEASEL),
                                                 'ts_only': True,
                                                'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                                'default_kwargs': {'support_probabilities':True}},
                             'TimeSeriesForestClassifier': {'object': UniToMultivariateWrapper(TimeSeriesForestClassifier),
                                          'ts_only': True,
                                       'req_3d': True, 'is_sklearn': False,'binary_only':False,
                              'default_kwargs': {}},
                                'SupervisedTimeSeriesForest': {'object':  UniToMultivariateWrapper(SupervisedTimeSeriesForest),
                                                             'ts_only': True,
                                                            'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                                            'default_kwargs': {}},
                                'Arsenal': {'object':Arsenal,
                                                 'ts_only': True,
                                                'req_3d': True, 'is_sklearn': False,'binary_only':False,
                                            'default_kwargs': {}},
                               'RotationForest': {'object':RotationForest,
                                   'ts_only': True,
                              'req_3d': False, 'is_sklearn': True,'binary_only':False,
                              'default_kwargs': {}},
                                }}
        self.configs['n_cvs']=10

        self.configs['hyperparamters']={'LogisticRegression':{'penalty': trial.suggest_categorical('penalty', ['l2']),
                    'C': trial.suggest_float("C", 0.5, 1.5),
                    'max_iter': trial.suggest_categorical('max_iter', [100, 200, 300]),
                    'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg'
                        , 'newton-cholesky', 'sag', 'saga'])},
                                        'LinearRegression':{},
                                        'HistGradientBoostingClassifier':{
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
            },'HistGradientBoostingRegressor':{
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
            },'SVC':{},'SVR':{'C': trial.suggest_float("C", 0.5, 1.5),
                    'kernel': trial.suggest_categorical('kernel',
                                                        ['linear', 'poly', 'rbf', 'sigmoid']),

                    },'XGBClassifier':{
                # this parameter means using the GPU when training our model to speedup the training process
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                'verbosity': 0,
                'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                              [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
                'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
                'n_estimators': trial.suggest_categorical('n_estimators', [i for i in range(1, 2000, 100)]),
                'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
                'random_state': trial.suggest_categorical('random_state', [2020]),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            },'RandomForestClassifier':{
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
            },'ExtraTreesRegressor':{
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
                'warm_start': trial.suggest_categorical('warm_start', [True, False])
            },'ExtraTreesClassifier':{
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
                'warm_start': trial.suggest_categorical('warm_start', [True, False])
            },'DecisionTreeClassifier':{'max_depth': trial.suggest_int('max_depth', 4, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
                    },'DecisionTreeRegressor':{'max_depth': trial.suggest_int('max_depth', 4, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
                    },'AdaBoostClassifier':{
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
            },'AdaBoostRegressor':{
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            },'RandomForestRegressor':{
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
            },'XGBRegressor':{
                # this parameter means using the GPU when training our model to speedup the training process
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                'verbose': 0,
                'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                              [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
                'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
                'n_estimators': trial.suggest_categorical('n_estimators', [i for i in range(1, 2000, 100)]),
                'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
                'random_state': trial.suggest_categorical('random_state', [2020]),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            },'LGBMRegressor':{
                "n_estimators": trial.suggest_categorical('n_estimators', [i for i in range(1, 2000, 100)]),
                "verbose": 0,
                "bagging_freq": 1,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 2 ** 10),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100)},
                                        'LGBMClassifier': {
                "n_estimators": trial.suggest_categorical('n_estimators', [i for i in range(1, 2000, 100)]),
                "verbose": 0,
                "bagging_freq": 1,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 2 ** 10),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100)},
                                        'CatBoostClassifier':{
                "n_estimators": trial.suggest_categorical('n_estimators', [i for i in range(1, 2000, 100)]),
                'leaf_estimation_iterations': 1,
                'boosting_type': 'Plain',
                'thread_count': -1,
                'silent': True,
                'depth': trial.suggest_int('depth', 4, 16),
                'random_strength': trial.suggest_int('random_strength', 0, 100),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.01, 100.00),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1),
                'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                'l2_leaf_reg': 50,
            },'CatBoostRegressor':{
                "n_estimators": trial.suggest_categorical('n_estimators', [i for i in range(1, 2000, 100)]),
                'leaf_estimation_iterations': 1,
                'boosting_type': 'Plain',
                'silent': True,
                'depth': trial.suggest_int('depth', 4, 16),
                'random_strength': trial.suggest_int('random_strength', 0, 100),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.01, 100.00),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1),
                'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                'l2_leaf_reg': 50},'ElasticNet':{'alpha': trial.suggest_float('alpha', 0.03, 2),
                    'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 2)},'Ridge':{'alpha': trial.suggest_float('alpha', 0.03, 2)},
                                        'MLPClassifier':{'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(100,), (300,), (500,)]),
                    'activation': trial.suggest_categorical('activation', ['logistic', 'relu', 'tanh'])},
                                        'MLPRegressor':{'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(100,), (300,), (500,)]),
                    'activation': trial.suggest_categorical('activation', ['logistic', 'relu', 'tanh'])},'KNeighborsClassifier':
                                            {'n_neighbors': trial.suggest_categorical('n_neighbors',
                                                                                      [i for i in range(1, 8)])},'KNeighborsRegressor':
                                            {'n_neighbors': trial.suggest_categorical('n_neighbors',
                                                                                      [i for i in range(1, 8)])},'TapNetRegressor':{
                'n_epochs': trial.suggest_int("n_epochs", 100, 3000),
                'batch_size': trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
                'dropout': trial.suggest_uniform("dropout", 0.0, 1.0),
                'filter_sizes': trial.suggest_categorical("filter_sizes", [(256, 256, 128), (128, 128, 64)]),
                'kernel_size': trial.suggest_categorical("kernel_size", [(8, 5, 3), (3, 3, 3)]),
                'dilation': trial.suggest_int("dilation", 1, 10),
                'layers': trial.suggest_categorical("layers", [(500, 300), (300, 100)]),
                'activation': trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"]),
                'loss': trial.suggest_categorical("loss", ["mean_squared_error", "mean_absolute_error"]),
                'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"]),
                'use_bias': trial.suggest_categorical("use_bias", [True, False]),
                'use_rp': trial.suggest_categorical("use_rp", [True, False]),
                'use_att': trial.suggest_categorical("use_att", [True, False]),
                'use_lstm': trial.suggest_categorical("use_lstm", [True, False]),
                'use_cnn': trial.suggest_categorical("use_cnn", [True, False]),
                'verbose': trial.suggest_categorical("verbose", [True, False]),
                'random_state': trial.suggest_int("random_state", 0, 1000)
            },'KNeighborsTimeSeriesRegressor':{
                'n_neighbors': trial.suggest_int("n_neighbors", 1, 20),
                'weights': trial.suggest_categorical("weights", ["uniform", "distance"]),
                'algorithm': trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
                'distance': trial.suggest_categorical("distance",
                                                      ["euclidean", "squared", "dtw", "ddtw", "wdtw", "wddtw", "lcss",
                                                       "edr", "erp", "msm"]),
                'leaf_size': trial.suggest_int("leaf_size", 10, 100),
                'n_jobs': trial.suggest_categorical("n_jobs", [None, -1, 1, 2, 4, 8])
            },'CNNRegressor':{
                'n_epochs': trial.suggest_int("n_epochs", 100, 3000),
                'batch_size': trial.suggest_int("batch_size", 8, 64),
                'kernel_size': trial.suggest_int("kernel_size", 2, 10),
                'avg_pool_size': trial.suggest_int("avg_pool_size", 2, 10),
                'n_conv_layers': trial.suggest_int("n_conv_layers", 1, 5),
                'random_state': trial.suggest_int("random_state", 0, 100),
                'verbose': trial.suggest_categorical("verbose", [True, False]),
                'loss': trial.suggest_categorical("loss", ["mean_squared_error", "mean_absolute_error"]),
                'activation': trial.suggest_categorical("activation", ["linear", "relu", "sigmoid"]),
                'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"]),
                'use_bias': trial.suggest_categorical("use_bias", [True, False])
            },'RocketRegressor':{
                'num_kernels': trial.suggest_int("num_kernels", 5000, 10000, 15000, 20000),
                # Add any other relevant hyperparameters here
                'random_state': trial.suggest_int("random_state", 0, 100)
            },'TimeSeriesForestClassifier':{
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            },'KNeighborsTimeSeriesClassifier':
                                            {
                                                'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
                                                'weights': trial.suggest_categorical('weights',
                                                                                     ['uniform', 'distance']),
                                                'algorithm': trial.suggest_categorical('algorithm', ['brute', 'auto']),
                                                'distance': trial.suggest_categorical('distance',
                                                                                      ['dtw', 'ddtw', 'wdtw', 'lcss'])
                                            },'ShapeletTransformClassifier':{
                'max_shapelet_length': trial.suggest_int('max_shapelet_length', 3, 10)
            },'TimeSeriesSVMClassifier':{
                'C': trial.suggest_float('C', 0.1, 10.0),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 5)
            },'RandomIntervalSpectralForest': {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_interval': trial.suggest_int('min_interval', 1, 10),
                'max_interval': trial.suggest_int('max_interval', 1, 10)
            },'RocketClassifier':{
                'num_kernels': trial.suggest_int('num_kernels', 1000, 10000)
            },'MrSEQLClassifier':{
                'symrep': trial.suggest_categorical('symrep', ['sax', 'sfa']),
                'seql_mode': trial.suggest_categorical('seql_mode', ['clf', 'fs'])
            },'WEASEL':{
                'anova': trial.suggest_categorical("anova", [True, False]),
                'bigrams': trial.suggest_categorical("bigrams", [True, False]),
                'binning_strategy': trial.suggest_categorical("binning_strategy",
                                                              ["equi-depth", "equi-width", "information-gain"]),
                'window_inc': trial.suggest_int("window_inc", 1, 10),
                'p_threshold': trial.suggest_float("p_threshold", 0.01, 0.1),
                'alphabet_size': trial.suggest_int("alphabet_size", 2, 10),
                'feature_selection': trial.suggest_categorical("feature_selection", ["chi2", "none", "random"]),
                'support_probabilities': trial.suggest_categorical("support_probabilities", [True]),
                'random_state': trial.suggest_int("random_state", 0, 100)
            },'MUSE':{
                'anova': trial.suggest_categorical("anova", [True, False]),
                'variance': trial.suggest_categorical("variance", [True, False]),
                'bigrams': trial.suggest_categorical("bigrams", [True, False]),
                'window_inc': trial.suggest_int("window_inc", 1, 10),
                'alphabet_size': trial.suggest_int("alphabet_size", 2, 10),
                'p_threshold': trial.suggest_float("p_threshold", 0.01, 0.1),
                'use_first_order_differences': trial.suggest_categorical("use_first_order_differences", [True, False]),
                'support_probabilities': trial.suggest_categorical("support_probabilities", [True]),
                'feature_selection': trial.suggest_categorical("feature_selection", ["chi2", "none", "random"]),
                'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]),
                # 1 for single-threaded or -1 for using all processors
                'random_state': trial.suggest_int("random_state", 0, 100)
            },'Arsenal':{
                'num_kernels': trial.suggest_int("num_kernels", 100, 5000),
                'n_estimators': trial.suggest_int("n_estimators", 10, 100),
                'rocket_transform': trial.suggest_categorical("rocket_transform",
                                                              ["rocket", "minirocket", "multirocket"]),
                'max_dilations_per_kernel': trial.suggest_int("max_dilations_per_kernel", 10, 100),
                'n_features_per_kernel': trial.suggest_int("n_features_per_kernel", 1, 10),
                'time_limit_in_minutes': trial.suggest_float("time_limit_in_minutes", 0.0, 60.0),
                'contract_max_n_estimators': trial.suggest_int("contract_max_n_estimators", 10, 500),
                'save_transformed_data': trial.suggest_categorical("save_transformed_data", [True, False]),
                'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]),
                # 1 for single-threaded or -1 for using all processors
                'random_state': trial.suggest_int("random_state", 0, 100)
            },'IndividualBOSS':{
                'window_size': trial.suggest_int("window_size", 5, 20),
                'word_length': trial.suggest_int("word_length", 3, 16),
                'norm': trial.suggest_categorical("norm", [True, False]),
                'alphabet_size': trial.suggest_int("alphabet_size", 2, 10),
                'save_words': trial.suggest_categorical("save_words", [True, False]),
                'use_boss_distance': trial.suggest_categorical("use_boss_distance", [True, False]),
                'feature_selection': trial.suggest_categorical("feature_selection", ['none', 'entropy', 'gini']),
                'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]),
                # 1 for single-threaded or -1 for using all processors
                'random_state': trial.suggest_int("random_state", 0, 100)
            },'BOSSEnsemble':{
                'threshold': trial.suggest_float("threshold", 0.8, 1.0),
                'max_ensemble_size': trial.suggest_int("max_ensemble_size", 100, 1000),
                'max_win_len_prop': trial.suggest_float("max_win_len_prop", 0.5, 1.0),
                'min_window': trial.suggest_int("min_window", 5, 20),
                'save_train_predictions': trial.suggest_categorical("save_train_predictions", [True, False]),
                'alphabet_size': trial.suggest_int("alphabet_size", 2, 6),
                'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]),
                # 1 for single-threaded or -1 for using all processors
                'use_boss_distance': trial.suggest_categorical("use_boss_distance", [True, False]),
                'feature_selection': trial.suggest_categorical("feature_selection", ["chi2", "none", "random"]),
                'random_state': trial.suggest_int("random_state", 0, 100)},'ContractableBOSS':{
                'n_parameter_samples': trial.suggest_int("n_parameter_samples", 100, 500),
                'max_ensemble_size': trial.suggest_int("max_ensemble_size", 10, 100),
                'max_win_len_prop': trial.suggest_float("max_win_len_prop", 0.1, 1.0),
                'time_limit_in_minutes': trial.suggest_float("time_limit_in_minutes", 0.0, 60.0),
                'contract_max_n_parameter_samples': trial.suggest_int("contract_max_n_parameter_samples", 100, 1000),
                'save_train_predictions': trial.suggest_categorical("save_train_predictions", [True, False]),
                'feature_selection': trial.suggest_categorical("feature_selection", ["chi2", "none", "random"]),
                'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]),
                # 1 for single-threaded or -1 for using all processors
                'random_state': trial.suggest_int("random_state", 0, 100)
            },'CNNClassifier':{
                'n_epochs': trial.suggest_int("n_epochs", 100, 2000),  # 100 to 2000
                'batch_size': trial.suggest_int("batch_size", 8, 64),  # 8 to 64

                'n_conv_layers': trial.suggest_int("n_conv_layers", 1, 5),  # 1 to 5
                'random_state': trial.suggest_int("random_state", 0, 100),  # 0 to 100
                'verbose': trial.suggest_categorical("verbose", [True, False]),
                'loss': trial.suggest_categorical("loss", ["mean_squared_error", "categorical_crossentropy",
                                                           "binary_crossentropy"]),
                'metrics': trial.suggest_categorical("metrics", [["accuracy"], ["precision", "recall"]]),
                'activation': trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"]),
                'use_bias': trial.suggest_categorical("use_bias", [True, False]),
                'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])},'RotationForest':
                                            {
                                                'n_estimators': trial.suggest_int("n_estimators", 100, 500),
                                                # 100 to 500
                                                'min_group': trial.suggest_int("min_group", 1, 5),  # 1 to 5
                                                'max_group': trial.suggest_int("max_group", 3, 10),  # 3 to 10
                                                'remove_proportion': trial.suggest_float("remove_proportion", 0.1, 1.0),
                                                # 0.1 to 1.0
                                                'time_limit_in_minutes': trial.suggest_float("time_limit_in_minutes",
                                                                                             0.0, 60.0),
                                                # 0.0 to 60.0 minutes
                                                'contract_max_n_estimators': trial.suggest_int(
                                                    "contract_max_n_estimators", 100, 1000),  # 100 to 1000
                                                'save_transformed_data': trial.suggest_categorical(
                                                    "save_transformed_data", [True, False]),
                                                'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]),
                                                # 1 for single-threaded or -1 for using all processors
                                                'random_state': trial.suggest_int("random_state", 0, 100)  # 0 to 100
                                            },'FCNClassifier': {
                'n_epochs': trial.suggest_categorical('n_epochs', [2000]),
                # Since default is 2000, you might want to add more options
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                # Examples of different batch sizes
                'random_state': trial.suggest_categorical('random_state', [None, 42, 2023]),
                # None or specific random states
                'verbose': trial.suggest_categorical('verbose', [False, True]),
                'loss': trial.suggest_categorical('loss', ['categorical_crossentropy', 'mean_squared_error',
                                                           'binary_crossentropy']),  # Add more losses as needed
                'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop']),
                # You might need to define learning rates separately
                'metrics': trial.suggest_categorical('metrics', [['accuracy'], ['mse']]),
                # Add more metric combinations as needed
                'activation': trial.suggest_categorical('activation', ['sigmoid', 'relu', 'tanh']),
                # Other activation functions
                'use_bias': trial.suggest_categorical('use_bias', [True, False])
                # If learning rate needs to be optimized, it should be separate for each optimizer type
                # 'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
            }}




    def checked_in_models(self,pred_med:Literal['Classification','Regression'])->List[str]:
        return self.configs['models'][pred_med].keys()
    @property
    def checked_in_transforms(self)->List[str]:
        return self.configs['transforms'].keys()
    @property
    def possible_transforms(self) -> List[str]:
        return self.configs['transforms'].keys()
    @property
    def possible_imputation(self) ->List[str]:
        return self.configs['imputers'].keys()
    @staticmethod
    def _val_task(task:Literal['TAB','TS'])-> str:
        if not(  task == 'TAB' or task == 'TS'):
            raise ValueError('expecting either TAB for Tabular task or TS for Timeseries task')
        else :
            return task
    @staticmethod
    def TS_check(obj:pd.DataFrame)-> bool:
        if  obj.index.inferred_type == 'datetime64':
            return True
        else:
            return False

    @staticmethod
    def _classif_type(obj:pd.DataFrame) -> Literal['binary','multiclass']:
        if not isinstance(obj, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if obj.shape[1] != 1:
            raise ValueError("DataFrame must have only one column.")

        unique_values = obj.iloc[:, 0].unique()

        if len(unique_values) == 2:
            return 'binary'
        elif len(unique_values) > 2:
            return 'multiclass'
        else:
            raise ValueError("DataFrame must have at least two unique values for classification.")

    @staticmethod
    def _validate_obj(obj:pd.DataFrame,task=None)-> bool:
        assert isinstance(obj, pd.DataFrame), "Invalid"
        if task=='TS':
            assert obj.index.inferred_type == 'datetime64', "must have a datetime index for Timeseries Mode"
    @staticmethod
    def _validate_3d(obj:pd.DataFrame) -> bool:
        valid=check_is_scitype(
            obj, scitype="Panel")
        if valid:
            return True
        else:
            return False
    @staticmethod
    def _is_df(obj:Union[pd.DataFrame,np.ndarray,pd.Series],
               prefix:str="trans",
               idx:Optional[List[Any]]=None) -> pd.DataFrame:
        if isinstance(obj,pd.DataFrame):
            return obj
        elif isinstance(obj,np.ndarray) or isinstance(obj,pd.Series):
            return pd.DataFrame(obj,index=idx).add_prefix(prefix)
        else:
            raise AttributeError("Nno convertible type needsto be Series ndarray or Dataframe")
    @staticmethod
    def _validate_categorical(obj:pd.DataFrame) -> bool:
        cat_cols = obj.select_dtypes(exclude=['float', 'integer']).columns
        if len(cat_cols) == 0:
            return False
        else:
            return True
    @staticmethod
    def _validate_null(obj:pd.DataFrame,is_3d:bool) -> bool:
        if is_3d:
            res=[]
            for col in obj.columns:
                for j in range(obj.loc[:, col].size):
                    res.append(obj.loc[:, col].iloc[j].isnull().any())
            return np.any(res)
        else:

            return obj.isnull().any().any()
    @staticmethod
    def eval_df(obj:pd.DataFrame) -> bool:
        assert isinstance(obj, pd.DataFrame), 'invalid needs to be DataFrame'
        assert obj.isnull().any().any() == False, 'no null/NA values allowed'
        return obj
    @staticmethod
    def is_2d(obj:pd.DataFrame) -> bool:
        if isinstance(obj.iat[0, 0], float) or isinstance(obj.iat[0, 0], int):
            return True
        else:
            False
    @staticmethod
    def to_panel(obj:pd.DataFrame, window_size:Optional[int]=None) -> pd.DataFrame:
        if window_size is None or obj.shape[0] < window_size:
            raise AttributeError("window can not be None or bigger than df.shape[0]")

        # obtain numeric columns

        numeric_cols = obj.select_dtypes(include=['float', 'integer']).columns.tolist()

        if len(numeric_cols) == 0:
            raise AttributeError('expecting numeric columns')

        transformer = TimeSeriesToPanelData(window_size=window_size)

        res = list(map(transformer.transform, [obj.loc[:, col] for col in numeric_cols]))

        return pd.concat(res, axis=1)
    @staticmethod
    def _class_or_reg(obj:pd.DataFrame)->Literal['Regression','Classification']:
            if ('float' in str(obj.apply(pd.to_numeric, downcast='integer').iloc[:, 0].dtypes))\
                    or  obj.apply(pd.to_numeric, downcast='integer').iloc[:, 0].nunique()>10:
                return 'Regression'
            else:
                return 'Classification'
    @staticmethod
    def _empty_dict(dic:Dict[Any,Any])->bool:

        if dic=={}:
            return True
        else:
            False

    def _define_cv(self,is_ts:bool)->Union[KFold,TimeSeriesSplit]:
        if not is_ts:
            cv = KFold(n_splits=self.configs['n_cvs'], shuffle=True, random_state=42)
        else:
            cv = TimeSeriesSplit(n_splits=self.configs['n_cvs'])
        return cv
    @staticmethod
    def _unpack_results(results :List[Dict[str,Any]])->Dict[str,List[Any]]:

        if not ((isinstance(results,list)) and  len(results)>0):
            raise AttributeError('Must be a list and can not be empty')
        collect = defaultdict(list)
        for idx,res in enumerate(results):
            idx=str(idx)
            obj = pd.DataFrame(res['metrics']).assign(CV=lambda df: np.arange(df.shape[0]) + 1)
            obj_2 = pd.concat([pd.DataFrame(res['processing'], index=[i]) for i in range(obj.shape[0])])
            cmb = pd.concat([obj, obj_2], axis=1)
            cmb=cmb.assign(features_selections=res['processing']['features_selection']['method'])

            collect['metrics_model'].append(cmb.assign(ID=idx,Rank=None,Tuned=False).copy())
            feat_met=res['processing']['features_selection']['feat_metrics']
            if not isinstance(feat_met,bool):
                collect['metrics_features'].append(pd.DataFrame(feat_met).assign(ID=idx))
            collect[f'X_{idx}'].append(res['X'])
            collect[f'y_{idx}'].append(res['y'])

        return collect
    @staticmethod
    def _check_none_negative(obj:pd.DataFrame)->bool:
        return obj.gt(0).all().all()
    def get_models_available(self,is_ts:bool,pred_med:str)->List[Any]:
        if not isinstance(is_ts,bool):
            raise AttributeError(" is_ts must be boolean either True or False")
        if pred_med!='Classification' and pred_med!='Regression':
            raise ValueError("specify prediction method either ==> Classification or Regression")

        return [ k for k in self.configs['models'][pred_med].keys() if is_ts== self.configs['models'][pred_med][k]['ts_only']]
    def get_transforms_available(self,is_ts:bool,pred_med:Literal['Classification','Regression'])->List[Any]:
        if not isinstance(is_ts,bool):
            raise AttributeError(f" is_ts must be boolean either True or False not {type(is_ts)}")
        if pred_med!='Classification' and pred_med!='Regression':
            raise ValueError(f"specify prediction method either ==> 'Classification' or 'Regression' not {pred_med}")
        if is_ts:
            return [ k for k in self.configs['transforms'].keys() ]
        else:
            return [k for k in self.configs['transforms'].keys() if not self.configs['transforms'][k]['ts_only']]
    def get_feat_selections_available(self,pred_med:Literal['Classification','Regression'])->List[Any]:
        return self.configs['feat_selections'][pred_med]
    def get_dim_reductions_available(self,pred_med:Literal['Classification','Regression']):
        dim_reducts=list(self.configs['dim_reduction'].keys())
        if pred_med!='Classification':
            dim_reducts.remove('LDA')
            return dim_reducts
        else:
            return dim_reducts


    def set_X_y(self,X:Optional[pd.DataFrame]=None,y:Optional[pd.DataFrame]=None):
        if not  y is  None:
            self.y=y
        if not X is None:
            self.X=X


