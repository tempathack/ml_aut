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
                                                                 'default_kwargs': {}},
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
                                               'default_kwargs': {}},
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
                                                'default_kwargs': {}},
                             'TimeSeriesForestClassifier': {'object': TimeSeriesForestClassifier,
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

    def checked_in_models(self,pred_med):
        return self.configs['models'][pred_med].keys()
    @property
    def checked_in_transforms(self):
        return self.configs['transforms'].keys()
    @property
    def possible_transforms(self) -> dict:
        return self.configs['transforms'].keys()
    @property
    def possible_imputation(self) -> dict:
        return self.configs['imputers'].keys()
    @staticmethod
    def _val_task(task)-> str:
        if not(  task == 'TAB' or task == 'TS'):
            raise ValueError('expecting either TAB for Tabular task or TS for Timeseries task')
        else :
            return task
    @staticmethod
    def TS_check(obj)-> bool:
        if  obj.index.inferred_type == 'datetime64':
            return True
        else:
            return False

    @staticmethod
    def _classif_type(obj) -> str:
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
    def _validate_obj(obj,task=None)-> bool:
        assert isinstance(obj, pd.DataFrame), "Invalid"
        if task=='TS':
            assert obj.index.inferred_type == 'datetime64', "must have a datetime index for Timeseries Mode"
    @staticmethod
    def _validate_3d(obj) -> bool:
        valid=check_is_scitype(
            obj, scitype="Panel")
        if valid:
            return True
        else:
            return False
    @staticmethod
    def _is_df(obj,prefix:str="trans",idx=None) -> pd.DataFrame:
        if isinstance(obj,pd.DataFrame):
            return obj
        elif isinstance(obj,np.ndarray) or isinstance(obj,pd.Series):
            return pd.DataFrame(obj,index=idx).add_prefix(prefix)
        else:
            raise AttributeError("Nno convertible type needsto be Series ndarray or Dataframe")
    @staticmethod
    def _validate_categorical(obj) -> bool:
        cat_cols = obj.select_dtypes(exclude=['float', 'integer']).columns
        if len(cat_cols) == 0:
            return False
        else:
            return True
    @staticmethod
    def _validate_null(obj,is_3d) -> bool:
        if is_3d:
            res=[]
            for col in obj.columns:
                for j in range(obj.loc[:, col].size):
                    res.append(obj.loc[:, col].iloc[j].isnull().any())
            return np.any(res)
        else:

            return obj.isnull().any().any()
    @staticmethod
    def eval_df(obj) -> bool:
        assert isinstance(obj, pd.DataFrame), 'invalid needs to be DataFrame'
        assert obj.isnull().any().any() == False, 'no null/NA values allowed'
        return obj
    @staticmethod
    def is_2d(obj) -> bool:
        if isinstance(obj.iat[0, 0], float) or isinstance(obj.iat[0, 0], int):
            return True
        else:
            False
    @staticmethod
    def to_panel(obj, window_size=None) -> pd.DataFrame:
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
    def _class_or_reg(obj):
            if ('float' in str(obj.apply(pd.to_numeric, downcast='integer').iloc[:, 0].dtypes))\
                    or  obj.apply(pd.to_numeric, downcast='integer').iloc[:, 0].nunique()>10:
                return 'Regression'
            else:
                return 'Classification'
    @staticmethod
    def _empty_dict(dic):

        if dic=={}:
            return True
        else:
            False

    def _define_cv(self,is_ts):
        if not is_ts:
            cv = KFold(n_splits=self.configs['n_cvs'], shuffle=True, random_state=42)
        else:
            cv = TimeSeriesSplit(n_splits=self.configs['n_cvs'])
        return cv
    @staticmethod
    def _unpack_results(results : list ):

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
    def _check_none_negative(obj):
        return obj.gt(0).all().all()
    def get_models_available(self,is_ts:bool,pred_med:str):
        if not isinstance(is_ts,bool):
            raise AttributeError(" is_ts must be boolean either True or False")
        if pred_med!='Classification' and pred_med!='Regression':
            raise ValueError("specify prediction method either ==> Classification or Regression")

        return [ k for k in self.configs['models'][pred_med].keys() if is_ts== self.configs['models'][pred_med][k]['ts_only']]
    def get_transforms_available(self,is_ts:bool,pred_med:str):
        if not isinstance(is_ts,bool):
            raise AttributeError(f" is_ts must be boolean either True or False not {type(is_ts)}")
        if pred_med!='Classification' and pred_med!='Regression':
            raise ValueError(f"specify prediction method either ==> 'Classification' or 'Regression' not {pred_med}")
        if is_ts:
            return [ k for k in self.configs['transforms'].keys() ]
        else:
            return [k for k in self.configs['transforms'].keys() if not self.configs['transforms'][k]['ts_only']]
    def get_feat_selections_available(self,pred_med:str):
        return self.configs['feat_selections'][pred_med]
    def get_dim_reductions_available(self,pred_med:str):
        dim_reducts=list(self.configs['dim_reduction'].keys())
        if pred_med!='Classification':
            dim_reducts.remove('LDA')
            return dim_reducts
        else:
            return dim_reducts


    def set_X_y(self,X=None,y=None):
        if not  y is  None:
            self.y=y
        if not X is None:
            self.X=X


