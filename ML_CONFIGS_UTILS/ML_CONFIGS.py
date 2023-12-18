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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.metrics import f1_score, precision_score,log_loss,accuracy_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression,LogisticRegression,ElasticNet,Ridge
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor,ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor,MLPClassifier
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
from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sktime.regression.deep_learning import CNNRegressor,TapNetRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.compose import DirectTabularRegressionForecaster
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from CUSTOM_MODELS.CUSTOM_MODELS import  UniToMultivariateWrapper,TimeSeriesToPanelData
from CUSTOM_TRANSFORMS.CUSTOM_TRANSFORMS import CustomDWTTransformer,CustomPartialAutoCorrelationTransformer,CustomAutoCorrelationTransformer
import time
import threading
from sktime.datatypes import check_is_scitype
clf = ElasticEnsemble(
    proportion_of_param_options=0.1,
    proportion_train_for_test=0.1,
    distance_measures = ["dtw","ddtw"],
    majority_vote=True,
)
def unpack_args(func):
    from functools import wraps
    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper

class MultiScorer():
	'''
	Use this class to encapsulate and/or aggregate multiple scoring functions so that it can be passed as an argument for scoring in scikit's cross_val_score function.
	Instances of this class are also callables, with signature as needed by `cross_val_score`.
	'''

	def __init__(self, metrics):
		'''
		Create a new instance of MultiScorer.


		Parameters
		----------
		metrics: dict
			The metrics to be used by the scorer.
			The dictionary must have as key a name (str) for the metric and as value a tuple containing the metric function itself and a dict literal of the additional named arguments to be passed to the function.
			The metric function should be one of the `sklearn.metrics` function or any other callable with the same signature: `metric(y_real, y, **kwargs)`.
		'''

		self.metrics = metrics
		self.results = {}
		self._called = False
		self.n_folds = 0

		for metric in metrics.keys():
			self.results[metric] = []



	def __call__(self, estimator, X, y):
		'''
		To be called by for evaluation from sklearn's GridSearchCV or cross_val_score.
		Parameters are as they are defined in the respective documentation.

		Returns
		-------
			A dummy value of 0.5 just for compatibility reasons.
		'''

		self.n_folds += 1
		yPred = estimator.predict(X)

		for key in self.metrics.keys():
			metric, kwargs = self.metrics[key]

			self.results[key].append(metric(y, yPred, **kwargs))

		self._called = True

		return 0.5

	def get_metric_names(self):
		'''
		Get all the metric names as given when initialized

		Returns
		-------
		A list containing the given names (str) of the metrics
		'''

		return self.metrics.keys()

	def get_results(self, metric=None, fold='all'):
		'''
		Get the results of a specific or all the metrics.
		This method should be called after the object itself has been called so that the metrics are applied.

		Parameters
		----------
		metric: str or None (default)
			The given name of a metric to return its result(s). If omitted the results of all metrics will be returned.

		fold: int in range [1, number_of_folds] or 'all' (Default)
		 	Get the metric(s) results for the specific fold.
			The number of folds corresponds to the number of times the instance is called.
			If its value is a number, either the score of a single metric for that fold or a dictionary of the (single) scores for that fold will be returned, depending on the value of `metric` parameter.
			If its value is 'all', either a list of a single metric or a dictionary containing the lists of scores for all folds will be returned, depending on the value of `metric` parameter.

		Returns
		-------
		metric_result_for_one_fold
			The result of the designated metric function for the specific fold, if `metric` parameter was not omitted and an integer value was given to `fold` parameter.
			If  the value of `metric` does not correspond to a metric name, `None` will be returned.

		all_metric_results_for_one_fold: dict
			A dict having as keys the names of the metrics and as values their results for the specific fold.
			This will be returned only if `metric` parameter was omitted and an integer value was given to `fold` parameter.

		metric_results_for_all_folds: list
			A list of length number_of_folds containing the results of all folds for the specific metric, if `metric` parameter was not omitted and value 'all' was given to `fold`.
			If  the value of `metric` does not correspond to a metric name, `None` will be returned.

		all_metric_results_for_all_folds: dict of lists
			A dict having as keys the names of the metrics and as values lists (of length number_of_folds) of their results for all folds.
			This will be returned only if `metric` parameter was omitted and 'all' value was given to `fold` parameter.

		Raises
		------
		UserWarning
			If this method is called before the instance is called for evaluation.

		ValueError
			If the value for `fold` parameter is not appropriate.
		'''

		if not self._called:
			raise UserWarning('Evaluation has not been performed yet.')


		if isinstance(fold, str) and fold == 'all':

			if metric is None:
				return self.results
			else:
				return self.results[metric]

		elif isinstance(fold, int):

			if fold not in range(1, self.n_folds+1): raise ValueError('Invalid fold index: '+str(fold))

			if metric is None:
				res = dict()

				for key in self.results.keys():
					res[key] = self.results[key][fold-1]

				return res

			else:
				return self.results[metric][fold-1]
		else:
			raise ValueError('Unexpected fold value: %s' %(str(fold)))

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
                                                          'default_kwargs': {'lags':[1,3,5,10]}},
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
                                                          'default_kwargs': {}}}

        self.configs['is_sk_transform']=['RobustScaler', 'MinMaxScaler', 'KBinsDiscretizer', 'QuantileTransformer',
                           'StandardScaler', 'PowerTransformer','KernelCenterer']

        self.configs['imputers']={'KNNImputer': KNNImputer,
                    'SimpleImputer': SimpleImputer}

        self.configs['feat_selections']={'Regression':['correlation', 'f_regression', 'DecisionTreeRegressor', 'RandomForestRegressor', 'mutual_info_regression', 'LassoCV','all'],
                              'Classification':['permutation importance', 'f_classif', 'chi2', 'DecisionTreeClassifier', 'RandomForestClassifier', 'mutual_info_classif', 'LogisticRegressionCV','all']}

        self.configs['dim_reduction']=['method']

        self.configs['metrics']={'tab':{'Regression':{'mean_squared_error':(mean_squared_error,{}),
                                               'mean_absolute_error':(mean_absolute_error,{}),
                                               'r2_score':(r2_score,{}),'explained_variance_score':(explained_variance_score,{}),
                                               'mean_absolute_percentage_error':(mean_absolute_percentage_error,{})},
                                 'Classification':{'log_loss':(log_loss,{}),'accuracy':(accuracy_score,{}),
                                                               'f1':(f1_score,{}),'auc':(roc_auc_score,{}),'percision':(precision_score,{})}},
                                'ts' :{'Regression': {'mean_squared_error': (MedianSquaredError, {}),
                                                 'mean_absolute_error': (MeanAbsoluteError, {}),
                                                 'mean_absolute_percentage_error': (
                                                 MeanAbsolutePercentageError, {})},
                                  'Classification': {'log_loss': (log_loss, {}), 'accuracy': (accuracy_score, {}),
                                                     'f1': (f1_score, {}), 'auc': (roc_auc_score, {}),
                                                     'percision': (precision_score, {})}}}

        self.configs['models']={'Regression':{'LinearRegression':{'object':LinearRegression,'ts_only':False,'req_3d':False,'is_sklearn':True,'default_kwargs':{}},
                                              'ElasticNet':{'object':ElasticNet,'ts_only':False,'req_3d':False,'is_sklearn':True,'default_kwargs':{}},
                                              'Ridge':{'object':Ridge,'ts_only':False,'req_3d':False,'is_sklearn':True,'default_kwargs':{}},
                                              'XGBRegressor':{'object':XGBRegressor,'ts_only':False,'req_3d':False,'is_sklearn':True,'default_kwargs':{}},
                                              'CatBoostRegressor': {'object': CatBoostRegressor, 'ts_only': False,
                                                               'req_3d': False, 'is_sklearn': True,
                                                               'default_kwargs': {}},
                                              'LGBMRegressor': {'object': LGBMRegressor, 'ts_only': False,
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
                                                               'req_3d': False, 'is_sklearn': False,
                                                               'default_kwargs': {}},
                                              'KNeighborsTimeSeriesRegressor': {'object': KNeighborsTimeSeriesRegressor,
                                                                  'ts_only':True,
                                                                  'req_3d': True, 'is_sklearn': False,
                                                                  'default_kwargs': {}},
                                              'CNNRegressor': {'object': CNNRegressor,
                                                                                'ts_only': True,
                                                                                'req_3d': False, 'is_sklearn': False,
                                                                                'default_kwargs': {}},
                                              'TapNetRegressor': {'object': TapNetRegressor,
                                                               'ts_only': True,
                                                               'req_3d': False, 'is_sklearn': False,
                                                               'default_kwargs': {}},
                                              },
                                'Classification':{'LogisticRegression':{'object':LogisticRegression,'ts_only':False,'req_3d':False,'is_sklearn':True,'default_kwargs':{}},
                                                'XGBClassifier':{'object':XGBClassifier,'ts_only':False,'req_3d':False,'is_sklearn':True,'default_kwargs':{}},
                                'CatBoostClassifier': {'object': CatBoostClassifier, 'ts_only': False, 'req_3d': False,'is_sklearn': True, 'default_kwargs': {}},
                                'LGBMClassifier': {'object': LGBMClassifier, 'ts_only': False, 'req_3d': False,
                                                  'is_sklearn': True, 'default_kwargs': {}},
                                'SVC': {'object': SVC, 'ts_only': False, 'req_3d': False,
                                                  'is_sklearn': True, 'default_kwargs': {}},
                                'RandomForestClassifier': {'object': RandomForestClassifier, 'ts_only': False, 'req_3d': False,
                                        'is_sklearn': True, 'default_kwargs': {}},
                                'DecisionTreeClassifier': {'object': DecisionTreeClassifier, 'ts_only': False,
                                                           'req_3d': False,
                                                           'is_sklearn': True, 'default_kwargs': {}},
                                'AdaBoostClassifier': {'object': AdaBoostClassifier, 'ts_only': False,
                                                       'req_3d': False,
                                                       'is_sklearn': True, 'default_kwargs': {}},
                                'MLPClassifier': {'object': MLPClassifier,
                                                 'ts_only': False,
                                                 'req_3d': False, 'is_sklearn': True,
                                                 'default_kwargs': {}},
                                'CNNClassifier': {'object': CNNClassifier,
                                                      'ts_only': True,
                                                      'req_3d': True, 'is_sklearn': False,
                                                      'default_kwargs': {}},
                                'FCNClassifier': {'object': FCNClassifier,
                                                  'ts_only': True,
                                                  'req_3d': True, 'is_sklearn': False,
                                                  'default_kwargs': {}},
                                'KNeighborsTimeSeriesClassifier': {'object': KNeighborsTimeSeriesClassifier,
                                                 'ts_only': True,
                                                 'req_3d': True, 'is_sklearn': False,
                                                 'default_kwargs': {}},
                                'HIVECOTEV2': {'object': HIVECOTEV2,
                                               'ts_only': True,
                                               'req_3d': True, 'is_sklearn': False,
                                               'default_kwargs': {}},
                                                  'HIVECOTEV1': {'object': UniToMultivariateWrapper(HIVECOTEV1),
                                                                 'ts_only': True,
                                                                 'req_3d': True, 'is_sklearn': False,
                                                                 'default_kwargs': {}},
                                                  'MUSE': {'object': MUSE,
                                                                 'ts_only': True,
                                                                 'req_3d': True, 'is_sklearn': False,
                                                                 'default_kwargs': {}},
                                                  'ContractableBOSS': {'object': UniToMultivariateWrapper(ContractableBOSS),
                                                           'ts_only': True,
                                                           'req_3d': True, 'is_sklearn': False,
                                                           'default_kwargs': {}},
                                'RocketClassifier': {'object': RocketClassifier,
                                               'ts_only': True,
                                               'req_3d': True, 'is_sklearn': False,
                                               'default_kwargs': {}},
                                                  'ElasticEnsemble': {'object': UniToMultivariateWrapper(ElasticEnsemble),
                                                                       'ts_only': True,
                                                                       'req_3d': True, 'is_sklearn': False,
                                                                       'default_kwargs': {'proportion_of_param_options':0.1,
                                                                                          'proportion_train_for_test':0.1,
                                                                                          'distance_measures':['dtw','ddtw'],
                                                                                          'majority_vote':True}},
                                'TimeSeriesSVC': {'object': TimeSeriesSVC,
                                               'ts_only': True,
                                               'req_3d':True, 'is_sklearn': False,
                                               'default_kwargs': {}},
                                'ShapeletTransformClassifier': {'object': UniToMultivariateWrapper(ShapeletTransformClassifier),
                                               'ts_only': True,
                                               'req_3d': True, 'is_sklearn': False,
                                               'default_kwargs': {}},
                                                  'BOSSEnsemble': {
                                                      'object': UniToMultivariateWrapper(BOSSEnsemble),
                                                      'ts_only': True,
                                                      'req_3d': True, 'is_sklearn': False,
                                                      'default_kwargs': {}},
                                                  'IndividualBOSS': {
                                                      'object': UniToMultivariateWrapper(IndividualBOSS),
                                                      'ts_only': True,
                                                      'req_3d': True, 'is_sklearn': False,
                                                      'default_kwargs': {}},
                            'WEASEL': {'object':UniToMultivariateWrapper(WEASEL),
                                                 'ts_only': True,
                                                'req_3d': True, 'is_sklearn': False,
                                                'default_kwargs': {}},
                             'TimeSeriesForestClassifier': {'object': TimeSeriesForestClassifier,
                                          'ts_only': True,
                                       'req_3d': True, 'is_sklearn': False,
                              'default_kwargs': {}},
                                'TimeSeriesForestClassifier': {'object':  UniToMultivariateWrapper(SupervisedTimeSeriesForest),
                                                             'ts_only': True,
                                                            'req_3d': True, 'is_sklearn': False,
                                                            'default_kwargs': {}},
                                'Arsenal': {'object':Arsenal,
                                                 'ts_only': True,
                                                'req_3d': True, 'is_sklearn': False,
                                            'default_kwargs': {}},
                               'RotationForest': {'object':RotationForest,
                                   'ts_only': True,
                              'req_3d': False, 'is_sklearn': True,
                              'default_kwargs': {}},
                                }}
        self.configs['n_cvs']=5

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
            False
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
    def _is_df(obj,prefix:str="trans") -> pd.DataFrame:
        if isinstance(obj,pd.DataFrame):
            return obj
        elif isinstance(obj,np.ndarray) or isinstance(obj,pd.Series):
            return pd.DataFrame(obj).add_prefix(prefix)
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
        collect = []
        for res in results:
            obj = pd.DataFrame(res['metrics']).assign(CV=lambda df: np.arange(df.shape[0]) + 1)
            obj_2 = pd.concat([pd.DataFrame(res['processing'], index=[i]) for i in range(obj.shape[0])])
            cmb = pd.concat([obj, obj_2], axis=1)
            collect.append(cmb)
        return pd.concat(collect)
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
            return [k for k in self.configs['transforms'].keys() if not is_ts == self.configs['transforms'][k]['ts_only']]
    def get_feat_selections_available(self,pred_med:str):
        return self.configs['feat_selections'][pred_med]
    def get_dim_reductions_available(self):
        return ['LDA','TSNE','PCA','SE','UMAP','LLE','MDS','ISOMAP']


    def set_X_y(self,X=None,y=None):
        if not  y is  None:
            self.y=y
        if not X is None:
            self.X=X


