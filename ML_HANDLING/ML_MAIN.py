import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from ML_CONFIGS_UTILS.ML_CONFIGS import  Config_Utils
from ML_FEATURE_SELECTION.ML_FEATURE_SELECTION import Ml_Select
from ML_TRAINING.ML_TRAINING import Ml_Train
from ML_TUNE.ML_TUNE import Ml_Tune
from ML_TRANSFORMING.ML_TRANSFORMING import Ml_Process
from ML_DIMENSIONALITY_REDUCTION.ML_DIMENSIONALITY_REDUCTION import Ml_Reduce
from ML_CONFIGS_UTILS.Custom_Errors import MethodNotExecutedError
from joblib import Parallel, delayed
from LOGGER.LOGGING import WrapStack
from multiprocessing import Manager,current_process,managers
from typing import Optional,Dict,List,Literal,Set,Tuple,Union,Callable,Any
class Ml_Main(Config_Utils):
    '''
    Main Class to handle all subclasses of config interchangable
    '''
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, transform:Union[str,List[str]],ml_model:Union[str,List[str]],
                 features_selection:Optional[str]=None,
                 dim_reduction:Union[str,List[str]]=None,
                 n_cvs:Optional[int]=None,
                 n_jobs:Optional[int]=1):
        '''

        :param X: :pd.DataFrame training data
        :param y: :pd.DataFrame traget data
        :param transform: Union[str,List[str]] transform string representation(s)
        :param ml_model: Union[str,List[str]] model string representation(s)
        :param features_selection: features selection string representation
        :param dim_reduction: Optional[str,List[str]] dimensionaliy reduction string representation(s)
        :param n_cvs: Optional[int] n_splits for crossfold validation
        :param n_jobs: Optional[int] amount of parallel jobs to run default ==> sequenctial validation
        '''

        super().__init__()
        if transform is None or ml_model is None:
            raise AttributeError("Handover transform as well as model argument")

        self.X = X
        self.y = self.eval_df(y)
        self.transform = transform if isinstance(transform, list) else [transform]
        self.features_selection = features_selection if features_selection is not None else False
        self.dim_reduction = dim_reduction if isinstance(dim_reduction, list) else [dim_reduction] if isinstance(dim_reduction,str) else [None]
        self.model = ml_model if isinstance(ml_model, list) else [ml_model]
        self.n_jobs=n_jobs
        self.ml_train = None
        self.pred_method=self._class_or_reg(self.y)
        if self.pred_method == 'Classification':
            self.classif_type = self._classif_type(y) 
        self.mode= 'seq' if n_jobs==1 else 'parallel'
        self.Logger=WrapStack()

        if n_cvs is not  None:
            self.configs['n_cvs']=n_cvs




        if isinstance(self.transform, list):
            self.ml_process = Ml_Process(X=self.X.copy())
            self.is_ts=getattr(self.ml_process,'is_ts')

        if isinstance(self.features_selection, str):
            self.ml_select = Ml_Select(X=self.X.copy(), y=self.y.copy())

        if self.dim_reduction[0] is not None:
            self.ml_reduce = Ml_Reduce(X=self.X.copy(), y=self.y.copy())

        self.ml_train = Ml_Train(X=self.X.copy(), y=self.y.copy())


    def Process(self, *args, **kwargs):
        '''

        :param args:
        :param kwargs:
        :return: self
        '''
        self.is_ml_select = hasattr(self, 'ml_select')



        if self.mode == 'parallel':
            results=self._process_parallel(self.is_ml_select)
        elif self.mode == 'seq':
            results=self._process_seq(self.is_ml_select, *args, **kwargs)



        self.unpacked_results=self._unpack_results(results)

        return self
    @WrapStack.FUNCTION_SCREEN
    def _process_parallel(self, is_ml_select:bool,*args, **kwargs)->List[Dict[str,Any]]:
        '''
        function to process parallel handle entire procedure
        :param is_ml_select: bool if feature selection is on
        :param args:
        :param kwargs:
        :return: List[Dict[str,Any]] list of all results
        '''
        # Initialize Ray (ideally done outside this method)
        manager=Manager()
        shared_dict = manager.dict()


        results = Parallel(n_jobs=self.n_jobs)(delayed(self._define_generator)
                                               (transform, model, is_ml_select, dim_reduction,shared_dict, *args, **kwargs)
                                               for dim_reduction in self.dim_reduction
                                                for model in self.model
                                                for transform in self.transform)

        return results
        # Shutdown Ray (ideally done outside this method)
    def __getattr__(self,key):

        if key=='Ml_Process':
            for trans in self.transforms:
                setattr(self.ml_process,'transform',trans)
                yield self.ml_process.main_transform(*self.args,**self.kwargs)
        elif key=='Ml_Select' and hasattr(self, 'ml_select'):
            setattr(self.ml_select,'method',self.features_selection)
            yield self.ml_select.feature_selection(*self.args,**self.kwargs)
        elif key=='Ml_Reduce' and hasattr(self,'ml_reduce'):
            for dim_red in self.dim_reduction:
                setattr(self.ml_reduce, 'method', dim_red)
                yield self.ml_reduce.dimensionality_reduction(*self.args,**self.kwargs)
        elif key=='Ml_Train':
            for model in self.model:
                setattr(self.ml_reduce, 'model', model)
                yield self.ml_train.train_model(*self.args,**self.kwargs)
        else:
            raise AttributeError(f"{key} is not an procedure object used in {self.__class__.__name__}")
    @WrapStack.FUNCTION_SCREEN
    def _process_seq(self, is_ml_select:bool, *args, **kwargs)->List[Dict[str,Any]]:
        '''
        function to handle procedure sequentially
        :param is_ml_select: boolean flag indicating whether feat select is needed
        :param args:
        :param kwargs:
        :return: ->List[Dict[str,Any]]
        '''
        results=[]
        for transform in self.transform:
            for dim_reduction in self.dim_reduction:
                for model in self.model:
                    result = self._define_generator(transform, model, is_ml_select,dim_reduction, *args, **kwargs)
                    results.append(result)
        return results
    def _define_generator(self, transform:str,
                          model:str,
                          is_ml_select:Optional[bool]=False,
                          dim_red:Optional[str]=None,
                          shared_dict:managers.DictProxy=None,
                          *args, **kwargs)->Dict[str,Any]:
        '''
        main handling function for ml procedure sequences

        :param transform: str string representation of transformer
        :param model: str string representation of model
        :param is_ml_select: Optional[bool] bool flg if feat select is needed
        :param dim_red: Optional[str] optional dimensionality reduciton
        :param shared_dict: managers.DictProxy Multiprocessing Shared Manager to avoid redundant transformations in parallel mode
        :param args:
        :param kwargs:
        :return: ->Dict[str,Any] results dictionary

        kwargs:
        k_feat:number of features to select based on feature selection
        upper_limit_dim:number of features to reduce via pca before dim reduction
        how_impute: str on how to impute
        '''

        if isinstance(transform,list):
            transform_cap = transform.copy()
            transform = '-'.join(transform)
            while len(transform_cap)>0 and isinstance(transform_cap ,list):
                X = self.ml_process.main_transform(transform=transform_cap[0], *args, **kwargs)
                transform_cap.pop(0)
        else:
            if isinstance(shared_dict,managers.DictProxy):
                if transform in shared_dict:
                    X=shared_dict[transform]
                else:
                    X = self.ml_process.main_transform(transform=transform, *args, **kwargs)
                    shared_dict[transform]=X
            else:
                X = self.ml_process.main_transform(transform=transform, *args, **kwargs)

        if is_ml_select and not self._validate_3d(X):
            self.ml_select.set_X_y(X=X)
            X = self.ml_select.feature_selection(method=self.features_selection, *args, **kwargs)

        if not dim_red is None and not self._validate_3d(X):
            self.ml_reduce.set_X_y(X=X)
            X = self.ml_reduce.dimensionality_reduction(method=dim_red, *args, **kwargs)


        self.ml_train.set_X_y(X=X)
        metrics = self.ml_train.train_model(model=model, *args, **kwargs)

        return {'processing': {'transform': transform, 'features_selection':
               {'method':self.features_selection,'feat_metrics':self.ml_select.feat_metrics()
               if is_ml_select else False},
                'dim_red':False if  dim_red is None else dim_red, 'model': model},
                'metrics': metrics,'X':X,'y':self.y}

    @WrapStack.FUNCTION_SCREEN
    def Tune(self,k_best:Optional[int]=3):
        '''
        hyperparameter tuning method

        :param k_best: Optional[int] number of best Estimators for hyperparameter tuning
        :return: self
        '''
        if not hasattr(self,'unpacked_results'):
            raise MethodNotExecutedError("please make sure to execute Process method beforehand")

        self.ml_tune=Ml_Tune(self.unpacked_results,
                             self.pred_method,
                             self.classif_type if hasattr(self, 'classif_type') else None,
                             self.is_ts,
                             k_best=k_best)
        self.tuned_results,self.tuned_objects=self.ml_tune.tune()


        return self
    def get_feature_selections(self)->pd.DataFrame:
        ''' return pd.DataFrame with feature selections results'''

        if  hasattr(self,'unpacked_results') and hasattr(self, 'ml_select'):
            return pd.concat(self.unpacked_results['metrics_features'])
        else:
            raise MethodNotExecutedError("please make sure to execute Process method beforehand and use some features_selection method")
    def get_model_metrics(self)->pd.DataFrame:
        ''' return pd.DataFrame with model metrics results'''

        if hasattr(self,'tuned_results'):
            return pd.concat([pd.concat(self.unpacked_results['metrics_model']),self.tuned_results]).drop(columns=['ID'])

        if  hasattr(self,'unpacked_results'):
            return pd.concat(self.unpacked_results['metrics_model'])
        else:
            raise MethodNotExecutedError("please make sure to execute Process method beforehand")
    def get_tuned_objects(self)->Dict[str,Any]:
        ''' return DICT with best params best predictor/hyperparameters'''
        if not hasattr(self,'tuned_results'):
            raise MethodNotExecutedError("please make sure to execute Tune method beforehand")
        else:
            return self.tuned_objects
    def Analyse(self):
        if not self.is_processed:
            raise MethodNotExecutedError("please make sure to atleast execute Process method beforehand")