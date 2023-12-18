from ML_CONFIGS_UTILS.ML_CONFIGS import  Config_Utils
from ML_FEATURE_SELECTION.ML_FEATURE_SELECTION import Ml_Select
from ML_TRAINING.ML_TRAINING import Ml_Train
from ML_TRANSFORMING.ML_TRANSFORMING import Ml_Process
from ML_DIMENSIONALITY_REDUCTION.ML_DIMENSIONALITY_REDUCTION import Ml_Reduce
from ML_CONFIGS_UTILS.Custom_Errors import MethodNotExecutedError
from joblib import Parallel, delayed
from LOGGER.LOGGING import WrapStack
from multiprocessing import Manager,current_process,managers


class Ml_Main(Config_Utils):
    def __init__(self, X, y, transform=None,
                 features_selection=None,
                 dim_reduction=None,
                 ml_model=None,
                 n_jobs=1):

        super().__init__()
        if transform is None or ml_model is None:
            raise AttributeError("Handover transform as well as model argument")

        self.X = self.eval_df(X)
        self.y = self.eval_df(y)
        self.transform = transform if isinstance(transform, list) else [transform]
        self.features_selection = features_selection if features_selection is not None else False
        self.dim_reduction = dim_reduction if isinstance(dim_reduction, str) else None
        self.model = ml_model if isinstance(ml_model, list) else [ml_model]
        self.n_jobs=n_jobs
        self.ml_train = None
        self.mode= 'seq' if n_jobs==1 else 'parallel'
        self.Logger=WrapStack()


        if isinstance(self.transform, list):
            self.ml_process = Ml_Process(X=self.X.copy())

        if isinstance(self.features_selection, str):
            self.ml_select = Ml_Select(X=self.X.copy(), y=self.y.copy())

        if isinstance(self.dim_reduction, str):
            self.ml_reduce = Ml_Reduce(X=self.X.copy(), y=self.y.copy())

        self.ml_train = Ml_Train(X=self.X.copy(), y=self.y.copy())


    def Process(self,results_return=False, *args, **kwargs):
        self.is_ml_select = hasattr(self, 'ml_select')
        self.is_ml_reduce = hasattr(self, 'ml_reduce')



        if self.mode == 'parallel':
            results=self._process_parallel(self.is_ml_select, self.is_ml_reduce)
        elif self.mode == 'seq':
            results=self._process_seq(self.is_ml_select, self.is_ml_reduce, *args, **kwargs)


        self.is_processed = True


        if results_return:
            return self._unpack_results(results)
        else:
            return self
    @WrapStack.FUNCTION_SCREEN
    def _process_parallel(self, is_ml_select, is_ml_reduce,*args, **kwargs):
        # Initialize Ray (ideally done outside this method)
        manager=Manager()
        shared_dict = manager.dict()


        results = Parallel(n_jobs=self.n_jobs)(delayed(self._define_generator)
                                               (transform, model, is_ml_select, is_ml_reduce, self.dim_reduction,shared_dict, *args, **kwargs)
                                                for model in self.model for transform in self.transform)

        return results
        # Shutdown Ray (ideally done outside this method)
    @WrapStack.FUNCTION_SCREEN
    def _process_seq(self, is_ml_select, is_ml_reduce, *args, **kwargs):
        results=[]
        for transform in self.transform:
                for model in self.model:
                    result = self._define_generator(transform, model, is_ml_select, is_ml_reduce, self.dim_reduction, *args, **kwargs)
                    results.append(result)
        return results
    def _define_generator(self, transform, model, is_ml_select=False, is_ml_reduce=False, dim_red=None,shared_dict=None, *args, **kwargs):

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

        if is_ml_reduce and not self._validate_3d(X):
            self.ml_reduce.set_X_y(X=X)
            X = self.ml_reduce.dimensionality_reduction(method=dim_red, *args, **kwargs)


        self.ml_train.set_X_y(X=X)
        metrics = self.ml_train.train_model(model=model, *args, **kwargs)

        return {'processing': {'transform': transform, 'features_selection': {'method':self.features_selection,'feat_metrics':self.ml_select.feat_metrics()
        if is_ml_select else False},
                               'dim_red': False if dim_red is None else dim_red, 'model': model},
                'metrics': metrics}
    def Tune(self,k_best=3,results_return=False):
        if not self.is_processed:
            raise MethodNotExecutedError("please make sure to execute Process method beforehand")

    def Analyse(self):
        if not self.is_processed:
            raise MethodNotExecutedError("please make sure to atleast execute Process method beforehand")