import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,log_loss
from ML_CONFIGS_UTILS.ML_CONFIGS import Config_Utils
from functools import partial
from imblearn.over_sampling import SMOTE
from collections import defaultdict
from optuna.pruners import MedianPruner
from typing import Optional,Dict,List,Literal,Set,Tuple,Union,Callable,Any


class Ml_Tune(Config_Utils):
    '''
    class to steer hyperparameter optimisation
    '''
    def __init__(self,
                 res_dic:Dict[str,Any],
                 pred_method:Literal['classification','regression'],
                 classif_type:Literal['binary','multiclass'],
                 is_ts:bool,
                 k_best:int=3,
                 *args,
                 **kwargs):
        '''

        :param res_dic: Dict[str,Any] containing the results of cross-validation
        :param pred_method: Literal['classification','regression'] prediction method
        :param classif_type: Literal['binary','multiclass'] if classification than this handles the sort of classification
        :param is_ts: bool flag informs about timeseries or not
        :param k_best: int integer that chooses K best estimators
        :param args:
        :param kwargs:
        '''
        super().__init__()
        self.k_best=k_best
        self.pred_method=pred_method
        self.classif_type=classif_type
        self.is_ts=is_ts
        self.res_dic=res_dic
        self.cv = self._define_cv(self.is_ts)
    def tune(self,*args,**kwargs)->Tuple[pd.DataFrame,Dict[str,Dict[str,Any]]]:
        '''

        :param args:
        :param kwargs:
        :return: Tuple[pd.DataFrame,Dict[str,Dict[str:Any]]] detailed results of hyperparameter tuning
        '''
        tune_results=[]
        tuned_ojects={}

        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)


        if self.pred_method=='Classification':
            dic={'log_loss':['mean']}
        else:
            dic={'mean_squared_error': ['mean']}


        model_metrics = pd.concat(self.res_dic['metrics_model'])
        dim=model_metrics.shape[0]
        k_best=int(min(dim/self.configs['n_cvs'],self.k_best))


        k_best_models=(model_metrics\
                            .groupby(['model','transform','dim_red','ID'])
                            .agg(dic)\
                            .droplevel(level=1,axis=1)\
                            .nsmallest(k_best,columns=['log_loss' if self.pred_method=='Classification' else 'mean_squared_error'])\
                            .sort_values(by=['log_loss' if self.pred_method=='Classification' else 'mean_squared_error'])
                            .reset_index())

        if self.pred_method=='Classification':
            target_metric='log_loss'
        else:
            target_metric='mean_squared_error'


        for num,(model, transform,dim_reducer,i_d) in enumerate(zip(k_best_models['model'].tolist(), k_best_models['transform'].tolist(),k_best_models['transform'].tolist(),k_best_models['ID'].tolist())):
            # Create a partial function that includes the extra_args and the trial object
            X=self.res_dic[f'X_{i_d}'][0]
            y=self.res_dic[f'y_{i_d}'][0]
            self.model=model
            partial_objective = partial(self.objective_function, self.model,X,y,target_metric)


            # Optimize using the partial function
            study = optuna.create_study(pruner=pruner, direction='maximize')
            study.optimize(lambda trial: self.optimize(partial_objective,trial), n_trials=2)

            model_obj,transformer_obj,dim_reducer_obj=self._define_classes(self.model,transform,dim_reducer,study)

            res_dic=self._cross_vals(model_obj, X, y)

            tune_results.append(self._process_dic(res_dic).assign(Rank=num,Tuned=True,dim_red=dim_reducer_obj,transform=transform,model=model))

            tuned_ojects[model+'_'+transform+'_'+i_d]={'model':model_obj,'transformer':transformer_obj,'dim_reducer':dim_reducer_obj,
                                                       'best_score':study.best_value,'best_params':study.best_params,'X':X,'y':y}

        return pd.concat(tune_results),tuned_ojects
    @staticmethod
    def _process_dic(res:Dict[str,List[float]])->pd.DataFrame:
        'helper function'
        return pd.DataFrame(res).assign(CV=lambda df: np.arange(df.shape[0]) + 1)
    def _define_classes(self,model:str,transformer:str,dim_reducer:str,params:Any)->Tuple[Any,Any,Any]:
        '''
        method to define set the best  hyperparameter retrieved of the tune

        :param model: stringrepresention of model
        :param transformer: stringrepresention of transformer
        :param dim_reducer: stringrepresention of dimensionality reduer
        :param params: Any[dict[str:Any]] containig all ideal hyperparameters
        :return: all the above objects
        '''

        model = self.configs['models'][self.pred_method][model]['object']
        transformer= self.configs['transforms'][transformer]['object']
        if not dim_reducer :
            dim_reducer=self.configs['dim_reduction'][dim_reducer]['object']
            dim_reducer = dim_reducer()
        else:
            dim_reducer=False

        model = model()
        transformer=transformer()

        for key, value in params.best_params.items():
            default_value = getattr(model, key, False)
            if not isinstance(default_value,bool):
                setattr(model, key, value)

        return model,transformer,dim_reducer

    def objective_function(self, model:Any,  X:pd.DataFrame, y:pd.DataFrame,target_metric:str, trial:Any)->float:
        '''
        objective function for optuna to perform optimization

        :param model: model object
        :param X: pd.DataFrame with training data
        :param y: pd.DataFrame with target data
        :param target_metric: str that holds metric to perform tune on
        :param trial: optuna hypers
        :return: float target metric
        '''
        params_model = self.hyper_parameter_register(model, trial)


        if not model in  self.configs['models'][self.pred_method]:
             raise KeyError("Model is not checked in ")


        model=self.configs['models'][self.pred_method][model]['object']


        model = model(**params_model)



        results=self._cross_vals(model,X,y)

        return -np.mean(results[target_metric])


    def optimize(self, partial_objective, trial):
        '''
        helper function to perform optuna optimization
        :param partial_objective:
        :param trial:
        :return:
        '''
        return partial_objective(trial)
    def _cross_vals(self,model:Any,X:pd.DataFrame,y:pd.DataFrame,handle_imbalance:Optional[bool] = True)->Dict[str,float]:
        '''

        :param model: model object
        :param X: pd.DataFrame
        :param y: pd.DataFrame
        :param handle_imbalance: bool True if imbalnce is handled
        :return: Dict[str,float] results of optimization metrics_name and value
        '''

        if self.is_ts:

            if self.configs['models'][self.pred_method][self.model]['req_3d'] and not self._validate_3d(X):
               X = self.to_panel(X, window_size=14)


            results = self._custom_evaluate(model=model,
                            y=y,
                            X=X,
                            cv=self.cv,
                            scoring=self._get_scorings())
        else:
            if handle_imbalance and self.pred_method == 'Classification':
                X, y = self._handle_imbalance(X,y)

            results = self._custom_evaluate(model=model,
                            y=y,
                            X=X,
                            cv=self.cv,
                            scoring=self._get_scorings())
        return results

    def hyper_parameter_register(self, key:str)->Dict[str,Any]:
        '''

        :param key: name of the model
        :param trial: optuna hyperparamterlist
        :return: Dict[str:Any] containing hyperpara searchspace
        '''

        if key in self.configs['hyperparamters']:
            return self.configs['hyperparamters'][key]
        else:
            KeyError(f"hyperparameter tuning is not available for {key}")
            # Note: The optimizer's learning rate (if variable) needs special handling depending on the chosen optimizer.

    @staticmethod
    def _custom_evaluate(model:Any, y:pd.DataFrame, X:pd.DataFrame, cv:Any=None, scoring:List[Callable]=None)->Dict[str,List[float]]:
        '''
        function to customly evaluate // hypertune

        :param model: model object
        :param y: pd.DataFrame target data
        :param X: pd.DataFrame trainings data
        :param cv: Any Cross Fold Split
        :param scoring: List[Callable]=None
        :return: Dict[str,List[float]] containing all metrics and corresponding crossfold results
        '''

        if cv is None or scoring is None:
            raise AttributeError("Please handover cv and scoring")

        results = defaultdict(list)

        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            model.fit(X.iloc[train_index], y.iloc[train_index,0])
            preds_total = model.predict(X.iloc[test_index])
            for metrics in scoring:
                name = metrics.__name__
                if name=='log_loss':
                    preds_proba = model.predict_proba(X.iloc[test_index])
                    res = metrics(y.iloc[test_index], preds_proba)
                else:
                    res = metrics(y.iloc[test_index], preds_total)
                results[name].append(res)

        return results
    @staticmethod
    def _handle_imbalance(X:pd.DataFrame,y:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
        '''
        account for imblancedness
        :param X: pd.DataFrame train data
        :param y: pd.DataFrame traget data
        :return: Tuple[pd.DataFrame,pd.DataFrame] updated trainingsdata
        '''

        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X,y)
        return X,y
    def _get_scorings(self)->List[Callable]:
        ''' return >List[Callable] all problem specific metrics functions'''
        if self.is_ts:
            if self.pred_method == 'Classification':
                if self.classif_type=='binary':
                    scoring=[val[0] for k, val in self.configs['metrics']['ts'][self.pred_method][self.classif_type].items()]
                else:
                    scoring = [val[0] for k, val in self.configs['metrics']['ts'][self.pred_method][self.classif_type].items()]
            else:
                scoring = [val[0] for k, val in self.configs['metrics']['ts'][self.pred_method].items()]

        else:
            if self.pred_method == 'Classification':
                if self.classif_type=='binary':
                    scoring = [ val[0] for k,val in self.configs['metrics']['tab'][self.pred_method][self.classif_type].items()]
                else:
                    scoring = [ val[0] for k,val in self.configs['metrics']['tab'][self.pred_method][self.classif_type].items()]
            else:
                scoring = [ val[0] for k,val in self.configs['metrics']['tab'][self.pred_method].items()]
        return scoring
