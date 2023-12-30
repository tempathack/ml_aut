from collections import defaultdict
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from ML_CONFIGS_UTILS.ML_CONFIGS import Config_Utils,MultiScorer,FunctionTimer
from LOGGER.LOGGING import WrapStack
from imblearn.over_sampling import SMOTE
from sktime.classification.base import BaseClassifier
from sktime.datatypes import check_is_scitype



class Models(Config_Utils):
    def __init__(self, model ,pred_method ,*args ,**kwargs):
        super(Models ,self).__init__()
        self.model = model
        self.args = args
        self.pred_method =pred_method
        if self._empty_dict(kwargs):
            self.kwargs =self.configs['models'][self.pred_method][self.model]['default_kwargs']


        else:
            self.kwargs = kwargs


    def get_model(self):
        if self.model in self.checked_in_models(self.pred_method):
            method =self.configs['models'][self.pred_method][self.model]['object']


            return method(*self.args, **self.kwargs)
        else:
            raise KeyError(f"{self.model} is not supported")


class Ml_Train(Config_Utils):
    def __init__(self, X, y, *args, **kwargs):
        super().__init__()
        self.X = X
        self.y = self.eval_df(y)
        self.args = args
        self.kwargs = kwargs
        self.is_ts = self.TS_check(X)
        self.pred_method = self._class_or_reg(y)
        if self.pred_method=='Classification':
            self.classif_type=self._classif_type(y)
        self.cv = self._define_cv(self.is_ts)
    def __repr__(self):
        return  f"Ml_Train()"
    @WrapStack.FUNCTION_SCREEN
    def train_model(self,model,timeout:int=30000, *args, **kwargs):

        if model is None or (not model in self.configs['models'][self.pred_method]):
            raise KeyError(f'model is either not specified or not part of {self.configs["models"][self.pred_method].keys()}')



        if self.is_ts:
            timer=FunctionTimer(func=self._train_ts, timeout=timeout)
            results = timer.run(model,*args, **kwargs)
        else:
            timer=FunctionTimer(func=self._train_tab, timeout=timeout)
            results = timer.run(model,*args, **kwargs)



        if results is None:
            scoring=self._get_scorings()
            func_names = [val.__name__ for k, val in scoring]
            results = {f_name: ['timed_out' for _ in range(self.configs['n_cvs'])] for f_name in func_names}

        return results

    def _train_ts(self,model:str,*args,**kwargs):
        if self.configs['models'][self.pred_method][model]['req_3d'] and not self._validate_3d(self.X):
            self.X = self.to_panel(self.X, window_size=14)
        model = Models(model, self.pred_method, *args, **kwargs).get_model()



        results = self._custom_evaluate(model=model,
                            y=self.y,
                            X=self.X,
                            cv=self.cv,
                            scoring=self._get_scorings())


        return results
    def _train_tab(self,model:str,handle_imbalance=True,*args,**kwargs):



        model = Models(model, self.pred_method, *args, **kwargs).get_model()

        if handle_imbalance and self.pred_method == 'Classification':
            X, y = self._handle_imbalance(self.X, self.y)

        if self.cv is None:
            raise ValueError("Please handover cv and scoring")

        results = self._custom_evaluate(model=model,
                                        y=self.y,
                                        X=self.X,
                                        cv=self.cv,
                                        scoring=self._get_scorings())

        return results
    def _get_scorings(self):
        if self.is_ts:
            if self.pred_method == 'Classification':
                if self.classif_type=='binary':
                    scoring=[val[0]() for k, val in self.configs['metrics']['ts'][self.pred_method][self.classif_type].items()]
                else:
                    scoring = [val[0]() for k, val in self.configs['metrics']['ts'][self.pred_method][self.classif_type].items()]
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
    @staticmethod
    def _handle_imbalance(X,y):
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X,y)
        return X,y
    @staticmethod
    def _custom_evaluate(model, y, X, cv=None, scoring=None):

        if cv is None or scoring is None:
            raise ValueError("Please handover cv and scoring")

        results = defaultdict(list)

        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            model.fit(X.iloc[train_index], y.iloc[train_index,0])
            preds_total = model.predict(X.iloc[test_index])
            preds_proba = model.predict_proba(X.iloc[test_index])
            for metrics in scoring:
                name = metrics.__name__
                if name=='log_loss':
                    res = metrics(y.iloc[test_index], preds_proba)
                else:
                    res = metrics(y.iloc[test_index], preds_total)
                results[name].append(res)

        return results
