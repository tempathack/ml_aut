from collections import defaultdict
from sklearn.model_selection import cross_val_score



from ML_CONFIGS_UTILS.ML_CONFIGS import CONFIG_UTILS,MultiScorer




class MODELS(CONFIG_UTILS):
    def __init__(self, model ,pred_method ,*args ,**kwargs):
        super(MODELS ,self).__init__()
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
            raise ValueError(f"{self.model} is not supported")


class ML_TRAIN(CONFIG_UTILS):
    def __init__(self, X, y, *args, **kwargs):
        super().__init__()
        self.X = self.eval_df(X)
        self.y = self.eval_df(y)
        self.args = args
        self.kwargs = kwargs
        self.is_ts = self.TS_check(X)
        self.pred_method = self._class_or_reg(y)
        self.cv = self._define_cv(self.is_ts)

    def train_model(self, model=None, *args, **kwargs):

        if model is None:
            raise ValueError('model must be specified')

        if self.is_ts:
            if self.configs['models'][self.pred_method][model]['req_3d']:
                self.X = self.to_panel(self.X, window_size=12)
            model = MODELS(model, self.pred_method, *args, **kwargs).get_model()

            results = self._custom_evaluate(
                model=model,
                y=self.y,
                X=self.X,
                cv=self.cv,
                scoring=[val[0]() for k, val in self.configs['metrics']['ts'][self.pred_method].items()],
            )

            return results
        else:
            self.metrics_scorer = MultiScorer(self.configs['metrics']['tab'][self.pred_method])

            model = MODELS(model, self.pred_method, *args, **kwargs).get_model()

            _ = cross_val_score(model, self.X, self.y,
                                cv=self.cv, scoring=self.metrics_scorer)

            return self.metrics_scorer.get_results()

    @staticmethod
    def _custom_evaluate(model, y, X, cv=None, scoring=None):

        if cv is None or scoring is None:
            raise ValueError("Please handover cv and scoring")

        results = defaultdict(list)

        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            model.fit(X.iloc[train_index], y.iloc[train_index])
            preds = model.predict(X.iloc[test_index])
            for metrics in scoring:
                name = str(metrics)[:-2]
                res = metrics(y.iloc[test_index], preds)
                results[name].append(res)

        return results
