import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import chi2,f_classif,mutual_info_classif,f_regression,mutual_info_regression
from sklearn.linear_model import LogisticRegressionCV,LassoCV
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
import plotly.express as px
from LOGGER.LOGGING import WrapStack
from ML_CONFIGS_UTILS.ML_CONFIGS import Config_Utils
from ML_CONFIGS_UTILS.Custom_Errors import MethodNotExecutedError
from typing import Optional,Dict,List,Literal,Set,Tuple,Union,Callable,Any


class Ml_Select(Config_Utils):
    'main feature selection class to ensure the right features'
    def __init__(self, X:pd.DataFrame,y:pd.DataFrame, *args, **kwargs):
        super().__init__()
        self.X =X
        self.y = self.eval_df(y)
        self.args = args
        self.kwargs = kwargs
        self.track_feat_metrics = {}
        self.is_ts = self.TS_check(X)
        self.pred_method=self._class_or_reg(y)

    @property
    def feat_dim(self) -> int:
        return self.X.shape[1]
    @staticmethod
    def _calc_permutation_importance(X:pd.DataFrame, y:pd.DataFrame, n_repeats:Optional[int]=10, *args, **kwargs) -> pd.DataFrame:
        clf = LogisticRegression()  # use logistic regression as sort of benachmark
        res = permutation_importance(clf.fit(X.values, y.values.reshape(-1)), X.values, y.values.reshape(-1),
                                     n_repeats=n_repeats, random_state=0, *args, **kwargs)
        return pd.DataFrame(data={'raw_metric': res.importances_mean,
                                  'importances_std': res.importances_std,
                                  'Ranks': res.importances_mean.argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_f_classif(X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        f_statistics = f_classif(X.values, y.values.reshape(-1), *args, **kwargs)[0]
        return pd.DataFrame(data={'raw_metric': f_statistics,
                                  'Ranks': f_statistics.argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_f_regression(X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        f_statistics = f_regression(X.values, y.values.reshape(-1), *args, **kwargs)[0]
        return pd.DataFrame(data={'raw_metric': f_statistics,
                                  'Ranks': f_statistics.argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_chi2(X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        ch2_statistics = chi2(X.values, y.values.reshape(-1), *args, **kwargs)[0]
        return pd.DataFrame(data={'raw_metric': ch2_statistics,
                                  'Ranks': ch2_statistics.argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_DecisionTreeClassifier(X:pd.DataFrame, y:pd.DataFrame, max_depth=15, *args, **kwargs) -> pd.DataFrame:
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0, *args, **kwargs)
        clf.fit(X.values, y.values.reshape(-1))
        return pd.DataFrame(data={'raw_metric': clf.feature_importances_,
                                  'Ranks': clf.feature_importances_.argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_DecisionTreeRegressor(X:pd.DataFrame, y:pd.DataFrame, max_depth=15, *args, **kwargs) -> pd.DataFrame:
        reg = DecisionTreeRegressor(max_depth=max_depth, random_state=0, *args, **kwargs)
        reg.fit(X.values, y.values.reshape(-1))
        return pd.DataFrame(data={'raw_metric': reg.feature_importances_,
                                  'Ranks': reg.feature_importances_.argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_RandomForestRegressor(X:pd.DataFrame, y:pd.DataFrame, max_depth=15, n_estimators=100, *args, **kwargs) -> pd.DataFrame:
        reg = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators, *args, **kwargs)
        reg.fit(X.values, y.values.reshape(-1))
        return pd.DataFrame(data={'raw_metric': reg.feature_importances_,
                                  'Ranks': reg.feature_importances_.argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_LogisticRegressionCV(X:pd.DataFrame, y:pd.DataFrame, cv, *args, **kwargs) -> pd.DataFrame:
        clf = LogisticRegressionCV(cv=cv, *args, **kwargs)
        clf.fit(X.values, y.values.reshape(-1))
        feats = (clf.coef_ ** 2).sum(axis=0)
        return pd.DataFrame(data={'raw_metric': feats,
                                  'Ranks': feats.argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_LassoCV(X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        reg = LassoCV(cv=5, random_state=5)
        reg.fit(X.values, y.values.reshape(-1))
        return pd.DataFrame(data={'raw_metric': np.abs(reg.coef_),
                                  'Ranks': np.abs(reg.coef_).argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_RandomForestClassifier(X:pd.DataFrame, y:pd.DataFrame, max_depth=15, n_estimators=100, *args, **kwargs) -> pd.DataFrame:
        clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=0, *args,
                                     **kwargs)
        clf.fit(X.values, y.values.reshape(-1))
        return pd.DataFrame(data={'raw_metric': clf.feature_importances_,
                                  'Ranks': clf.feature_importances_.argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_mutual_info_classif(X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        mutual_info = mutual_info_classif(X.values, y.values.reshape(-1))
        return pd.DataFrame(data={'raw_metric': mutual_info,
                                  'Ranks': mutual_info.argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_mutual_info_regression(X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        mutual_info = mutual_info_regression(X.values, y.values.reshape(-1))
        return pd.DataFrame(data={'raw_metric': mutual_info,
                                  'Ranks': mutual_info.argsort().argsort(),
                                  'Columns': X.columns.tolist()})
    @staticmethod
    def _calc_correlation(X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        corr_info = X.join(y).corr(**kwargs).loc[:, y.columns.tolist()[0]].values
        return pd.DataFrame(data={'raw_metric': corr_info[:-1],
                                  'Ranks': corr_info[:-1].argsort().argsort(),
                                  'Columns': X.columns.tolist()})

    @WrapStack.FUNCTION_SCREEN
    def feature_selection(self, method:Optional[str]=None,k_feat:int=30, *args, **kwargs)->pd.DataFrame:
        '''
        main function  to handle feature selection

        :param method: str corresponding to feature selection
        :param k: number of freatures to reduce by feature selection
        :param args:
        :param kwargs:
        :return: reduced Dataframe
        '''
        if method is None:
            if hasattr(self, 'method'):
                method = self.method
            else:
                raise ValueError("mehtod must eiter be given to feature_selection or set on class level")



        k_best=min(k_feat,self.feat_dim)

        if (method is None) or  (method not in self.configs['feat_selections'][self.pred_method]) :
            raise KeyError(f"method must be one of {self.configs['feat_selections'][self.pred_method]}  ")

        if self._validate_3d(self.X):
            return self.X

        self.cv = self._define_cv(self.is_ts)


        if self.pred_method =='Classification':
            if method == 'permutation_importance':
                self.track_feat_metrics['permutation_importance'] = self._calc_permutation_importance(self.X,
                                                                                                      self.y, *args,
                                                                                                      **kwargs)
            elif method == 'f_classif':
                self.track_feat_metrics['f_classif'] = self._calc_f_classif(self.X, self.y, *args, **kwargs)
            elif method == 'chi2' :
                assert self._check_none_negative(self.X),'Non Negative criterion not passed'
                self.track_feat_metrics['chi2'] = self._calc_chi2(self.X, self.y, *args, **kwargs)
            elif method == 'DecisionTreeClassifier':
                self.track_feat_metrics['DecisionTreeClassifier'] = self._calc_DecisionTreeClassifier(self.X,
                                                                                                      self.y, *args,
                                                                                                      **kwargs)
            elif method == 'RandomForestClassifier':
                self.track_feat_metrics['RandomForestClassifier'] = self._calc_RandomForestClassifier(self.X,
                                                                                                      self.y, *args,
                                                                                                      **kwargs)
            elif method == 'mutual_info_classif':
                self.track_feat_metrics['mutual_info_classif'] = self._calc_mutual_info_classif(self.X, self.y,
                                                                                                *args, **kwargs)
            elif method == 'LogisticRegressionCV':
                self.track_feat_metrics['LogisticRegressionCV'] = self._calc_LogisticRegressionCV(self.X, self.y,
                                                                                                  self.cv, *args,
                                                                                                  **kwargs)
            elif method == 'all':
                self.track_feat_metrics['permutation_importance'] = self._calc_permutation_importance(self.X,
                                                                                                      self.y, *args,
                                                                                                      **kwargs)
                self.track_feat_metrics['f_classif'] = self._calc_f_classif(self.X, self.y, *args, **kwargs)
                if self._check_none_negative(self.X):
                    self.track_feat_metrics['chi2'] = self._calc_chi2(self.X, self.y, *args, **kwargs)
                self.track_feat_metrics['DecisionTreeClassifier'] = self._calc_DecisionTreeClassifier(self.X,
                                                                                                      self.y, *args,
                                                                                                      **kwargs)
                self.track_feat_metrics['RandomForestClassifier'] = self._calc_RandomForestClassifier(self.X,
                                                                                                      self.y, *args,
                                                                                                      **kwargs)
                self.track_feat_metrics['mutual_info_classif'] = self._calc_mutual_info_classif(self.X, self.y,
                                                                                                *args, **kwargs)
                self.track_feat_metrics['LogisticRegressionCV'] = self._calc_LogisticRegressionCV(self.X, self.y,
                                                                                                  self.cv, *args,
                                                                                                  **kwargs)
                method = 'RandomForestClassifier' if method == 'all' else method
        elif self.pred_method == 'Regression':
            if method == 'correlation':
                self.track_feat_metrics['correlation'] = self._calc_correlation(self.X, self.y,
                                                                                *args, **kwargs)
            elif method == 'f_regression':
                self.track_feat_metrics['f_regression'] = self._calc_f_regression(self.X, self.y, *args, **kwargs)
            elif method == 'DecisionTreeRegressor':
                self.track_feat_metrics['DecisionTreeRegressor'] = self._calc_DecisionTreeRegressor(self.X, self.y,
                                                                                                    *args, **kwargs)
            elif method == 'RandomForestRegressor':
                self.track_feat_metrics['RandomForestRegressor'] = self._calc_RandomForestRegressor(self.X, self.y,
                                                                                                    *args, **kwargs)
            elif method == 'mutual_info_regression':
                self.track_feat_metrics['mutual_info_regression'] = self._calc_mutual_info_regression(self.X,
                                                                                                      self.y, *args,
                                                                                                      **kwargs)
            elif method== 'LassoCV':
                self.track_feat_metrics['LassoCV'] = self._calc_LassoCV(self.X, self.y, *args,
                                                                        **kwargs)
            elif method == 'all':
                self.track_feat_metrics['correlation'] = self._calc_correlation(self.X, self.y, *args, **kwargs)
                self.track_feat_metrics['f_regression'] = self._calc_f_regression(self.X, self.y, *args, **kwargs)
                self.track_feat_metrics['DecisionTreeRegressor'] = self._calc_DecisionTreeRegressor(self.X, self.y,
                                                                                                    *args, **kwargs)
                self.track_feat_metrics['RandomForestRegressor'] = self._calc_RandomForestRegressor(self.X, self.y,
                                                                                                    *args, **kwargs)
                self.track_feat_metrics['mutual_info_regression'] = self._calc_mutual_info_regression(self.X,
                                                                                                      self.y, *args,
                                                                                                      **kwargs)
                self.track_feat_metrics['LassoCV'] = self._calc_LassoCV(self.X, self.y, self.cv, *args, **kwargs)
                method='RandomForestRegressor' if method=='all' else method

        return self.X.loc[:,self.feat_metrics().query(f"Selection_Method==@method").nlargest(k_best, columns=['Ranks']).Columns.tolist()]
    def feat_metrics(self)->pd.DataFrame:
        '''

        :return: pd.DataFrame including feature_selection results
        '''

        if  self.track_feat_metrics == {} :
            if not  self._validate_3d(self.X):
                raise MethodNotExecutedError('Execute featureselection method first')
            else:
                raise AttributeError('Feature Selection is not Available for 3D Datasets')
        else:
             return pd.concat([pd.DataFrame(data={'Ranks': self.track_feat_metrics[k]['Ranks'],
                                             'Columns': self.track_feat_metrics[k]['Columns'],
                                             'Raw_Metric':self.track_feat_metrics[k]['raw_metric'],
                                             'Selection_Method': k
                                             }) for k in self.track_feat_metrics])
    def feat_curve(self, upper_limit=None, n_fits=5, *args, **kwargs):
        self.track_feat_curve = {}
        if upper_limit is None:
            upper_limit = self.feat_dim
        n_fits = min(upper_limit, n_fits)
        thres_x = [int(x) for x in np.linspace(1, upper_limit, n_fits)]
        feat_df = self.feat_metrics()
        if self.pred_method == 'Regression':
            base_est = RandomForestRegressor(*args, **kwargs, random_state=0)
            scoring = 'neg_mean_absolute_percentage_error'
        else:
            base_est = RandomForestClassifier(*args, **kwargs, random_state=0)
            scoring = 'neg_log_loss'
        for name, group in feat_df.groupby('Selection_Method'):
            scores = []
            for thres in thres_x:
                col_filter = group.nlargest(thres, columns=['Ranks']).Columns.tolist()
                X_slc = self.X.loc[:, col_filter]
                # Perform cross-validation and calculate AUC for each fold
                score = cross_validate(base_est, X_slc.values, self.y.values.reshape(-1), cv=self.cv,
                                       scoring=scoring)
                scores.append(np.mean(score['test_score']))
            self.track_feat_curve[name] = scores
        plot_df = pd.DataFrame(self.track_feat_curve)
        plot_df['x'] = thres_x
        fig = px.line(plot_df.melt(id_vars='x'), x='x', y='value', color='variable',
                      title=f'Regression_k_best={upper_limit}_feat_curve'
                      if self.pred_method == 'reg' else f'Classification_k_best={upper_limit}_feat_curve')
        fig.show()
