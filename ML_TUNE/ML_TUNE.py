import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,log_loss
from ML_CONFIGS_UTILS.ML_CONFIGS import Config_Utils,MultiScorer
from functools import partial
from imblearn.over_sampling import SMOTE
from collections import defaultdict
from optuna.pruners import MedianPruner
class Ml_Tune(Config_Utils):
    def __init__(self,res_dic,pred_method,is_ts,k_best=3, *args, **kwargs):
        super().__init__()
        self.k_best=k_best
        self.pred_method=pred_method
        self.is_ts=is_ts
        self.res_dic=res_dic
        self.cv = self._define_cv(self.is_ts)
    def tune(self):
        tune_results=[]
        tuned_ojects={}

        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)


        if self.pred_method=='Classification':
            dic={'log_loss':['mean']}
        else:
            dic={'mean_squared_error': ['mean']}

        dim=self.res_dic['metrics_model'][0].shape[0]
        k_best=int(min(dim/self.configs['n_cvs'],self.k_best))


        model_metrics=pd.concat(self.res_dic['metrics_model'])
        k_best_models=(model_metrics\
                            .groupby(['model','transform','dim_red','ID'])
                            .agg(dic)\
                            .droplevel(level=1,axis=1)\
                            .nsmallest(k_best,columns=['log_loss' if self.pred_method=='Classification' else 'mean_squared_error'])\
                            .sort_values(by=['log_loss' if self.pred_method=='Classification' else 'mean_squared_error'])
                            .reset_index())

        X_train= pd.concat(self.res_dic['X'])
        y_train= pd.concat(self.res_dic['y'])


        for model, transform,i_d in zip(k_best_models['model'].tolist(), k_best_models['transform'].tolist(),k_best_models['ID'].tolist()):
            # Create a partial function that includes the extra_args and the trial object
            X=X_train.query("ID==@i_d").drop(columns=['ID'])
            y=y_train.query("ID==@i_d").drop(columns=['ID'])
            partial_objective = partial(self.objective_function, model, transform,X.copy(),y)

            # Optimize using the partial function
            study = optuna.create_study(pruner=pruner, direction='maximize')
            study.optimize(lambda trial: self.optimize(partial_objective,trial), n_trials=200)

            model_obj,transformer_obj=self._define_classes(model,transform,study)

            X = transformer_obj.fit_transform(X)

            res_dic=self._cross_vals(model_obj, X, y)
            tune_results.append(self._process_dic(res_dic).assign(Rank=i_d,Tuned=True,dim_red=None,transform=transform,model=model))

            tuned_ojects[model+'_'+transform+'_'+i_d]={'model':model_obj,'transformer':transformer_obj,
                                                       'best_score':study.best_value,'best_params':study.best_params,'X':X,'y':y}

        return pd.concat(tune_results),tuned_ojects
    @staticmethod
    def _process_dic(res):
        return pd.DataFrame(res).assign(CV=lambda df: np.arange(df.shape[0]) + 1)
    def _define_classes(self,model,transform,params):
        model = self.configs['models'][self.pred_method][model]['object']
        transform = self.configs['transforms'][transform]['object']

        model = model()
        transformer = transform()
        for key, value in params.best_params.items():
            default_value = getattr(model, key, False)
            if not isinstance(default_value,bool):
                setattr(transformer, key, value)
            default_value = getattr(transformer, key, False)
            if not isinstance(default_value,bool):
                setattr(transformer, key, value)
        return model,transformer

    def objective_function(self, model, transform, X, y, trial,handle_imbalance=True):
        params_model = self.hyper_parameter_register(model, trial)
        params_transform= self.hyper_parameter_register(transform, trial)

        if not model in  self.configs['models'][self.pred_method]:
             KeyError("Model is not checked in ")
        if not transform in self.configs['transforms']:
            KeyError("Transform is not checked in ")

        model=self.configs['models'][self.pred_method][model]['object']
        transformer=self.configs['transforms'][transform]['object']

        transformer=transformer(**params_transform)
        model = model(**params_model)

        X=transformer.fit_transform(X)

        results=self._cross_vals(model,X,y)

        return -np.mean(results['log_loss'])


    def optimize(self, partial_objective, trial):
        return partial_objective(trial)
    def _cross_vals(self,model,X,y,handle_imbalance=True):

        if self.is_ts:

            results = self._custom_evaluate(model=model,
                                        y=y,
                                        X=X,
                                        cv=self.cv,
                                        scoring=[val[0] if self.pred_method == 'Classification' else
                                     val[0]() for k, val in self.configs['metrics']['ts'][self.pred_method].items()])
        else:
            self.metrics_scorer = MultiScorer(self.configs['metrics']['tab'][self.pred_method])
            if handle_imbalance and self.pred_method == 'Classification':
                X,y = self._handle_imbalance(X,y)

            _= cross_val_score(estimator=model, X=X, y=y,
                                cv=self.cv, scoring=self.metrics_scorer)

        return self.metrics_scorer.get_results()
    def hyper_parameter_register(self, key, trial):

        if key == 'LogisticRegression':
            return  {'penalty': trial.suggest_categorical('penalty', [ 'l2']),
                     'C':trial.suggest_float("C", 0.5, 1.5),
                     'max_iter':trial.suggest_categorical('max_iter', [100,200,300]),
                     'solver':trial.suggest_categorical('solver',['lbfgs', 'liblinear', 'newton-cg'
                         , 'newton-cholesky', 'sag', 'saga'])}
        elif key=='StandardScaler':
            return {}
        elif key=='LinearRegression':
            return {}
        elif key=='QuantileTransformer':
            return {'n_quantiles':trial.suggest_categorical('n_quantiles', [j for j in range(1,3000,500)]),
                    'output_distribution':trial.suggest_categorical('output_distribution',['uniform','normal'])}
        elif key=='PowerTransformer':
            return {'method':trial.suggest_categorical('method',['yeo-johnson'])}
        elif key=='RobustScaler':
            return {'unit_variance': trial.suggest_categorical('unit_variance', [False, True])}
        elif key=='PolynomialFeatures':
            return {'degree': trial.suggest_categorical('degree', [ i for i in range(1,5)])}
        elif key=='SVC':
            return {}
        elif key=='MinMaxScaler':
            return {}
        elif key=='SVR':
            return {'C':trial.suggest_float("C", 0.5, 1.5),
                    'kernel': trial.suggest_categorical('kernel',
                    ['linear', 'poly', 'rbf', 'sigmoid']),

                    }
        elif key=='XGBClassifier':
            return {
                # this parameter means using the GPU when training our model to speedup the training process
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                'verbosity':0,
                'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                              [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
                'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
                'n_estimators': trial.suggest_categorical('n_estimators', [i for i in range(1,2000,100)]),
                'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17,20]),
                'random_state': trial.suggest_categorical('random_state', [2020]),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            }
        elif key=='RandomForestClassifier':
            return  {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
        }
        elif key == 'ExtraTreesRegressor':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
                'warm_start': trial.suggest_categorical('warm_start', [True, False])
            }
        elif key == 'ExtraTreesClassifier':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
                'warm_start': trial.suggest_categorical('warm_start',[True,False])
            }
        elif key=='DecisionTreeClassifier':
            return {'max_depth': trial.suggest_int('max_depth', 4, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
                    }
        elif key=='DecisionTreeRegressor':
            return {'max_depth': trial.suggest_int('max_depth', 4, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
                    }
        elif key=='AdaBoostClassifier':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            }
        elif key=='AdaBoostRegressor':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            }
        elif key=='RandomForestRegressor':
            return  {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
        }
        elif key=='XGBRegressor':
            return {
                # this parameter means using the GPU when training our model to speedup the training process
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                'verbose': 0,
                'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                              [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
                'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
                'n_estimators': trial.suggest_categorical('n_estimators', [i for i in range(1,2000,100)]),
                'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17,20]),
                'random_state': trial.suggest_categorical('random_state', [2020]),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            }
        elif key=='LGBMRegressor':
            return {
        "n_estimators": trial.suggest_categorical('n_estimators', [i for i in range(1,2000,100)]),
        "verbose": 0,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100) }

        elif key=='LGBMClassifier':
            return {
        "n_estimators": trial.suggest_categorical('n_estimators', [i for i in range(1,2000,100)]),
                "verbose": 0,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100) }
        elif key=='CatBoostClassifier':
            return {
        "n_estimators": trial.suggest_categorical('n_estimators', [i for i in range(1, 2000, 100)]),
        'leaf_estimation_iterations':1,
        'boosting_type':'Plain',
        'thread_count':-1,
                'silent': True,
        'depth' : trial.suggest_int('depth', 4, 16),
        'random_strength' :trial.suggest_int('random_strength', 0, 100),
        'bagging_temperature' : trial.suggest_float('bagging_temperature', 0.01, 100.00),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'l2_leaf_reg':50,
            'verbose': False}
        elif key=='CatBoostRegressor':
            return {
        "n_estimators": trial.suggest_categorical('n_estimators', [i for i in range(1, 2000, 100)]),
        'leaf_estimation_iterations':1,
        'boosting_type':'Plain',
                'silent': True,
        'depth' : trial.suggest_int('depth', 4, 16),
        'random_strength' :trial.suggest_int('random_strength', 0, 100),
        'bagging_temperature' : trial.suggest_float('bagging_temperature', 0.01, 100.00),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'l2_leaf_reg':50,
            'verbose': False}
        elif key=='ElasticNet':
            return {'alpha':trial.suggest_float('alpha', 0.03, 2),
                    'l1_ratio':trial.suggest_float('l1_ratio', 0.1, 2)}
        elif key=='Ridge':
            return {'alpha':trial.suggest_float('alpha', 0.03, 2)}
        elif key=='MLPClassifier':
            return {'hidden_layer_sizes':trial.suggest_categorical('hidden_layer_sizes', [(100,),(300,),(500,)]),
                    'activation':trial.suggest_categorical('activation', ['logistic','relu','tanh'])}
        elif key=='MLPRegressor':
            return {'hidden_layer_sizes':trial.suggest_categorical('hidden_layer_sizes', [(100,),(300,),(500,)]),
                    'activation':trial.suggest_categorical('activation', ['logistic','relu','tanh'])}
        elif key=='CustomMathTransformer':
            return {}
        elif key=='EqualWidthDiscretiser':
            return {}
        elif key=='MinMaxScaler':
            return {}
        elif key=='PolynomialFeatures':
            return {'degree':trial.suggest_categorical('degree', [ i for i in range(1,4)])}

    @staticmethod
    def _custom_evaluate(model, y, X, cv=None, scoring=None):

        if cv is None or scoring is None:
            raise ValueError("Please handover cv and scoring")

        results = defaultdict(list)

        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            model.fit(X.iloc[train_index], y.iloc[train_index,0])
            preds = model.predict(X.iloc[test_index])
            res = scoring(y.iloc[test_index], preds)
            results[scoring.__name__].append(res)

        return -np.mean(results[scoring.__name__])
    @staticmethod
    def _handle_imbalance(X,y):
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X,y)
        return X,y
