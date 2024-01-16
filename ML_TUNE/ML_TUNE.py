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
class Ml_Tune(Config_Utils):
    def __init__(self,res_dic,pred_method,classif_type,is_ts,k_best=3, *args, **kwargs):
        super().__init__()
        self.k_best=k_best
        self.pred_method=pred_method
        self.classif_type=classif_type
        self.is_ts=is_ts
        self.res_dic=res_dic
        self.cv = self._define_cv(self.is_ts)
    def tune(self,*args,**kwargs):
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
    def _process_dic(res):
        return pd.DataFrame(res).assign(CV=lambda df: np.arange(df.shape[0]) + 1)
    def _define_classes(self,model,transformer,dim_reducer,params):
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

    def objective_function(self, model,  X, y,target_metric, trial):
        params_model = self.hyper_parameter_register(model, trial)


        if not model in  self.configs['models'][self.pred_method]:
             KeyError("Model is not checked in ")


        model=self.configs['models'][self.pred_method][model]['object']


        model = model(**params_model)



        results=self._cross_vals(model,X,y)

        return -np.mean(results[target_metric])


    def optimize(self, partial_objective, trial):
        return partial_objective(trial)
    def _cross_vals(self,model,X,y,handle_imbalance=True):



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
        elif key=='HistGradientBoostingClassifier':
            return  {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
        }
        elif key=='HistGradientBoostingRegressor':
            return  {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
        }
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
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
        }
        elif key == 'ExtraTreesRegressor':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
                'warm_start': trial.suggest_categorical('warm_start', [True, False])
            }
        elif key == 'ExtraTreesClassifier':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
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
            }
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
        'l2_leaf_reg':50}
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
        elif key=='KNeighborsClassifier':
            return {'n_neighbors':trial.suggest_categorical('n_neighbors', [ i for i in range(1,8)])}
        elif key=='KNeighborsRegressor':
            return {'n_neighbors':trial.suggest_categorical('n_neighbors', [ i for i in range(1,8)])}
        elif key=='TapNetRegressor':
            return {
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
            }
        elif key=='KNeighborsTimeSeriesRegressor':
            return {
    'n_neighbors': trial.suggest_int("n_neighbors", 1, 20),
    'weights': trial.suggest_categorical("weights", ["uniform", "distance"]),
    'algorithm': trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
    'distance': trial.suggest_categorical("distance", ["euclidean", "squared", "dtw", "ddtw", "wdtw", "wddtw", "lcss", "edr", "erp", "msm"]),
    'leaf_size': trial.suggest_int("leaf_size", 10, 100),
    'n_jobs': trial.suggest_categorical("n_jobs", [None, -1, 1, 2, 4, 8])
}
        elif key=='CNNRegressor':
            return {
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
}
        elif key=='RocketRegressor':
            return {
    'num_kernels': trial.suggest_int("num_kernels", 5000,10000,15000, 20000),
    # Add any other relevant hyperparameters here
    'random_state': trial.suggest_int("random_state", 0, 100)
}
        elif  key == 'TimeSeriesForestClassifier':
            return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                 }
        elif key == 'KNeighborsTimeSeriesClassifier':
            return {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'algorithm': trial.suggest_categorical('algorithm', ['brute', 'auto']),
        'distance': trial.suggest_categorical('distance', ['dtw', 'ddtw', 'wdtw', 'lcss'])
    }
        elif key == 'ShapeletTransformClassifier':
            return {
    'max_shapelet_length': trial.suggest_int('max_shapelet_length', 3, 10)
}
        elif key == 'TimeSeriesSVMClassifier':
            return {
    'C': trial.suggest_float('C', 0.1, 10.0),
    'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
    'degree': trial.suggest_int('degree', 2, 5)
}
        elif key == 'RandomIntervalSpectralForest':
            return {
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    'min_interval': trial.suggest_int('min_interval', 1, 10),
    'max_interval': trial.suggest_int('max_interval', 1, 10)
}
        elif key == 'RocketClassifier':
            return {
    'num_kernels': trial.suggest_int('num_kernels', 1000, 10000)
}
        elif key == 'MrSEQLClassifier':
            return {
    'symrep': trial.suggest_categorical('symrep', ['sax', 'sfa']),
    'seql_mode': trial.suggest_categorical('seql_mode', ['clf', 'fs'])
}
        elif key=='WEASEL':
            return {
    'anova': trial.suggest_categorical("anova", [True, False]),
    'bigrams': trial.suggest_categorical("bigrams", [True, False]),
    'binning_strategy': trial.suggest_categorical("binning_strategy", ["equi-depth", "equi-width", "information-gain"]),
    'window_inc': trial.suggest_int("window_inc", 1, 10),
    'p_threshold': trial.suggest_float("p_threshold", 0.01, 0.1),
    'alphabet_size': trial.suggest_int("alphabet_size", 2, 10),
    'feature_selection': trial.suggest_categorical("feature_selection", ["chi2", "none", "random"]),
    'support_probabilities': trial.suggest_categorical("support_probabilities", [True]),
    'random_state': trial.suggest_int("random_state", 0, 100)
}
        elif key == 'MUSE':
            return {
    'anova': trial.suggest_categorical("anova", [True, False]),
    'variance': trial.suggest_categorical("variance", [True, False]),
    'bigrams': trial.suggest_categorical("bigrams", [True, False]),
    'window_inc': trial.suggest_int("window_inc", 1, 10),
    'alphabet_size': trial.suggest_int("alphabet_size", 2, 10),
    'p_threshold': trial.suggest_float("p_threshold", 0.01, 0.1),
    'use_first_order_differences': trial.suggest_categorical("use_first_order_differences", [True, False]),
    'support_probabilities': trial.suggest_categorical("support_probabilities", [True]),
    'feature_selection': trial.suggest_categorical("feature_selection", ["chi2", "none", "random"]),
    'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]), # 1 for single-threaded or -1 for using all processors
    'random_state': trial.suggest_int("random_state", 0, 100)
}
        elif key=='Arsenal':
            return {
    'num_kernels': trial.suggest_int("num_kernels", 100, 5000),
    'n_estimators': trial.suggest_int("n_estimators", 10, 100),
    'rocket_transform': trial.suggest_categorical("rocket_transform", ["rocket", "minirocket", "multirocket"]),
    'max_dilations_per_kernel': trial.suggest_int("max_dilations_per_kernel", 10, 100),
    'n_features_per_kernel': trial.suggest_int("n_features_per_kernel", 1, 10),
    'time_limit_in_minutes': trial.suggest_float("time_limit_in_minutes", 0.0, 60.0),
    'contract_max_n_estimators': trial.suggest_int("contract_max_n_estimators", 10, 500),
    'save_transformed_data': trial.suggest_categorical("save_transformed_data", [True, False]),
    'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]),  # 1 for single-threaded or -1 for using all processors
    'random_state': trial.suggest_int("random_state", 0, 100)
}
        elif key=='IndividualBOSS':
            return {
    'window_size': trial.suggest_int("window_size", 5, 20),
    'word_length': trial.suggest_int("word_length", 3, 16),
    'norm': trial.suggest_categorical("norm", [True, False]),
    'alphabet_size': trial.suggest_int("alphabet_size", 2, 10),
    'save_words': trial.suggest_categorical("save_words", [True, False]),
    'use_boss_distance': trial.suggest_categorical("use_boss_distance", [True, False]),
    'feature_selection': trial.suggest_categorical("feature_selection", ['none', 'entropy', 'gini']),
    'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]),  # 1 for single-threaded or -1 for using all processors
    'random_state': trial.suggest_int("random_state", 0, 100)
}
        elif key=='BOSSEnsemble':
            return {
            'threshold': trial.suggest_float("threshold", 0.8, 1.0),
             'max_ensemble_size': trial.suggest_int("max_ensemble_size", 100, 1000),
             'max_win_len_prop': trial.suggest_float("max_win_len_prop", 0.5, 1.0),
             'min_window': trial.suggest_int("min_window", 5, 20),
            'save_train_predictions': trial.suggest_categorical("save_train_predictions", [True, False]),
            'alphabet_size': trial.suggest_int("alphabet_size", 2, 6),
            'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]),  # 1 for single-threaded or -1 for using all processors
            'use_boss_distance': trial.suggest_categorical("use_boss_distance", [True, False]),
         'feature_selection': trial.suggest_categorical("feature_selection", ["chi2", "none", "random"]),
            'random_state': trial.suggest_int("random_state", 0, 100)}
        elif key=='ContractableBOSS':
            return {
    'n_parameter_samples': trial.suggest_int("n_parameter_samples", 100, 500),
    'max_ensemble_size': trial.suggest_int("max_ensemble_size", 10, 100),
    'max_win_len_prop': trial.suggest_float("max_win_len_prop", 0.1, 1.0),
    'time_limit_in_minutes': trial.suggest_float("time_limit_in_minutes", 0.0, 60.0),
    'contract_max_n_parameter_samples': trial.suggest_int("contract_max_n_parameter_samples", 100, 1000),
    'save_train_predictions': trial.suggest_categorical("save_train_predictions", [True, False]),
    'feature_selection': trial.suggest_categorical("feature_selection", ["chi2", "none", "random"]),
    'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]),  # 1 for single-threaded or -1 for using all processors
    'random_state': trial.suggest_int("random_state", 0, 100)
}
        elif key=='CNNClassifier':
            return {
        'n_epochs': trial.suggest_int("n_epochs", 100, 2000),  # 100 to 2000
        'batch_size': trial.suggest_int("batch_size", 8, 64),  # 8 to 64

        'n_conv_layers': trial.suggest_int("n_conv_layers", 1, 5),  # 1 to 5
        'random_state': trial.suggest_int("random_state", 0, 100),  # 0 to 100
        'verbose': trial.suggest_categorical("verbose", [True, False]),
        'loss': trial.suggest_categorical("loss", ["mean_squared_error", "categorical_crossentropy", "binary_crossentropy"]),
        'metrics': trial.suggest_categorical("metrics", [["accuracy"], ["precision", "recall"]]),
        'activation': trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"]),
        'use_bias': trial.suggest_categorical("use_bias", [True, False]),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"]) }
        elif key=='RotationForest':
            return {
                'n_estimators': trial.suggest_int("n_estimators", 100, 500),  # 100 to 500
                'min_group': trial.suggest_int("min_group", 1, 5),  # 1 to 5
                'max_group': trial.suggest_int("max_group", 3, 10),  # 3 to 10
                'remove_proportion': trial.suggest_float("remove_proportion", 0.1, 1.0),  # 0.1 to 1.0
                'time_limit_in_minutes': trial.suggest_float("time_limit_in_minutes", 0.0, 60.0),  # 0.0 to 60.0 minutes
                'contract_max_n_estimators': trial.suggest_int("contract_max_n_estimators", 100, 1000),  # 100 to 1000
                'save_transformed_data': trial.suggest_categorical("save_transformed_data", [True, False]),
                'n_jobs': trial.suggest_categorical("n_jobs", [1, -1]),
                # 1 for single-threaded or -1 for using all processors
                'random_state': trial.suggest_int("random_state", 0, 100)  # 0 to 100
            }
        elif key=='FCNClassifier':
            return {
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
            }

            # Note: The optimizer's learning rate (if variable) needs special handling depending on the chosen optimizer.

    @staticmethod
    def _custom_evaluate(model, y, X, cv=None, scoring=None):

        if cv is None or scoring is None:
            raise ValueError("Please handover cv and scoring")

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
    def _handle_imbalance(X,y):
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X,y)
        return X,y
    def _get_scorings(self):
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
