import ray
from ML_CONFIGS_UTILS.ML_CONFIGS import  CONFIG_UTILS
from ML_FEATURE_SELECTION.ML_FEATURE_SELECTION import ML_SELECT
from ML_TRAINING.ML_TRAINING import ML_TRAIN
from ML_TRANSFORMING.ML_TRANSFORMING import ML_PROCESS
from ML_DIMENSIONALITY_REDUCTION.ML_DIMENSIONALITY_REDUCTION import ML_REDUCE
from ML_TARGET_TRANSFORMS.ML_TARGET import ML_TARGET
from itertools import product

class ML_MAIN(CONFIG_UTILS):
    def __init__(self,X,y,transform=None,
                      features_selection=None,
                      dim_reduction=None,
                      target_adjustment=None,
                      ml_model=None):

        super(ML_MAIN, self).__init__()


        assert transform is not None, "please specfiy a transform you want to train"
        assert ml_model is not None,"please specfiy a model you want to train"

        self.X = self.eval_df(X)
        self.y = self.eval_df(y)
        self.transform = False if transform is None else transform if isinstance(transform,list) else [transform]
        self.features_selection = False if features_selection is None else  features_selection
        self.dim_reduction = False if dim_reduction is None else dim_reduction if isinstance(dim_reduction,list) else [dim_reduction]
        self.target_adjustment = False if target_adjustment is None else target_adjustment


        self.model= ml_model if isinstance(ml_model,list) else [ml_model]

        if isinstance(self.transform, list):
            self.ml_process = ML_PROCESS(X=self.X)

        if isinstance(self.features_selection, str):
            self.ml_select = ML_SELECT(X=self.X, y=self.y)

        if isinstance(self.target_adjustment, list):
            self.ml_target = ML_TARGET(X=self.X, y=self.y)

        if isinstance(self.dim_reduction, list):
            self.ml_reduce = ML_REDUCE(X=self.X, y=self.y)

        self.ml_train = ML_TRAIN(X=self.X, y=self.y)

    def Process(self,mode='raw',*args, **kwargs):

        is_ml_select = hasattr(self, 'ml_select')
        is_ml_reduce = hasattr(self, 'ml_reduce')
        self.results=[]

        if mode == 'ray':
            # Initialize Ray
            ray.init()

            # Define the generator function to be mapped to the Ray actors
            @ray.remote
            def generate_result(args, is_ml_select=is_ml_select, is_ml_reduce=is_ml_reduce, dim_red=None):
                return self._define_generator_wrapper(args, is_ml_select, is_ml_reduce, dim_red)

            # Use Ray to parallelize the generator function
            if is_ml_reduce:
                # If ml_reduce attribute exists, define the generator function with dim_red
                # parameter set to self.dim_reduction
                futures = [generate_result.remote(
                    transform_model,
                    dim_red=self.dim_reduction if isinstance(self.dim_reduction, list) else [self.dim_reduction]
                ) for transform_model in product(self.transform, self.model)]

            else:
                # If ml_reduce attribute does not exist, define the generator function with
                # dim_red parameter set to None
                futures = [generate_result.remote(transform_model, dim_red=None) for transform_model in
                           product(self.transform, self.model)]

            # Collect the results from the Ray futures and append them to the results list
            for future in ray.get(futures):
                self.results.append(future)

            # Shut down Ray
            ray.shutdown()




        elif mode == 'seq':
            # Sequential processing
            if is_ml_reduce:
                # Iterate through combinations of transform, dim_red, and model
                for transform in self.transform:
                    for dim_red in self.dim_reduction:
                        for model in self.model:
                            self.results.append(self._define_generator(
                                transform, dim_red, model,
                                is_ml_select, is_ml_reduce,
                                dim_red=dim_red, *args, **kwargs
                            ))
            else:
                # Iterate through combinations of transform and model
                for transform in self.transform:
                    for model in self.model:
                        self.results.append(self._define_generator(
                            transform, model,
                            is_ml_select, is_ml_reduce,
                            *args, **kwargs
                        ))


        return self._unpack_results(self.results)
    def _define_generator_wrapper(self,args,is_ml_select=False,is_ml_reduce=False ,dim_red=None):
        return self._define_generator(*args,is_ml_select=is_ml_select,is_ml_reduce=is_ml_reduce ,dim_red=dim_red)


    def _define_generator(self,transform,model,is_ml_select=False,is_ml_reduce=False ,dim_red=None,*args, **kwargs):


        X=self.ml_process.main_transform(transform=transform,*args, **kwargs)

        if is_ml_select:
            self.ml_select.set_X_y(X=X,*args,**kwargs)
            X=self.ml_select.feature_selection(method=self.features_selection,*args,**kwargs)
        if is_ml_reduce:
            self.ml_reduce.set_X_y(X=X)
            X=self.ml_reduce.dimensionality_reduction(method=dim_red,*args, **kwargs)


        self.ml_train.set_X_y(X=X)
        metrics=self.ml_train.train_model(model=model,*args, **kwargs)

        return {'processing':{'transform':transform,'features_selection':self.features_selection,
                             'dim_red':False if dim_red is None else dim_red,'model':model},
                'metrics':metrics}