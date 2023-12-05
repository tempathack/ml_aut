from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from ML_CONFIGS_UTILS.ML_CONFIGS import CONFIG_UTILS

class ML_TARGET(CONFIG_UTILS):
    def __init__(self, X,y, *args, **kwargs):
        super().__init__(X, y, *args, **kwargs)
        self.X = self.eval_df(X)
        self.y = self.eval_df(y)
        self._unison_check(self.X,self.y)
        self.is_ts = self.TS_check(X)
    @staticmethod
    def _standardize(y):
        return (y-y.mean())/y.std()
    @staticmethod
    def _unison_check(x,y):
        assert (x.index==y.index).all(),"index needs to be equal across all entries"
    @staticmethod
    def _box_cox(y):
        assert y.iloc[:,0].gt(0).all(),'can only be applied to positive values'
        return  pd.DataFrame(stats.boxcox(y)[0],columns=y.columns,index=y.index)
    @staticmethod
    def _yeo_johnson(y):
        return  pd.DataFrame(stats.yeojohnson(x)[0],columns=y.columns,index=y.index)
    @staticmethod
    def _decomp(y,deseasonalize=False,detrend=False):

        decomped=seasonal_decompose(y)

        if deseasonalize:
            return pd.DataFrame(decomped.resid+decomped.trend,index=y.index,columns=y.columns)
        elif detrend:
            return pd.DataFrame(decomped.resid + decomped.seasonal, index=y.index, columns=y.columns)
        elif deseasonalize and detrend:
            return pd.DataFrame(decomped.resid, index=y.index, columns=y.columns)
    def target_design(self,target_steps,scale_method='standardize',deseasonalize=False,detrend=False,*args,**kwargs):

        if self.is_ts:
            self.y=self._decomp(self.y,deseasonalize,detrend)


        if scale_method=='standardize':
            self.y =self._standardize(self.y)
        elif scale_method=='box_cox':
            self.y = self._box_cox(self.y)
        elif scale_method=='yeo_johnson':
            self.y = self._yeo_johnson(self.y)

        for step in target_steps:
            cmb=self.X.join(self.y.iloc[:,0].shift(step)).dropna()

            yield (cmb.loc[:,self.X.columns],cmb.drop(columns=self.X.columns))