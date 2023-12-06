import seaborn as sns
import pandas as pd
import numpy as np
from ML_CONFIGS_UTILS.ML_CONFIGS import Config_Utils
from ML_HANDLING.ML_MAIN import Ml_Main


ds = sns.load_dataset("flights")
print(ds)
ds['month']=np.hstack([np.arange(12)+1 for _ in range(12)]).flatten()

ds['year']=ds['year'].astype(str)
ds['date'] = (ds['year'] +'-'+ ds['month'].astype(str)).apply(pd.to_datetime,format='%Y-%m')
ds.set_index('date',inplace=True)
target=ds[['passengers']]
X = ds.drop(columns=['passengers','month'])


configs=Config_Utils()

obj = Ml_Main(X, y=target, transform=[['RobustScaler', 'StandardScaler'],'StandardScaler'],
                  features_selection=None, ml_model=['KNeighborsTimeSeriesRegressor']).Process(mode='seq')
print(obj)



