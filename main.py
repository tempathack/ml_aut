import seaborn as sns
import pandas as pd
import numpy as np
from ML_CONFIGS_UTILS.ML_CONFIGS import Config_Utils
from ML_HANDLING.ML_MAIN import Ml_Main
import warnings
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define the start and end dates for the time series
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)

# Generate a DateTimeIndex with hourly frequency
date_range = pd.date_range(start=start_date, end=end_date, freq='H')

# Create a DataFrame with DateTimeIndex
data = pd.DataFrame(index=date_range).iloc[:1000]

# Generating features (random in this example)
data['Feature1'] = np.random.randn(len(data))
data['Feature2'] = np.random.rand(len(data))
# Add more features as needed

# Generating labels for classification (binary labels in this example)
data['Label'] = np.random.randint(0, 2, size=len(data))

configs=Config_Utils()
X,target=data.drop(columns=['Label']),data[['Label']]
ct=0
if __name__ == '__main__':

#notworked 'WindowSummarizer' 'Rocket'  'MiniRocket''MiniRocketMultivariate'
    for trans in configs.get_transforms_available(is_ts=True,pred_med='Classification'):
        for model in configs.get_models_available(is_ts=True,pred_med='Classification'):
            print(trans,model)
            if not trans in ['SummaryTransformer']:
                continue
            ct+=1
            obj = Ml_Main(X, y=target, transform=trans,
                    features_selection=None,n_jobs=-1, ml_model=model).Process(results_return=True)




