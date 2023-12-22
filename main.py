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
import yfinance as yf

import seaborn

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


X,target=data.drop(columns=['Label']),data[['Label']]

sns.load_dataset('titanic')
d=sns.load_dataset('titanic').drop(columns=['alive'])
X,target=d.drop(columns=['survived']),d[['survived']]
ct=0

if __name__ == '__main__':

    def check_model_already_trained(model_name):
        # Read the text file containing the list of trained models
        with open('trained_models.txt', 'r') as file:
            lines = file.readlines()
            trained_models = [line.strip() for line in lines]

        # Check if the model_name exists in the list of trained models
        return model_name in trained_models

#

    def save_model_to_file(model_name):
        # Save the model_name to the text file
        with open('trained_models.txt', 'a') as file:
            file.write(model_name + '\n')


    k=configs.get_models_available(is_ts=False, pred_med='Classification')
    k.remove('SVC')
    for trans in configs.get_transforms_available(is_ts=False,pred_med='Classification'):
        for model in k:
            for dim_red in configs.get_dim_reductions_available()+[None]:
                obj = Ml_Main(X, y=target, transform=trans,#DWTTransformer#PartialAutoCorrelationTransformer
                          features_selection='LogisticRegressionCV',dim_reduction=dim_red
                              , n_jobs=1, ml_model=model).Process()
                obj.Tune()


                #obj.to_csv(f"./Outputs/{trans+model+str(dim_red)}.csv",index=None)




    for trans in configs.get_transforms_available(is_ts=False,pred_med='Classification'):
        for model in configs.get_models_available(is_ts=False,pred_med='Classification'):
            for dim_red in configs.get_dim_reductions_available()+[None]:
                print(trans,model)
                try:
                    obj = Ml_Main(X, y=target, transform=trans,#DWTTransformer#PartialAutoCorrelationTransformer
                          features_selection='LogisticRegressionCV',dim_reduction=dim_red, n_jobs=1, ml_model=model).Process(
                results_return=True)
                    obj.to_csv(f"./Outputs/{trans+model+str(dim_red)}.csv",index=None)
                except:
                    pass



