
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
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


def datasets(t='ts',d='classification',Multiclass=True):
    if t=='ts' and d=='classification':
        if Multiclass:
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
            data['Label'] = np.random.randint(0, 4, size=len(data))

            return data.drop(columns=['Label']), data[['Label']]
        else:
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

            configs = Config_Utils()
            return  data.drop(columns=['Label']), data[['Label']]
    elif t=='ts' and d=='regression':
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        datetime_index= pd.date_range(start=start_date, end=end_date, freq='H')
        feature_1 = np.random.rand(len(datetime_index))
        feature_2 = np.random.rand(len(datetime_index))

# Combine into a DataFrame
        time_series_data = pd.DataFrame({
                'Feature_1': feature_1,
                'Feature_2': feature_2
            }, index=datetime_index)

        time_series_data['target'] =time_series_data['Feature_1'].shift(-10)

        time_series_data.dropna(inplace=True)

        return time_series_data.drop(columns=['target']),time_series_data[['target']]
    elif    t=='tab' and d=='regression':
        X=sns.load_dataset('diamonds').drop(columns=['carat'])
        target=sns.load_dataset('diamonds')[['carat']]
        return X,target
    elif    t=='tab' and d=='classification' and Multiclass :
        X=sns.load_dataset('iris').drop(columns=['species'])
        target=sns.load_dataset('iris')[['species']].map({'setosa':1,'versicolor':0,'virginica':2})
        return X,target



import pandas as pd
import numpy as np

configs=Config_Utils()


#print(X.isnull().sum().any(),target.isnull().sum().any())
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
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
    #configs.get_models_available(is_ts=True,pred_med='Classification')
    #configs.get_transforms_available(is_ts=False,pred_med='Regression')
    l=configs.get_transforms_available(is_ts=True,pred_med='Classification')
    l.remove('Rocket')
    l.remove('MiniRocketMultivariate')
    l.remove('TSFreshFeatureExtractor')
    res={}



    for ele in ['Classification', 'Regression']:
        for stuff in ['ts','tab']:
            for mod in configs.get_models_available(is_ts=True,pred_med=ele):
                X,target=datasets(stuff,ele,False)
                if mod in ['HIVECOTEV2','HIVECOTEV1','Arsenal','ElasticEnsemble','RocketClassifier']:
                    continue

                try:
                    obj = Ml_Main(X, y=target, transform=['MinMaxScaler','StandardScaler'],  # DWTTransformer#PartialAutoCorrelationTransformer
                  features_selection=None, dim_reduction=None
                  , n_jobs=1, ml_model=mod).Process()

                    obj.Tune(5).get_model_metrics().to_csv(f"./Outputs/Tuned_results.csv",index=None)
                except Exception as e:
                    res[mod]=e
                print(res)

