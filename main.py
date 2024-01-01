
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
k=sns.load_dataset('iris')
k['species']=k['species'].map({'setosa':1,'versicolor':0,'virginica':2})
X,target=k.drop(columns=['species']),k[['species']]
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data: Flatten and scale
train_images = train_images.reshape((train_images.shape[0], -1)) / 255.0
test_images = test_images.reshape((test_images.shape[0], -1)) / 255.0




IDX=pd.DataFrame(train_images).sample(n=1000).index

X=pd.DataFrame(train_images).loc[IDX]
#X=pd.DataFrame(np.random.randn(len(train_images),34))
target=pd.DataFrame(train_labels).loc[IDX]


X=sns.load_dataset('diamonds').drop(columns=['carat'])
target=sns.load_dataset('diamonds')[['carat']]
ct=0

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
    #configs.get_transforms_available(is_ts=False,pred_med='Classification')
    #configs.get_models_available(is_ts=False,pred_med='Classification')
    obj = Ml_Main(X, y=target, transform=configs.get_transforms_available(is_ts=False,pred_med='Regression'),  # DWTTransformer#PartialAutoCorrelationTransformer
                  features_selection='LassoCV', dim_reduction=configs.get_dim_reductions_available(pred_med='Regression')
                  , n_jobs=1, ml_model=configs.get_models_available(is_ts=False,pred_med='Regression')).Process()

    obj.Tune(5).get_model_metrics().to_csv(f"./Outputs/Tuned_results.csv",index=None)

