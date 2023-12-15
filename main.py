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
from functools import wraps
import time
# Decorator function to run a function for a specified duration
def run_for(duration):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            while True:
                func_result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time

                if elapsed_time >= duration:
                    # Optional: Print a message before jumping out of the loop
                    print(f"Function ran for {elapsed_time} seconds. Jumping out of the loop.")
                    break

                # Replace this with your actual function code
                print("Running...")  # Replace this with your actual function code

                # Simulate some process time
                time.sleep(6)

            # Return the original function result
            return func_result
        return wrapper
    return decorator

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
        with open('trained_models.txt', 'w') as file:
            file.write(model_name + '\n')

    for trans in configs.get_transforms_available(is_ts=True,pred_med='Classification'):
        for model in configs.get_models_available(is_ts=True,pred_med='Classification'):
            print(trans,model)
            if (not trans in ['SummaryTransformer'] and model in ['HIVECOTEV2','HIVECOTEV1']) or check_model_already_trained(model) :
                continue
            start=datetime.now()

            try:
                ct+=1
                obj=Ml_Main(X, y=target, transform=trans,
                    features_selection=None,n_jobs=-1, ml_model=model).Process(results_return=True)
                if obj is not None:
                    obj.to_csv(f"./Outputs/{model}.csv",index=None)
                save_model_to_file(model)
            except:
                print("did not",model)


