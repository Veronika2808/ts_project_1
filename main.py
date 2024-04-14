from preprocessing import preprocess_data, generate_features, intime_oot_split, train_test, train_test_for_prophet
from feature_selection import correlation_between_features_and_target
from fitting_model import fit_model
from changepoints_detection import alarm_changepoint

from datetime import timedelta
import pickle
import optuna
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', None)
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error

from sklearn.svm import SVR
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.utils import resample
import stability as st

pd.DataFrame.iteritems=pd.DataFrame.items


from sklearn.linear_model import Ridge, Lasso


from scipy import stats

import seaborn as sns 
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings

refit_task = False
get_prediction_task = False
train_task = False
alarm_changepoint = False

if __name__ == "__main__":
    
    if train_task:

        #read data
        data = pd.read_excel('Project 1_2024.xlsx', sheet_name=0)

        #some preprocessing
        data_with_macro  = preprocess_data(data)

        #generating features with TSFresh
        X_clean = generate_features(data_with_macro)

        #split intime/out-of-time
        intime_df, oot_df = intime_oot_split(X_clean, data_with_macro, oot_start='2021-01-01')

        #train/val/oot split 
        X_train, X_val, X_oot, y_train, y_val, y_oot = train_test(intime_df, oot_df)
        
        # #train/val/oot split for prophet model
        # df_train_for_prophet, df_val_for_prophet, df_oot_for_prophet = train_test_for_prophet(intime_df, oot_df)

        #select features by correlation  with target - the most stable method
        selected_features = correlation_between_features_and_target(intime_df.drop('date', axis=1))

        X_train, X_val, X_oot = X_train[selected_features], X_val[selected_features], X_oot[selected_features]
        
        #fit Ridge
        fitted_model = fit_model(X_train, X_val, y_train, y_val)

        #save dataset to parquet
        old_processed_data = pd.concat([X_train, X_val, X_oot])
        old_processed_data['target'] = pd.concat([y_train, y_val, y_oot])
        old_processed_data['date'] = pd.to_datetime(data['Date'])
        old_processed_data.to_parquet('Old_data_with_selected_features.parquet')
    
    if get_prediction_task:

         #read data
        data = pd.read_excel('Project 1_2024.xlsx', sheet_name=0)
        new_data = pd.read_excel('new_data.xlsx', sheet_name=0) #with current day data
        current_date = new_data['Date']
        new_df = pd.concat([data, new_data])
        #checking for changepoint
        alarm_changepoint, breakpoint = alarm_changepoint(new_df)
        
        if not alarm_changepoint:
            #some preprocessing
            data_with_macro  = preprocess_data(data)

            old_data = pd.read_parquet('Old_data_with_selected_features.parquet')

            #generating features with TSFresh
            X_clean = generate_features(data_with_macro)[old_data.columns]

            with open('fitted_model.pkl', 'rb') as f:
                lr = pickle.load(f)

            preds = lr.predict(X_clean)

            new_data['preds'] = preds

            new_data.to_parquet(f'Predictions_{current_date}.parquet')
        else:
            raise ValueError('Changepoint in data is present, refit the model manually')



    
    if refit_task:

        #read data
        old_data = pd.read_parquet('Old_data_with_selected_features.parquet')
        new_data = pd.read_excel('new_data.xlsx', sheet_name=0)
        #some preprocessing
        data_with_macro  = preprocess_data(new_data)
        #generating features with TSFresh
        X_clean = generate_features(data_with_macro)

        new_data_selected = X_clean[old_data.columns]

        #concat with old data
        new_df = pd.concat([old_data, new_data])
        
        #checking for changepoint
        alarm_changepoint, breakpoint = alarm_changepoint(new_df)
        
        if not alarm_changepoint:

            new_date = new_df['date'].max() - timedelta(days=30*3)

            #split intime/out-of-time
            intime_df, oot_df = intime_oot_split(X_clean, data_with_macro, oot_start=new_date)

            #train/val/oot split 
            X_train, X_val, X_oot, y_train, y_val, y_oot = train_test(intime_df, oot_df)

            refitted_model = fit_model(X_train, X_val, y_train, y_val)

        else:
            raise ValueError('Changepoint in data is present, refit the model manually')




