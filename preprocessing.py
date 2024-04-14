import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings

from sklearn.model_selection import train_test_split


def preprocess_data(data):
    """
    Preprocess data: add macro features and tax days
    In: dataset
    Out: dataset with added features
    """
    data['Date_short'] =  data['Date'].apply(lambda x: x[:7])
    macro_data1 = pd.read_excel('Инфляция и ключевая ставка Банка России_F27_07_2013_T23_01_2023.xlsx'
                            , sheet_name=0, converters={'Дата': str})\
                            .rename(columns={'Ключевая ставка, % годовых': 'key_rate', 'Инфляция, % г/г': 'infl_rate'})
    macro_data1['Date_short'] = macro_data1['Дата']\
    .apply(lambda x: x.split('.')[1]+'-'+ x.split('.')[0] if len(x.split('.')[0]) == 2 else x.split('.')[1]+'-0'+ x.split('.')[0])

    data_with_macro= data.merge(macro_data1.drop(['Дата', 'Цель по инфляции'], axis=1), on='Date_short', how='left')
    data_with_macro['day']= data_with_macro['Date'].apply(lambda x: int(x[8:10]))

    data_with_macro['tax_day'] = 0
    data_with_macro.loc[data_with_macro['day'] == 28, 'tax_day'] = 1

    return data_with_macro 

def primary_feature_filtering(df, nans_thrs = 0.5, std_thrs = 0):
    """
    Filter features
    In: dataset with generated features
    Out: clean dataset without constant features and many nulls
    """
    features = df.columns
    shape = df.shape[0]
    excluded_by_nans = []
    excluded_by_std = []

    for f in features:
        if df[f].isnull().sum() / shape > nans_thrs:
            excluded_by_nans.append(f)
        if df[f].std() <= std_thrs:
            excluded_by_std.append(f)

    clean_df = df.drop(excluded_by_std + excluded_by_nans, axis=1)
    print(f'Num cols excluded by null share: {len(excluded_by_nans)}')
    print(f'Num cols excluded by standard deviation: {len(excluded_by_std)}')
    print(f'Clean df shape: {clean_df.shape}')
    
    return excluded_by_nans, excluded_by_std, clean_df

def generate_features(data):
    """
    Generate features from dataset with TSFresh library and get clean dataset with some features eliminated by primary_feature_filtering
    In: dataset
    Out: dataset with new features
    """
    settings_efficient = settings.EfficientFCParameters()
    data_clean = data.drop(['Date', 'Outcome', 'Income', 'Date_short', 'day', 'tax_day'], axis=1)
    data_long = pd.DataFrame({0: data_clean.values.flatten(),
                          1: np.arange(data_clean.shape[0]).repeat(data_clean.shape[1])})
    X = extract_features(data_long, column_id=1, impute_function=impute, default_fc_parameters=settings_efficient)
    excluded_by_nans, excluded_by_std, X_clean = primary_feature_filtering(X, nans_thrs = 0.5, std_thrs = 0.1)
    return X_clean

def intime_oot_split(X_clean, data_with_macro, oot_start):
    X_clean['tax_day'] = data_with_macro['tax_day']
    X_clean['date'] = data_with_macro['Date']
    df_clean_with_target = X_clean.copy(deep=True)
    df_clean_with_target['target'] = data_with_macro['Balance']

    oot_start = '2021-01-01'
    intime_df = df_clean_with_target[df_clean_with_target['date'] < oot_start]
    oot_df = df_clean_with_target[df_clean_with_target['date'] >= oot_start]
    return intime_df, oot_df


def train_test(intime_df, oot_df):
    """
    Create train/val/oot datasets (without shuffling) 
    In: 
        X_clean: dataset with generated features
        data_with_macro: dataset before preprocessing 
        oot_start: date from which out-of-time starts 

    Out: X_train, X_val, X_oot, y_train, y_val, y_oot 
    """
    
    X_train, X_val, y_train, y_val = train_test_split(intime_df.drop(['target', 'date'], axis=1), intime_df['target'], test_size=0.3, shuffle=False)

    X_oot = oot_df.drop(['target', 'date'], axis=1)
    y_oot = oot_df['target']

    return X_train, X_val, X_oot, y_train, y_val, y_oot

def train_test_for_prophet(intime_df, oot_df):
    """
    Create train/val/oot datasets (without shuffling) for prophet model 
    They contain target and date (y and ds cols, respectively) as well as features 
    In: 
        X_clean: dataset with generated features
        data_with_macro: dataset before preprocessing 
        oot_start: date from which out-of-time starts 

    Out: df_train_for_prophet, df_val_for_prophet, df_oot_for_prophet
    """

    df_train_for_prophet, df_val_for_prophet, _, _ = train_test_split(intime_df, intime_df['target'], test_size=0.3, shuffle=False)
    df_train_for_prophet = df_train_for_prophet.rename(columns={'date': 'ds', 'target': 'y'})
    df_val_for_prophet = df_val_for_prophet.rename(columns={'date': 'ds', 'target': 'y'})
    df_oot_for_prophet = oot_df.drop('target', axis=1).rename(columns={'date': 'ds'})

    return df_train_for_prophet, df_val_for_prophet, df_oot_for_prophet
