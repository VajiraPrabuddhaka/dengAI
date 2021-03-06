import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels as stats
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import theano 
import tensorflow as tf
import keras

def set_index(df):
    df['yearweekofyear'] = df[['year','weekofyear']].\
                           apply(lambda x: str(x[0]) + str(x[1]).zfill(2), axis=1)
    df.set_index('yearweekofyear', inplace=True)
    return df

def get_indexed_dataset(path):
    df = pd.read_csv(path)
    return set_index(df)

def split_dataset_by_city(df):
    return df[df['city']=='iq'], df[df['city']=='sj']

# will set nan to avarage value for the column for the given weekofyear
def set_nan_to_week_mean(df_with_nans):

    cityweek_mean = df_with_nans.groupby(['city','weekofyear']).mean().to_dict()
    # here's how we'd retrive mean ndvi_ne for city of iquito, week 1
    # cityweek_mean['ndvi_ne'][('iq',1)]  

    df_clean = df_with_nans.copy()
    
    # row is index to row
    # cols is series with series index = dataframe column name
    #                     series value = dataframe column value 
    #
    skip_columns = set(['city','weekofyear','week_start_date','total_cases'])

    # process iquito first
    where = df_with_nans['city'] == 'iq'
    for (row, cols) in df_with_nans[where].iterrows():
        for idx in cols.index:
            if pd.isnull(cols[idx]):
                 #print('In rows {}, found null for field {}'.format(row, idx))
                 if idx not in skip_columns: 
                      # there are no values for weekofyear = 53
                      week_of_year = min(52, cols['weekofyear'])
                      df_clean.loc[row, idx] = cityweek_mean[idx][('iq', week_of_year)]

    # process san juan
    where = df_with_nans['city'] == 'sj'
    for (row, cols) in df_with_nans[where].iterrows():
        for idx in cols.index:
            if pd.isnull(cols[idx]):
                 #print('In rows {}, found null for field {}'.format(row, idx))
                 if idx not in skip_columns: 
                      # there are no values for weekofyear = 53
                      week_of_year = min(52, cols['weekofyear'])
                      df_clean.loc[row, idx] = cityweek_mean[idx][('sj', week_of_year)]

    return df_clean


def convert_k_to_c(df):

    # convert all temperature measurements to Celsius
    df['reanalysis_dew_point_temp_k'] = df['reanalysis_dew_point_temp_k'] - 273.15 
    df['reanalysis_air_temp_k'] = df['reanalysis_air_temp_k'] - 273.15
    df['reanalysis_max_air_temp_k'] = df['reanalysis_max_air_temp_k'] - 273.15
    df['reanalysis_min_air_temp_k'] = df['reanalysis_min_air_temp_k'] - 273.15
    df['reanalysis_avg_temp_k'] = df['reanalysis_avg_temp_k'] - 273.15
    df['reanalysis_tdtr_k'] = df['reanalysis_tdtr_k'] - 273.15

    return df
 
# preprocess train data
def shift(npdata, shifts):

    if shifts == 0:
        return npdata 

    X = npdata[:,:-1]
    y = npdata[:,-1:]

    # shifts features one row at a time and pads them to the left of existing feature set
    # for each shift, a row is discarded at the bottom of the dataset
    feature_count = X.shape[1]

    # first shift always happens
    X = X[:-1,:feature_count] 

    for i in range(1, shifts):
        leftadd = X[:-1,:feature_count] 
        X = np.hstack((leftadd, X[1:,:]))

    return np.hstack((X, y[shifts:]))

