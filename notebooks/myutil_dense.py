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

def plot_cols(arr):
    plt.figure(figsize=(15,40))
    for i in np.arange(0, arr.shape[1]):
        plt.subplot(arr.shape[1], 1, i+1)
        plt.plot(arr[:, i])
        
def plot_cols2(df):
    # won't plot certain columns, string, constant
    plotcolumns = set(df.columns) - {'city', 'year', 'weekofyear', 'week_start_date'}
    i = 1
    plt.figure(figsize=(25,60))
    for column in plotcolumns: 
        s = df[column].apply(pd.to_numeric)
        plt.subplot(len(df.columns), 1, i)
        s.plot(kind='line', color='blue', lw=2)
        s.rolling(window=12, center=False).mean().plot(kind='line', color='red', lw=0.8)
        plt.title(column, y=0.8, loc='right', fontsize=18)
        #s.rolling(window=12, center=False).std().plot(kind='line', color='black', lw=0.8)
        i += 1

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


def init_preprocess(df, flagin):

    # create total_cases series to append to dataset 
    # series will be added as a feature
    s_total_cases2 = df['total_cases'].astype('float64')
    s_total_cases2.name = 'total_cases2'

    # remove nans
    df = set_nan_to_week_mean(df.copy())

    df_X = df.iloc[:,:-1]
    df_y = df.iloc[:,-1]
    df_X = pd.concat((df_X, pd.DataFrame(s_total_cases2)), axis=1)
    df = pd.concat((df_X, df_y), axis=1)

    # convert heavily skewed features to log scale
    df['station_precip_mm'] = np.log(df['station_precip_mm'])
    df['reanalysis_precip_amt_kg_per_m2'] = np.log(df['reanalysis_precip_amt_kg_per_m2'])
    df['precipitation_amt_mm'] = np.log(df['precipitation_amt_mm'])
    df.iloc[:,:-416]['total_cases2'] = np.log(df['total_cases2'])
    df.loc[df['station_precip_mm']<0, 'station_precip_mm'] = 0.0
    df.loc[df['reanalysis_precip_amt_kg_per_m2']<0, 'reanalysis_precip_amt_kg_per_m2'] = 0.0
    df.loc[df['precipitation_amt_mm']<0, 'precipitation_amt_mm'] = 0.0
    df.loc[df['total_cases2']<0, 'total_cases2'] = 0.0

    # drop irrelevant columns
    # Features “precipitation_amount_mm” and “reanalyzes_sat_precipi_amt_mm” were 
    #	found to be 100% correlated (pearson), so we dropped the latter.
    # Feature “week_start_date” was dropped since timescale is set by year and weekofyear features.
    # 
    df.drop(['week_start_date','reanalysis_sat_precip_amt_mm'], axis=1, inplace=True)

    # convert all temperature measurements to Celsius
    df['reanalysis_dew_point_temp_k'] = df['reanalysis_dew_point_temp_k'] - 273.15 
    df['reanalysis_air_temp_k'] = df['reanalysis_air_temp_k'] - 273.15
    df['reanalysis_max_air_temp_k'] = df['reanalysis_max_air_temp_k'] - 273.15
    df['reanalysis_min_air_temp_k'] = df['reanalysis_min_air_temp_k'] - 273.15
    df['reanalysis_avg_temp_k'] = df['reanalysis_avg_temp_k'] - 273.15
    df['reanalysis_tdtr_k'] = df['reanalysis_tdtr_k'] - 273.15

    # will not scale city, year and weekofyear
    cyw_arry = df.values[:,0:3]
    X = df.values[:,3:]

    # scale all features
    if flagin:
       # dataframe has flag
       y = X[:,-1].reshape(X.shape[0],1)
       X = X[:,:-1].astype('float32')
    else:
       X = df.values.astype('float32')

    scaler = MinMaxScaler(feature_range=(-1,1))
    X_scaled = scaler.fit_transform(X)

    # put back city, year and weekofyear
    X_scaled = np.concatenate((cyw_arry, X_scaled), axis=1) 

    if flagin: 
       df_arry = np.concatenate((X_scaled, y), axis=1) 
    else:
       df_arry = X_scaled   

    return pd.DataFrame(df_arry, columns=df.columns) 
 

# preprocess train data
def preprocess(df, timesteps=1):

    X = df.values[:,:-1].astype('float32')
    y = df.values[:,-1].reshape(X.shape[0],1)

    # shifts features one row at a time and pads them to the left of existing feature set
    feature_count = X.shape[1]
    X_scaled= X[:-1,:feature_count] 
    for i in range(1, timesteps):
        leftadd = X_scaled[:-1,:feature_count] 
        X_scaled = np.concatenate((leftadd, X_scaled[1:,:]), axis=1)

    return np.concatenate((X_scaled, y[timesteps:]), axis=1) 

# preprocess test data
def preprocess_test(df_train, df_test, timesteps=1):

    df_test_rowcount = df_test.shape[0]

    # will append training data, which preceeds test data in time
    # so we can create sequences using previous periods data for our predictions
    # just like we did during training 
    Xtrain = df_train.values[:,:-1].astype('float32')

    X = np.concatenate((Xtrain, df_test.values.astype('float32')), axis=0)
    scaler = MinMaxScaler(feature_range=(-1,1))
    X_scaled = scaler.fit_transform(X)

    # shifts features one row at a time and pads them to the left of existing feature set
    feature_count = X.shape[1]
    X_scaled= X_scaled[:-1,:feature_count] 
    for i in range(1, timesteps):
        leftadd = X_scaled[:-1,:feature_count] 
        X_scaled = np.concatenate((leftadd, X_scaled[1:,:]), axis=1)

    return X_scaled[-df_test_rowcount:,:] 

