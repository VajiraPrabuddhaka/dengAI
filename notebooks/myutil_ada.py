import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

    # update total case outliers
    df['total_cases2'] = df[df['city']=='iq']['total_cases'].apply\
                         (lambda x: np.where(x>40,40.0,x/1.0))
    df['total_cases2'] = df[df['city']=='sj']['total_cases'].apply\
                         (lambda x: np.where(x>150,150.0,x/1.0))

    # remove nans
    df = set_nan_to_week_mean(df.copy())

    df_X = df.iloc[:,:-2]
    df_total_cases2 = df.iloc[:,-1]
    df_y = df.iloc[:,-2:-1]
    df_X = pd.concat((df_X, df_total_cases2), axis=1)
import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels as stats
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

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

# preprocess train data
def preprocess(df, do_minmax, timesteps):

    X = df.values
    no_features = X.shape[1]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler_tc = MinMaxScaler(feature_range=(0,1))

    if do_minmax:
        # scale total_cases
        scaler_tc.fit(X[:,-1:])

    # first shift always happens
    X = X[:-1,:]

    for i in range(1, timesteps):
        addleft = X[:-1,:no_features]
        X = np.hstack((addleft, X[1:,:]))

    if do_minmax:
        # scale all features
        scaler.fit(X)
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    return X_scaled, scaler, scaler_tc


# preprocess train data
def preprocess_test(df, scaler, timesteps=1):

    X = df.values
    no_features = X_scaled.shape[1]

    # scale all features
    #X_scaled = scaler.transform(X)
    scaler = MinMaxScaler(feature_range=(0,1))
    X_scaled = scaler.fit_transform(X)

    # first shift always happenj
    X_scaled = X_scaled[:-1,:] 

    for i in range(1, timesteps):
        addleft = X_scaled[:-1,:no_features] 
        X_scaled = np.hstack((addleft, X_scaled[1:,:]))

    return X_scaled


    X_scaled= X_scaled[:-1,:feature_count] 
    for i in range(1, timesteps):
        leftadd = X_scaled[:-1,:feature_count] 
        X_scaled = np.concatenate((leftadd, X_scaled[1:,:]), axis=1)

    return X_scaled[-df_test_rowcount:,:] 

