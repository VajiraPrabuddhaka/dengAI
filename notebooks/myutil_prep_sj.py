import scipy as sp
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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


def add_total_cases_feature(df):

    df['total_cases2'] = df['total_cases']

    df_X = df.iloc[:,:-2]
    df_total_cases2 = df.iloc[:,-1]
    df_y = df.iloc[:,-2:-1]
    df_X = pd.concat((df_X, df_total_cases2), axis=1)
    df = pd.concat((df_X, df_y), axis=1)
 
    return df


def convert_k_to_celsius(df):

    # convert all temperature measurements to Celsius
    df['reanalysis_dew_point_temp_k'] = df['reanalysis_dew_point_temp_k'] - 273.15 
    df['reanalysis_air_temp_k'] = df['reanalysis_air_temp_k'] - 273.15
    df['reanalysis_max_air_temp_k'] = df['reanalysis_max_air_temp_k'] - 273.15
    df['reanalysis_min_air_temp_k'] = df['reanalysis_min_air_temp_k'] - 273.15
    df['reanalysis_avg_temp_k'] = df['reanalysis_avg_temp_k'] - 273.15
    #df['reanalysis_tdtr_k'] = df['reanalysis_tdtr_k'] - 273.15
  
    return df

