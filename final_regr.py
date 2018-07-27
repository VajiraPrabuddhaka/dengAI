import datetime
import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import statsmodels as stats
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

#%run /home/vajira/PycharmProjects/dengAI/notebooks/myutil_regr.py
from sklearn.svm import SVR
# from statsmodels.discrete.discrete_model import NegativeBinomial
import importlib
from sklearn.preprocessing import MinMaxScaler


def get_indexed_dataset(path):
    df = pd.read_csv(path)
    return set_index(df)

def set_index(df):
    df['yearweekofyear'] = df[['year','weekofyear']].\
                           apply(lambda x: str(x[0]) + str(x[1]).zfill(2), axis=1)
    df.set_index('yearweekofyear', inplace=True)
    return df


dfx_train = get_indexed_dataset('data/dengue_features_train.csv')
#dfx_train = dfx_train[503:]
dfy_train = get_indexed_dataset('data/dengue_labels_train.csv')
dfx_test = get_indexed_dataset('data/dengue_features_test.csv')
# combine training features with training labels for data exploration later on
dftrain = set_index(pd.merge(dfx_train, dfy_train))

#dftrain.dtypes
#dftrain.isnull().sum()
#dftest.isnull().sum()
# Will stack the train and test datasets to treat all NaN values together
# Need to add bogus total_cases column to test dataset so the files can be easily concatenated
# update total_cases = -1 to easily identify the records for later split data to original partitions
dfx_test['total_cases'] = -1
dfall = set_index(pd.concat((dftrain, dfx_test), axis=0))
dfall.sort_index(axis=0, inplace=True)
#dfall.head()


dfall.isnull().sum()


def set_nan_to_week_mean(df_with_nans):
    cityweek_mean = df_with_nans.groupby(['city', 'weekofyear']).mean().to_dict()
    # here's how we'd retrive mean ndvi_ne for city of iquito, week 1
    # cityweek_mean['ndvi_ne'][('iq',1)]

    df_clean = df_with_nans.copy()

    # row is index to row
    # cols is series with series index = dataframe column name
    #                     series value = dataframe column value
    #
    skip_columns = set(['city', 'weekofyear', 'week_start_date', 'total_cases'])

    # process iquito first
    where = df_with_nans['city'] == 'iq'
    for (row, cols) in df_with_nans[where].iterrows():
        for idx in cols.index:
            if pd.isnull(cols[idx]):
                # print('In rows {}, found null for field {}'.format(row, idx))
                if idx not in skip_columns:
                    # there are no values for weekofyear = 53
                    week_of_year = min(52, cols['weekofyear'])
                    df_clean.loc[row, idx] = cityweek_mean[idx][('iq', week_of_year)]

    # process san juan
    where = df_with_nans['city'] == 'sj'
    for (row, cols) in df_with_nans[where].iterrows():
        for idx in cols.index:
            if pd.isnull(cols[idx]):
                # print('In rows {}, found null for field {}'.format(row, idx))
                if idx not in skip_columns:
                    # there are no values for weekofyear = 53
                    week_of_year = min(52, cols['weekofyear'])
                    df_clean.loc[row, idx] = cityweek_mean[idx][('sj', week_of_year)]

    return df_clean

dfall = set_nan_to_week_mean(dfall.copy())
dfall.isnull().sum()

def split_dataset_by_city(df):
    return df[df['city']=='iq'], df[df['city']=='sj']

dfall_iq, dfall_sj = split_dataset_by_city(dfall)

# drop unnecessary columns
def drop_columns(df):
    df.drop(['city','year','week_start_date'], axis=1, inplace=True)
    return df

dfall_iq = drop_columns(dfall_iq.copy())
dfall_sj = drop_columns(dfall_sj.copy())

dftrain_iq = dfall_iq[dfall_iq['total_cases']>0].copy()     # total_cases was set to -1 for test partition
dftrain_sj = dfall_sj[dfall_sj['total_cases']>0].copy()     # total_cases was set to -1 for test partition

dftest_iq = dfall_iq[dfall_iq['total_cases']<0].copy()
dftest_sj = dfall_sj[dfall_sj['total_cases']<0].copy()
dftest_iq.drop('total_cases', axis=1, inplace=True)
dftest_sj.drop('total_cases', axis=1, inplace=True)


# preprocess train data

def preprocess(df, timesteps=1):
    ndvi_mean = df[['ndvi_nw', 'ndvi_ne', 'ndvi_se', 'ndvi_sw']].mean(axis=1)
    df.drop(['ndvi_nw', 'ndvi_ne', 'ndvi_se', 'ndvi_sw', 'reanalysis_air_temp_k', 'reanalysis_dew_point_temp_k',
                 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 
                 'precipitation_amt_mm', 'reanalysis_relative_humidity_percent',
                 'reanalysis_sat_precip_amt_mm'], axis=1, inplace=True)
    df.insert(1, 'ndvi_mean', ndvi_mean)
    print(df)
    # step 4: split array into features (starting at col 5) and labels
    X = df.values[:, :-1].astype('float32')
    y = df.values[:, -1].reshape(X.shape[0], 1)

    # step 5: normalize all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # shifts features one row at a time and pads them to the left of existing feature set
    feature_count = X.shape[1]
    X_scaled = X_scaled[:-1, :feature_count]
    for i in range(1, timesteps):
        leftadd = X_scaled[:-1, :feature_count]
        X_scaled = np.concatenate((leftadd, X_scaled[1:, :]), axis=1)

    return np.concatenate((X_scaled, y[timesteps:]), axis=1)


# preprocess test data
def preprocess_test(df_train, df_test, timesteps=1):
    df_test_rowcount = df_test.shape[0]

    # will append training data, which preceeds test data in time
    # so we can create sequences using previous periods data for our predictions
    # just like we did during training
    Xtrain = df_train.values[:, :-1].astype('float32')

    X = np.concatenate((Xtrain, df_test.values.astype('float32')), axis=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # shifts features one row at a time and pads them to the left of existing feature set
    feature_count = X.shape[1]
    X_scaled = X_scaled[:-1, :feature_count]
    for i in range(1, timesteps):
        leftadd = X_scaled[:-1, :feature_count]
        X_scaled = np.concatenate((leftadd, X_scaled[1:, :]), axis=1)
    # 200jul116
    return X_scaled[-df_test_rowcount:, :]


# split dataset into test and validation partitions
def prep_regr_run(city_data):
    X = city_data[:, :-1]
    y = city_data[:, -1]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.33, random_state=42)

    #return X_train, X_valid, y_train, y_valid
    return X, X_valid, y, y_valid


def regr_run(nptrain, poly_degree=1, exploring=False):
    # create partitions for training
    X_train, X_valid, y_train, y_valid = prep_regr_run(nptrain)

    if poly_degree > 1:
        poly = PolynomialFeatures(poly_degree, interaction_only=True)
        X_train = poly.fit_transform(X_train)
        X_valid = poly.fit_transform(X_valid)

    if exploring: print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

    # Create linear regression object
    #regr = linear_model.LinearRegression()
    # regr = linear_model.Ridge(alpha = .5)
    # regr = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    # regr = linear_model.Lasso(alpha = .1)
    # regr = linear_model.LassoLars(alpha = .1)
    regr = linear_model.BayesianRidge()
    # regr = SVR(kernel='rbf', C=1e3, gamma=0.1)
    print("fuck")
    print(X_train.shape)
    #regr = NegativeBinomial(y_train,X_train)

    # Train the model using the training sets
    regr.fit(X_train, y_train)
    # regr.fit()

    # Make predictions using the testing set
    y_pred = regr.predict(X_valid)

    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean absolute error: %.2f"
          % mean_absolute_error(y_valid, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_valid, y_pred))

    return regr


# Get test dataset, create predictions and save them in the proper submission file format
def regr_predict_and_save(df_iq, model_iq, ts_iq, df_sj, model_sj, ts_sj, dftest_iq, dftest_sj, filename):
    nptest_iq = preprocess_test(df_iq.copy(), dftest_iq, ts_iq)
    nptest_sj = preprocess_test(df_sj.copy(), dftest_sj, ts_sj)

    yhat_iq = model_iq.predict(nptest_iq)
    yhat_sj = model_sj.predict(nptest_sj)

    yhat_iq = yhat_iq.reshape(156, 1)
    yhat_sj = yhat_sj.reshape(260, 1)

    print(yhat_iq.shape)
    print(yhat_sj.shape)

    dfsubm = pd.read_csv('data/submission_format.csv')
    npsubm_sj = np.concatenate((dfsubm[dfsubm['city'] == 'sj'][['city', 'year', 'weekofyear']].values, \
                                yhat_sj.round().astype('int64')), axis=1)
    npsubm_iq = np.concatenate((dfsubm[dfsubm['city'] == 'iq'][['city', 'year', 'weekofyear']].values, \
                                yhat_iq.round().astype('int64')), axis=1)
    dfresults = pd.DataFrame(np.concatenate((npsubm_sj, npsubm_iq), axis=0), columns=dfsubm.columns)

    dfresults['total_cases'] = dfresults['total_cases'].clip(lower=0)
    #dfresults[dfresults['city'] == 'sj'] += 17

    dfresults.to_csv(filename, index=False)


periods_iq = 1   # best 12
degree_iq = 1     # best 1
# print (dfall_iq)
nptrain_iq = preprocess(dftrain_iq.copy(), periods_iq)
dftest_iq = preprocess(dftest_iq.copy(), periods_iq)
regr_iq = regr_run(nptrain_iq, degree_iq, exploring=True)

periods_sj = 1   # best 250
degree_sj = 1      # best 1
nptrain_sj = preprocess(dftrain_sj.copy(), periods_sj)
dftest_sj = preprocess(dftest_sj.copy(), periods_sj)
regr_sj = regr_run(nptrain_sj, degree_sj, exploring=True)

regr_predict_and_save(dftrain_iq, regr_iq, periods_iq, dftrain_sj, regr_sj, periods_sj, dftest_iq, dftest_sj,\
                      datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")+".csv")