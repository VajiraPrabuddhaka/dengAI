import datetime

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

import notebooks.myutil_regr as myutil
from statsmodels.discrete.discrete_model import NegativeBinomial

import importlib
_ = importlib.reload(myutil)

dfx_train = myutil.get_indexed_dataset('data/dengue_features_train.csv')
dfy_train = myutil.get_indexed_dataset('data/dengue_labels_train.csv')
dfx_test = myutil.get_indexed_dataset('data/dengue_features_test.csv')
# combine training features with training labels for data exploration later on
dftrain = myutil.set_index(pd.merge(dfx_train, dfy_train))
dftrain.head()

#dftrain.dtypes
#dftrain.isnull().sum()
#dftest.isnull().sum()
# Will stack the train and test datasets to treat all NaN values together
# Need to add bogus total_cases column to test dataset so the files can be easily concatenated
# update total_cases = -1 to easily identify the records for later split data to original partitions
dfx_test['total_cases'] = -1
dfall = myutil.set_index(pd.concat((dftrain, dfx_test), axis=0))
dfall.sort_index(axis=0, inplace=True)
dfall.head()

dfall.isnull().sum()

dfall = myutil.set_nan_to_week_mean(dfall.copy())
dfall.isnull().sum()

#Split dataset
dfall_iq, dfall_sj = myutil.split_dataset_by_city(dfall)

# drop unnecessary columns
def drop_columns(df):
    df.drop(['city','year','week_start_date', 'ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','reanalysis_air_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c'], axis=1, inplace=True)
    return df

dfall_iq = drop_columns(dfall_iq.copy())
dfall_sj = drop_columns(dfall_sj.copy())

dftrain_iq = dfall_iq[dfall_iq['total_cases']>0].copy()     # total_cases was set to -1 for test partition
dftrain_sj = dfall_sj[dfall_sj['total_cases']>0].copy()     # total_cases was set to -1 for test partition

dftest_iq = dfall_iq[dfall_iq['total_cases']<0].copy()
dftest_sj = dfall_sj[dfall_sj['total_cases']<0].copy()
dftest_iq.drop('total_cases', axis=1, inplace=True)
dftest_sj.drop('total_cases', axis=1, inplace=True)


# split dataset into test and validation partitions
def prep_regr_run(city_data):
    X = city_data[:, :-1]
    y = city_data[:, -1]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.33, random_state=42)

    return X_train, X_valid, y_train, y_valid


def regr_run(nptrain, poly_degree=1, exploring=False):
    # create partitions for training
    X_train, X_valid, y_train, y_valid = prep_regr_run(nptrain)

    if poly_degree > 1:
        poly = PolynomialFeatures(poly_degree, interaction_only=True)
        X_train = poly.fit_transform(X_train)
        X_valid = poly.fit_transform(X_valid)

    if exploring: print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # regr = linear_model.Ridge(alpha = .5)
    # regr = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    # regr = linear_model.Lasso(alpha = .1)
    # regr = linear_model.LassoLars(alpha = .1)
    #regr = linear_model.BayesianRidge()
    #regr = SVR(kernel='rbf', C=1e3, gamma=0.1)
    print("fuck")
    print(X_train.shape)
    #regr = NegativeBinomial(y_train, X_train)
    print("fuck")
    # Train the model using the training sets
    regr.fit(X_train, y_train)
    #regr.fit()

    # Make predictions using the testing set
    y_pred = regr.predict(X_valid)

    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    #print("Mean absolute error: %.2f"
         # % mean_absolute_error(y_valid, y_pred))
    # Explained variance score: 1 is perfect prediction
    #print('Variance score: %.2f' % r2_score(y_valid, y_pred))

    return regr


# Get test dataset, create predictions and save them in the proper submission file format
def regr_predict_and_save(df_iq, model_iq, ts_iq, df_sj, model_sj, ts_sj, dftest_iq, dftest_sj, filename):
    nptest_iq = myutil.preprocess_test(df_iq.copy(), dftest_iq, ts_iq)
    nptest_sj = myutil.preprocess_test(df_sj.copy(), dftest_sj, ts_sj)

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

    dfresults.to_csv(filename, index=False)


periods_iq = 12   # best 12
degree_iq = 1     # best 1
nptrain_iq = myutil.preprocess(dftrain_iq.copy(), periods_iq)
regr_iq = regr_run(nptrain_iq, degree_iq, exploring=True)


periods_sj = 250   # best 250
degree_sj = 1      # best 1
nptrain_sj = myutil.preprocess(dftrain_sj.copy(), periods_sj)
regr_sj = regr_run(nptrain_sj, degree_sj, exploring=True)

regr_predict_and_save(dftrain_iq, regr_iq, periods_iq, dftrain_sj, regr_sj, periods_sj, dftest_iq, dftest_sj,\
                      datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")+".csv")