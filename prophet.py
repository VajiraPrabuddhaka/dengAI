import pandas as pd
from fbprophet import Prophet


def split_dataset_by_city(df):
    return df[df['city']=='iq'], df[df['city']=='sj']

# Python
df_features = pd.read_csv('data/dengue_features_train.csv')
df_labels = pd.read_csv('data/dengue_labels_train.csv')


iq_features, sj_features = split_dataset_by_city(df_features)
iq_labels, sj_labels = split_dataset_by_city(df_labels)

df_new_iq = pd.concat([iq_features['week_start_date'], iq_labels['total_cases']], ignore_index=True, axis=1)
df_new_sj = pd.concat([sj_features['week_start_date'], sj_labels['total_cases']], ignore_index=True, axis=1)


df_new_iq.columns , df_new_sj.columns = ['ds', 'y'], ['ds', 'y']
df_new_iq, df_new_sj = df_new_iq.reset_index(drop=True), df_new_sj.reset_index(drop=True)

df_test = pd.read_csv('data/dengue_features_test.csv')
df_test_iq, df_test_sj = split_dataset_by_city(df_test)

df_test_iq, df_test_sj = df_test_iq.reset_index(drop=True), df_test_sj.reset_index(drop=True)
df_test_iq, df_test_sj = df_test_iq['week_start_date'], df_test_sj['week_start_date']

df_test_iq, df_test_sj  = df_test_iq.to_frame(), df_test_sj.to_frame()
df_test_iq.columns , df_test_sj.columns = ['ds'], ['ds']
df_test_iq, df_test_sj = df_test_iq.reset_index(drop=True), df_test_sj.reset_index(drop=True)



m_iq = Prophet()
m_sj = Prophet()
m_iq.fit(df_new_iq)
m_sj.fit(df_new_sj)


forecast_iq = m_iq.predict(df_test_iq)
forecast_sj = m_sj.predict(df_test_sj)

df_sub = pd.concat([forecast_sj['yhat'], forecast_iq['yhat']])
df_sub = df_sub.reset_index(drop=True)
df_sub = df_sub.to_frame()
df_sub.columns = ['total_cases']
# df_sub = df_sub.rename(columns={0: 'total_cases'})

df_sub_format = pd.read_csv('data/submission_format.csv')
# df_sub_format.drop(['total_cases'], inplace=True, axis=1)
# df_sub_format.insert(column='total_cases', value=df_sub['total_cases'])
df_sub_format.update(df_sub)
df_sub_format['total_cases'] = df_sub_format['total_cases'].round(0).astype(int)
df_sub_format['total_cases'] = df_sub_format['total_cases'].clip(lower=0)
df_sub_format.to_csv('submission_timeseries.csv', index=False)

fig1 = m_iq.plot(forecast_iq)
fig1.show()

print ('dvdv')


