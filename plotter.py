from pandas import DataFrame
from plotly.utils import pandas
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
import pandas as pd

dataset = pandas.read_csv('02:57PM on July 19, 2018.csv', engine='python')
old_data = pandas.read_csv('07:24PM on July 18, 2018.csv', engine='python', skipfooter=3)
scaler = MinMaxScaler(feature_range=(0, 1))
#dataset['total_cases'] = scaler.fit_transform(dataset['total_cases'])


def split_dataset_by_city(df):
    return df[df['city'] == 'iq'], df[df['city'] == 'sj']
print (dataset)

#////////////////////////////////////////////////////////////for san juan
iq, sj = split_dataset_by_city(dataset)
iq_old, sj_old = split_dataset_by_city(old_data)

T = np.array(sj.index)
power = np.array(sj['total_cases'])

xnew = np.linspace(T.min(),T.max(),20)



power_smooth = spline(T,power,xnew)

xnew_new = np.linspace(xnew.min(), xnew.max(), 260)

power_smooth_new = spline(xnew,power_smooth,xnew_new)

power_smooth_new = power_smooth_new.astype(int)

df_bbbb =   DataFrame(power_smooth_new)
#df_bbbb = df_bbbb.clip(lower=0)


#////////////////////////////////////////////////////

T = np.array(iq.index)
power = np.array(iq['total_cases'])

xnew = np.linspace(T.min(),T.max(),12)



power_smooth = spline(T,power,xnew)

xnew_new = np.linspace(xnew.min(), xnew.max(),156 )

power_smooth_new = spline(xnew,power_smooth,xnew_new)

power_smooth_new = power_smooth_new.astype(int)

df_bb = pd.concat([df_bbbb, DataFrame(power_smooth_new)])
df_bb = df_bb.reset_index(drop=True)
df_bb = df_bb.clip(lower=0)
dataset['total_cases'] = df_bb[0].values






#dataset[dataset['city'] == 'sj']['total_cases'].assign(df_bbbb[0])
#dataset.replace(dataset[dataset['city'] == 'sj']['total_cases'], df_bbbb[0])
#plt.plot(sj_old['total_cases'])

dataset.to_csv("sub.csv",index=False)


plt.plot(xnew_new,power_smooth_new)

plt.show()
