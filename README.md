# dengAI
Predicting Disease Spread
## Introduction 
Dengue fever is a mosquito-borne life threatening disease that occurs mostly in tropical and subtropical regions of the world. Since the disease is carried by mosquitoes, the transmission dynamics of dengue are related to climate variables such as temperature and precipitation which might help in reproduction of mosquitoes. Although the relation between those are complex and many researchers say that the climate change has a considerable effect on distributional shifts of reported cases. Therefore, if there is a way to estimate the relationship between dengue dynamics and climate changes, the research initiatives and resource allocation to help fight life-threatening pandemics can be improved. The main consideration of this project is to predict the possible number of dengue cases for San Juan, Puerto Rico and Iquitos in Peru using data mining aspects. This task is available as a data science competition in www.drivedata.org website.

## About data
/data directory contains the data related to this model, followings are the attributes that they provided.


1) San Juan as sj or Iquitos as iq 
2) year : Year of the recorded value
3) weekofyear : Week of the year of recorded value
4) week_start_date : First date of the week
5) total_cases : Total dengue cases reported
6) ndvi_se : Normalized difference vegetation index  for south east side from city centroid
7) ndvi_sw : Normalized difference vegetation index  for south west side from city centroid
8) ndvi_ne : Normalized difference vegetation index  for north east side from city centroid
9) ndvi_nw : Normalized difference vegetation index  for north west side from city centroid 
10) precipitation_amt_mm : Total precipitation amount (PERSIANN)
11) reanalysis_sat_precip_amt_mm
12) reanalysis_air_temp_k : Mean air temperature (NCEP)
13) reanalysis_avg_temp_k : Average air temperature (NCEP)
14) reanalysis_dew_point_temp_k : Mean dew point temperature (NCEP)
15) reanalysis_max_air_temp_k : Maximum air temperature (NCEP)
16) reanalysis_min_air_temp_k : Minimum air temperature (NCEP)
17) reanalysis_precip_amt_kg_per_m2 : Total precipitation amount (NCEP)
18) reanalysis_relative_humidity_percent : Mean relative humidity (NCEP)
19) reanalysis_specific_humidity_g_per_kg : Mean specific humidity (NCEP)
20) reanalysis_tdtr_k : Diurnal temperature range (NCEP)
21) station_avg_temp_c : Average temperature (GHCN)
22) station_diur_temp_rng_c : Diurnal temperature range (GHCN)
23) station_max_temp_c : Maximum air temperature (GHCN)
24) station_min_temp_c : Minimum air temperature (GHCN)
25) station_precip_mm : Total precipitation (GHCN)

## Getting Started..

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

```angular2html
Linux pc with python3 installed 
keras with Tensorflow backend
scikit-learn toolkit for python
Other python libraries like numpy, matplotlib(for visualizations), pandas, skimage etc..
``` 
Please refer to the documentations of above tools and install latest versions that they supported. We recommand to install them in a seperate python virtual environment for convenience.
Please note that If you are willing to run `prophet.py` you should install prophet library as follows.
```angular2html
# bash
$ pip install fbprophet
```
For more details, refer to this: https://facebook.github.io/prophet/docs/installation.html

### Do simple Inference

#### To run Bayesian Ridge model
```angular2html
python3 final_regr.py
```
If you want to use ipython notebook
```angular2html
final_regr.ipynb
```
#### To run Time Series model
```angular2html
python3 prophet.py
```
