# -*- coding: utf-8 -*-
"""
Weather Conditions in World War Two:
    Is there a relationship between the daily minimum and maximum temperature?
    Can you predict the maximum temperature given the minimum temperature?


                   
"""

import numpy as np
import pandas as pd 

# Importing the dataset
df_summary = pd.read_csv("Summary of Weather.csv")
df_station = pd.read_csv("Weather Station Locations.csv")

## Take a quick look at the data structure
# Exploring the data
print(df_summary.shape)
print(df_station.shape)
# Exploring the data columns
print(df_summary.info())
print(df_station.info())



# Exploring in df_summary the number of values of each 159 Stations: 
print(df_summary.STA.value_counts())
print(df_summary.STA.unique().size)

# Exploring in df_station the number of values of each 161 Stations: 
print(df_station.WBAN.value_counts())
print(df_station.WBAN.unique().size)

# Renaming the name of columns of the dataset of df_summary and df_station 
df_summary = df_summary.rename(columns={'STA': 'station'})
df_station = df_station.rename(columns={'WBAN': 'station'})

# Merging the both data frames by the station key
df = pd.merge(df_summary, df_station, on="station")

# Checking  
print(df.station.unique().size)

print(df.shape)
print(df_summary.shape)
print(df_station.shape)
print(df.info())
print(df.PoorWeather.unique())
print(df.TSHDSBRSGF.unique())

# Renaming the name of columns of the dataset of df_summary and df_station
df = df.rename(columns={'STATE/COUNTRY ID': 'country'})
df.info()

# Counting the 64 distincs countries:
df_station.country.unique().size


## First Histogram Before clean the Data
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(8,6))
plt.tight_layout()
plt.show()

## Cleaning the data

# Removing columns without significants values: 
df = df.drop(["Date","WindGustSpd","DR","SPD","SND","PGT","FT","FB","FTI","ITH","SD3","RHX","RHN","RVG","WTE","PoorWeather","TSHDSBRSGF"], axis=1)
print(df.info())

# Removing SNF - SnowFall
df = df.drop(["SNF"], axis=1)

# Removing SNF - NAME of Station
df = df.drop(["NAME"], axis=1)
print(df.info())



# Removing duplicate columns with Fahrenheit and PRCP values: 
df = df.drop(["MAX","MIN","MEA", "PRCP", "LAT","LON"], axis=1)

# Note that an elevation of 9999 means unknown, so we replace by NaN:
# So there is 3510 readings without Elevation
print(df.ELEV.isnull().sum())


df.ELEV = pd.to_numeric(df.ELEV, float)

print(df.ELEV.isnull().sum())

df.loc[df.station == 11501]
# Replacing by NaN:
df.loc[df.ELEV == 9999, "ELEV"] = np.nan
print(df.ELEV.isnull().sum())
df.ELEV = df.Snowfall.fillna(df.ELEV.mean())

print(df.isnull().sum())
print(df.info())
print(df.Snowfall.unique())

#def clean_columns(df_col):
#    unique_val = df_col.unique()
#    mean = unique_val.mean()    
    

df.Precip = pd.to_numeric(df.Precip, float)
df.Precip = df.Precip.fillna(0)
print(df.info())

df.Snowfall = pd.to_numeric(df.Snowfall, float)
print(df.Snowfall.isnull().sum())
df.Snowfall = df.Snowfall.fillna(0)
print(df.Snowfall.isnull().sum())

# Checking if there is any other NaN Value in the Dataset:
print(df.isnull().sum())

df.describe()
# 


## Histogram After cleaning the Data
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(8,6))
plt.tight_layout()
plt.show()


"""
# Cell Configuration for plotly over google colab
# This method pre-populates the outputframe with the configuration that Plotly
# expects and must be executed for every cell which is displaying a Plotly graph.
def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))

import numpy as np
import pandas as pd
# plotly online requires login and password
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

# basic functions to run on google colab
configure_plotly_browser_state()
init_notebook_mode(connected=False)

# create a generic layout
layout = go.Layout(width=1000,height=400)

# generate a histogram from column median_house_value 
trace = [go.Histogram(x=df["MaxTemp"], nbinsx=50)]

# create a figure and plot in notebook. 
# if you wish save the hmtl, change iplot to plot
fig = go.Figure(data=trace,layout=layout)
pyo.iplot(fig)
"""


df.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.5)
"""    
from mpl_toolkits.basemap import Basemap 
# Extract the data we're interested in
lat = df['Latitude'].values
lon = df['Longitude'].values
maxTemp = df['MaxTemp'].values
maxTemp_max = max(df['MaxTemp'].unique())
maxTemp_min = min(df['MaxTemp'].unique())   
# 1. Draw the map background
fig = plt.figure(figsize=(16, 12), edgecolor='w')

m = Basemap(projection='mill',lon_0=0)

m.drawcoastlines()
m.drawparallels(np.arange(-180,180,30),labels=[1,0,0,0])
m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,60),labels=[0,0,0,1])
m.drawcountries(color='gray')

# 2. scatter MaxTemp data, with color reflecting 
m.scatter(lon, lat, latlon=True,
          c=maxTemp, cmap='Reds', alpha=0.5)

# 3. create colorbar and legend
plt.colorbar(label='MaxTemp')
plt.clim(maxTemp_max, maxTemp_min)
"""
           
## Ploting with details     

# Extract the data we're interested in
lat = df['Latitude'].values
lon = df['Longitude'].values
maxTemp = df['MaxTemp'].values
maxTemp_max = max(df['MaxTemp'].unique())
maxTemp_min = min(df['MaxTemp'].unique())   
from mpl_toolkits.basemap import Basemap
from itertools import chain

# 1. Draw the map background          
def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    #m.etopo()
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')

fig = plt.figure(figsize=(18, 7), edgecolor='w')

m = Basemap(projection='cyl', resolution=None,
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, )
# 2. scatter MaxTemp data, with color reflecting 
m.scatter(lon, lat, latlon=True,
          c=maxTemp, cmap='Reds', alpha=0.5)
# 3. create colorbar and legend
plt.colorbar(label='MaxTemp')
plt.clim(maxTemp_max, maxTemp_min)
draw_map(m)
plt.show()
#"""


## Looking for Correlations 


# 1- Is there a relationship between the daily minimum and maximum temperature?
# 2- Can you predict the maximum temperature given the minimum temperature?

corr_matrix = df.corr()
corr_matrix["MaxTemp"].sort_values(ascending=False)
corr_matrix["MinTemp"].sort_values(ascending=False)


# Ploting correlation Matrix
import seaborn as sns
sns.heatmap(df.corr(), 
            annot=True, fmt=".2f")

# *italicized text*##  Visualizing Geographical Data

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, 
                                       test_size=0.2, 
                                       random_state=35)

print("data has {} instances\n {} train instances\n {} test intances".
      format(len(df),len(train_set),len(test_set)))


#df_summary.Date.value_counts()


train_set.info()

## Prepare the Data for Machine Learning Algorithms

# drop creates a copy of the remain data and does not affect train_set
# And drop the station key 
train_X = train_set.drop(["station","MaxTemp"], axis=1)

# copy the label (y) from train_set
train_y = train_set.MaxTemp.copy()



## Imputer


# First, you need to create an Imputer instance, specifying that you want 
# to replace each attribute’s missing values with the median of that attribute:
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")

# Since the median can only be computed on numerical attributes, we need to 
# create a copy of the data without the text attribute country:
train_X_num = train_X.drop(["country"], axis=1)

# Now you can fit the imputer instance to the training data using 
# the fit() method:
imputer.fit(train_X_num)


imputer.statistics_

train_X_num.median().values

# Now you can use this “trained” imputer to transform the training set by 
# replacing missing values by the learned medians:
train_X_num_array = imputer.transform(train_X_num)

# The result is a plain Numpy array containing the transformed features. 
# If you want to put it back into a Pandas DataFrame, it’s simple:
train_X_num_df = pd.DataFrame(train_X_num_array, columns=train_X_num.columns)

train_X_num_df.isnull().sum()





## Handling Text and Categorical Attributes


# For this, we can use Pandas' factorize() method which maps each 
# category to a different integer:

train_X.country.unique().size

train_X_cat_encoded, train_X_categories = train_X.country.factorize()

train_X_categories.size
# train_X_cat_encoded is now purely numerical
train_X_cat_encoded
train_X_cat_encoded.size
# Scikit-Learn provides a OneHotEncoder encoder to convert 
# integer categorical values into one-hot vectors.

from sklearn.preprocessing import OneHotEncoder 

encoder = OneHotEncoder()

# Numpy's reshape() allows one dimension to be -1, which means "unspecified":
# the value is inferred from the lenght of the array and the remaining
# dimensions
train_X_cat_1hot = encoder.fit_transform(train_X_cat_encoded.reshape(-1,1))

# it is a column vector
train_X_cat_1hot

train_X_cat_1hot.toarray().shape

import sys

print("Using a sparse matrix: {} bytes".format(sys.getsizeof(train_X_cat_1hot.toarray())))
print("Using a dense numpy array: {} bytes".format(sys.getsizeof(train_X_cat_1hot)))


### Feature Scaling

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


num_pipeline = Pipeline([('imputer', Imputer(strategy="median")),
                         ('std_scaler', StandardScaler())
                        ])
train_X_num_pipeline = num_pipeline.fit_transform(train_X_num)

train_X_num_pipeline


from sklearn.base import BaseEstimator, TransformerMixin

# This class will transform the data by selecting the desired attributes,
# dropping the rest, and converting the resulting DataFrame to a NumPy array.
class DataFrameSelector(BaseEstimator, TransformerMixin):
  def __init__(self, attribute_names):
    self.attribute_names = attribute_names

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return X[self.attribute_names].values


# Used to join two or more pipelines into a single pipeline
from sklearn.pipeline import FeatureUnion

# https://github.com/scikit-learn/scikit-learn/issues/10521
from future_encoders import OneHotEncoder

# numerical columns 
num_attribs = list(train_X_num.columns)

# categorical columns
cat_attribs = ["country"]

# pipeline for numerical columns
num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                         ('imputer', Imputer(strategy="mean")),
                         ('std_scaler', StandardScaler())
                        ])

# pipeline for categorical column
cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                         ('cat_encoder', OneHotEncoder(sparse=False))
                        ])

# a full pipeline handling both numerical and categorical attributes
full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),
                                               ("cat_pipeline", cat_pipeline)
                                              ])
    
    
# you can run the whole pipeline simply
train_X_prepared = full_pipeline.fit_transform(train_X)
train_X_prepared   
    
    
# Select and Train a Model with Linear Regression:    
    
from sklearn.linear_model import LinearRegression

# create a LinearRegression model
lin_reg = LinearRegression()

# fit it
lin_reg.fit(train_X_prepared, train_y)


# Done!! You now have a working Linear Regression Model.
# Let's try it out on a few instances from the trainning set.

# prepare the data
some_data = train_X.iloc[:10]
some_labels = train_y.iloc[:10]
some_data_prepared = full_pipeline.transform(some_data)

# make predictions
print("Predictions:", lin_reg.predict(some_data_prepared)) 

# Compare against the actual values:
print("Labels:", list(some_labels))


from sklearn.metrics import mean_squared_error

maxTemp_predictions = lin_reg.predict(train_X_prepared)
lin_mse = mean_squared_error(train_y, maxTemp_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse



from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_X_prepared, train_y)


# now that the model is trained, let's evaluate it on the training set

maxTemp_predictions = tree_reg.predict(some_data_prepared)
tree_mse = mean_squared_error(some_labels, maxTemp_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# Out[31]: 2.5961563077381001e-14
# !!! 


## Better Evaluation Using Cross-Validation

from sklearn.model_selection import cross_val_score

# create a LinearRegression model
lin_reg = LinearRegression()

scores = cross_val_score(lin_reg, 
                         train_X_prepared, 
                         train_y,
                         scoring="neg_mean_squared_error",
                         cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores

def display_scores(scores):
  print("Scores:", scores)
  print("Mean:", scores.mean())
  print("Standard deviation:", scores.std())

display_scores(rmse_scores)


## Ensemble Learning with RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

# create a RandomForestRegressor model
forest_reg = RandomForestRegressor()

# fit it
forest_reg.fit(train_X_prepared, train_y)

# predict the prepared data
maxTemp_predictions = forest_reg.predict(train_X_prepared)

forest_mse = mean_squared_error(train_y, maxTemp_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


from sklearn.model_selection import cross_val_score

forest_reg = RandomForestRegressor()

forest_scores = cross_val_score(forest_reg,
                                train_X_prepared, 
                                train_y,
                                scoring="neg_mean_squared_error", 
                                cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)



## Grid Search
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# hyperparameters values
# param_grid[0] - 12 combinations
# param_grid[1] - 6 combinations
param_grid = [{'n_estimators': [3, 10, 30], 
               'max_features': [2, 4, 6, 8]
              },
              {'bootstrap': [False], 
               'n_estimators': [3, 10],
               'max_features': [2, 3, 4]
              }
             ]

# create a randomforeestregressor model
forest_reg = RandomForestRegressor()


# run the grid search with cross validation
# (12 + 6) x 5 = 90 combinations
grid_search = GridSearchCV(forest_reg, 
                           param_grid, 
                           cv=5,
                           scoring='neg_mean_squared_error')

# see 90 combinations!!!
# it may take quite a long time
grid_search.fit(train_X_prepared, train_y)






grid_search.best_params_
#Out[42]: {'max_features': 8, 'n_estimators': 30}

grid_search.best_estimator_
'''
Out[43]: 
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
'''

# and of course, the evaluation scores are also available
cvres = grid_search.cv_results_
cvres

###  Analyze the Best Models and Their Errors

# can indicate the relative importance of each attribute 
# for making accurate predictions
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# categorical component of pipeline
cat_encoder = cat_pipeline.named_steps["country"]

# get the names
cat_one_hot_attribs = list(cat_encoder.categories_[0])

# all columns names
attributes = num_attribs 

sorted(zip(feature_importances, num_attribs), reverse=True)



# Presenting my solution:

# best model found in gridsearch step
final_model = grid_search.best_estimator_

# predictors and label
test_X = test_set.drop(["station","MaxTemp"], axis=1)
test_y = test_set["MaxTemp"].copy()

# prepared test's predictors
test_X_prepared = full_pipeline.transform(test_X)


final_predictions = final_model.predict(test_X_prepared)
final_mse = mean_squared_error(test_y, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)