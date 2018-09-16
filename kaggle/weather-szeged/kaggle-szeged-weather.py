# -*- coding: utf-8 -*-
"""
Spyder Editor

Weather in Szeged 2006-2016:
Is there a relationship between humidity and temperature? 
What about between humidity and apparent temperature? 
Can you predict the apparent temperature given the humidity?
"""

import pandas as pd
import numpy as np

data = pd.read_csv("weatherHistory.csv")

data.info()

data.isnull().sum()

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, 
                                       test_size=0.2, 
                                       random_state=35)

print("data has {} instances\n {} train instances\n {} test intances".
      format(len(data),len(train_set),len(test_set)))

train = train_set.copy()

import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(8,6))
plt.tight_layout()
plt.show()

train.columns = train.columns.str.replace(r"\(.*\)","")
train.columns = train.columns.str.rstrip() 
train.columns = train.columns.str.lower() 
train.columns = train.columns.str.replace(" ","_")

train = train.drop("loud_cover", axis=1)
train.info()

# 1- Is there a relationship between humidity and temperature? 
corr_matrix = train.corr()
corr_matrix["humidity"].sort_values(ascending=False)

# 2- What about between humidity and apparent temperature? 

corr_matrix["apparent_temperature"].sort_values(ascending=False)
  
# 3- Can you predict the apparent temperature given the humidity?
corr_matrix["temperature"].sort_values(ascending=False)


import seaborn as sns
#sns.heatmap(train.corr(), 
#            annot=True, fmt=".2f")  

columns = ["temperature", "apparent_temperature", "visibility","humidity","wind_speed","pressure"]
sns.pairplot(train[columns], diag_kind='hist')

# just to remind ...
# train_set, test_set = train_test_split(data, test_size=0.2, random_state=35)

train.formatted_date.value_counts()
sameday = train.formatted_date.str.contains(r'2010-08-02.*')
train.loc[sameday,"temperature"] = train.loc[sameday].temperature.mean()
train.loc[sameday,"apparent_temperature"] = train.loc[sameday].apparent_temperature.mean()
train.loc[sameday,"humidity"] = train.loc[sameday].humidity.mean()
train.loc[sameday,"wind_speed"] = train.loc[sameday].wind_speed.mean()
train.loc[sameday,"wind_bearing"] = train.loc[sameday].wind_bearing.mean()
train.loc[sameday,"visibility"] = train.loc[sameday].visibility.mean()
train.loc[sameday,"pressure"] = train.loc[sameday].pressure.mean()
train.loc[sameday,"daily_summary"] = "Partly cloudy starting in the afternoon contin..."
train.loc[sameday,"formatted_date"] = "2010-08-02 18:00:00.000 +0200"

train.loc[sameday]

humidity

train.summary.value_counts()

train.precip_type.value_counts()
train.daily_summary.value_counts()
