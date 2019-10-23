# Importing the libraries

import numpy as np
import pandas as pd
import pickle
import requests
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import json# Importing the dataset

df = pd.read_csv('/Users/tako/Desktop/repos/build week/zillow-prize-1/properties_2017.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values# Splitting the dataset into the Training set and Test set

df2 = df[['parcelid', 'bathroomcnt', 'fips', 'latitude', 'longitude', 
          'rawcensustractandblock', 'regionidcity', 'regionidcounty', 
          'regionidzip', 'yearbuilt', 'taxvaluedollarcnt', 'bedroomcnt']]

df = df2.dropna()

train = df.loc['0' : '1975773']
test = df.loc['1975774' : '2200000']
val = df.loc['2200001' : '2853766']

trainval = train
trainval_id = trainval['parcelid'].unique()
train_id, val_id = train_test_split(trainval_id, random_state=42)
train = trainval[trainval.parcelid.isin(train_id)]
target = 'taxvaluedollarcnt'
x_train = train.drop(columns=target)
x_val = val.drop(columns=target)
x_test = test.drop(columns=target)
y_test = test[target]
y_train = train[target]
y_val = val[target]
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
y_val_log = np.log1p(y_val)

encoder = ce.OrdinalEncoder()
xtre = encoder.fit_transform(x_train)
xve = encoder.transform(x_val)

x_test = encoder.transform(x_test)

eval_set = [(xtre, y_train_log), (xve, y_val_log)]

model = XGBRegressor(n_estimators=50, n_jobs=-1)
model.fit(xtre, y_train_log, eval_set=eval_set, eval_metric='rmse', early_stopping_rounds=50)

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4, 4, 96268]]))
