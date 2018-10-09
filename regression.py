# Regression

# Regression analisys - form of supervised
# machine learning. Performs by showing the machine features
# and then showing it what the correct answer is. After that
# machine is being tested by measuring the ration of correct
# answers.

import math
import datetime
#Data analysis lib.
import pandas as pd
#Sci-computing lib (N-dimensional array).
import numpy as np
#Financial data providing lib.
import quandl
#SciKit machine learning lib.
from sklearn import preprocessing, svm, linear_model
from sklearn.model_selection import train_test_split
#Graphing lib.
import matplotlib.pyplot as plt
from matplotlib import style

# By default quandl offers 50 api request a day. To expand requests
# number register at quandl.com and set your api key with
# quandl.ApiConfig.api_key = {your api key}
# and mark it with gitignoreline line commentary

# Get data from quandl server
df = quandl.get("WIKI/GOOGL")
# High - low percent.
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / \
  df['Adj. Low'] * 100.0
# Daily change percent.

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / \
  df['Adj. Open'] * 100.0
# Final data frame structure.
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Forecast column.
forecast_col = 'Adj. Close'
# Fill missing values with -99999.
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

# Features (regression term) - descriptive attributes.
# Label (regression term) - value you're attempting to predict.

# Standard ML code abbreviations.
# X(capital) = features.
X = np.array(df.drop(['label'], 1))
# Scale X to be in range of (-1, 1).
X = preprocessing.scale(X)
# Slice most recent features from array to predict against
X_recent = X[-forecast_out:]
X = X[:-forecast_out]

# Drop all NaN values from data frame.
df.dropna(inplace=True)

# y = label that corresponds to the features.
y = np.array(df['label'])

# Split data frame into training and test parts.
X_train, X_test, y_train, y_test = \
  train_test_split(X, y, test_size=0.2)

# Set style of graph.
style.use('ggplot')

# Add forecast col into data frame
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# Linear regression.
# n_jobs - number of threads to calculate with. Set to -1 to
# use all available threads.
clf = linear_model.LinearRegression(n_jobs=-1)

# Support vector regression.
# Set kernel to one of this ['linear', 'poly', 'rbf', 'sigmoid']
# clf = svm.SVR(gamma='auto', kernel='linear')

# Train classifier.
clf.fit(X_train, y_train)
# Test classifier.
confidence = clf.score(X_test, y_test)

# Predicted forecasts array
forecast_set = clf.predict(X_recent)
print("LR confidence = {}".format(confidence))
# Bind forecasts to days
for f in forecast_set:
  next_date = datetime.datetime.fromtimestamp(next_unix)
  next_unix += one_day
  df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [f]

# Build graph
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
