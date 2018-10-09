#Regression (regression analisys) - form of supervised
#machine learning. Performs by showing the machine features
#and then showing it what the correct answer is. After that
#machine is being tested by measuring the ration of correct
#answers.

import math
#Data analysis lib.
import pandas
#Sci-computing lib (N-dimensional array).
import numpy
#Financial data providing lib.
import quandl
#SciKit machine learning lib.
from sklearn import preprocessing, svm, linear_model
from sklearn.model_selection import train_test_split

#By default quandl offers 50 api request a day. To expand requests
#number register at quandl.com and set your api key with
#quandl.ApiConfig.api_key = {your api key}
#and mark it with gitignoreline line commentary

#Get data from quandl server
df = quandl.get("WIKI/GOOGL")
#High - low percent.
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / \
    df['Adj. Low'] * 100.0
#Daily change percent.

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / \
    df['Adj. Open'] * 100.0
#Final data frame structure.
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#Forecast column.
forecast_col = 'Adj. Close'
#Fill missing values with -99999.
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

#Drop all NaN values from data frame.
df.dropna(inplace=True)

#Features (regression term) - descriptive attributes.
#Label (regression term) - value you're attempting to predict.

#Standard ML code abbreviations.
#X(capital) = features.
#y = label that corresponds to the features.
X = numpy.array(df.drop(['label'], 1))
#Scale X to be in range of (-1, 1).
X = preprocessing.scale(X)
y = numpy.array(df['label'])

#Split data frame into training and test parts.
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2)

#Linear regression.
clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("Linear regression confidence = {}".format(confidence))

#Support vector regression.
for k in ['linear', 'poly', 'rbf', 'sigmoid']:
  clf = svm.SVR(gamma='auto', kernel=k)
  clf.fit(X_train, y_train)
  confidence = clf.score(X_test, y_test)
  print("SRV ({}) confidence = {}".format(k, confidence))
