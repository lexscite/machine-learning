#Regression (regression analisys) - form of supervised
#machine learning. Performs by showing the machine features
#and then showing it what the correct answer is. After that
#machine is being tested by measuring the ration of correct
#answers.

import math
#Data analysis lib
import pandas
#Sci-computing lib (N-dimensional array)
import numpy
#Financial data providing lib
import quandl
#SciKit machine learning lib
import sklearn

#Get data from server
df = quandl.get("WIKI/GOOGL")
#High - low percent
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / \
    df['Adj. Low'] * 100.0
#Daily change percent
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) \
    / df['Adj. Open'] * 100.0
#Final data frame structure
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#Forecast column
#-99999 stands for missing data (NaN)
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

print(df.head())
