import pandas
import quandl

#Get data from server
df = quandl.get("WIKI/GOOGL")
#High - low percentage
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
#Daily change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
#Final data frame structure
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print(df.head())
