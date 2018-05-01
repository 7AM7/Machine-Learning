import numpy as np
import pandas as pd
import quandl
import math ,datetime
from sklearn import preprocessing ,cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
#####

 #   STOCK MARKET
#####
 
## get data from quandl
df = quandl.get('WIKI/GOOGL')

## add column to df
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
## add new column to df is precent between high & close
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
## add new column to df is precent between close & open
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0


## now add importent column(features) to df
#       price            x          x           x
# price will be label
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume',]]

## forecast column
forecast_col = 'Adj. Close'
## replace any nan value with -99999
df.fillna(-99999,inplace=True)

## now get forecast out with 10% 
forecast_out = int(math.ceil(0.1*len(df)))

#print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out) ## this get data after forecast_out day and shifting them to first



## X is features all data without label
# X = np.array(df.drop(['label','Adj. Close'],1)) use price label
X = np.array(df.drop(['label'],1)) 

X=preprocessing.scale(X) ## scale features to low time

X_lately = X[-forecast_out:] #from -forecast_out to last data 10%
#print(X_lately)

X = X[:-forecast_out] #from first data to -forecast_out    90%
#print(X)


## Remove all nan value row or column
df.dropna(inplace=True)

## Y is Label
y = np.array(df['label'])

X_train,X_text,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

#classifier with LinearRegression with do N of jobs in one preprocessing
#clf = LinearRegression(n_jobs=-1) 


#train label and features  with X_train,y_train
# this step is trainning model with take alot of time and
# we don't want train model everytime or every ryn time.So will save trainning in file
# to save the train and get it fast every time.

#clf.fit(X_train,y_train)    
#with open('linearregression.pickle','wb') as f:
    # dump classifier in file
   #pickle.dump(clf,f)



## now we classifier model one time and save it after this we don't want classifier and train
## it again so we will load model eveytime from file 
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)



accuracy = clf.score(X_text,y_test) ## accuracy of test label and features
print("accuracy: ",accuracy)
forecast_set = clf.predict(X_lately)  # predict 10% of data
df['Forecast'] = np.nan # set forcast column with Nan VALUE

last_date = df.iloc[-1].name ## last row in df

last_unix = last_date.timestamp()
one_day = 86400  # all time in one unix day 
next_unix = last_unix + one_day
for i in forecast_set:
    ## convert to datetime(year,day,hour,sec) from timestamp
    next_date = datetime.datetime.fromtimestamp(next_unix)
    ## increment one day
    next_unix += 86400
    ## add nan data to row who have a day

    ## add all column nan without forecast column
    ## because we will make graph whith X and Y where Y is Price and price is forecast
    ## X is date 
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    

##create graph
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4) ## add plt info in the bottom
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
