import pandas as pd
import numpy as  np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from utils import *
from model import *


epochs = 10
batch_size = 250
lr = 1e-3
path='data/kaggle/'
df_train = pd.read_csv(path +'train.csv', index_col=0)
df_train.index = pd.to_datetime(df_train.index)

windows = range(1,3)
lags = [1]
for window,lag in [(w,v) for w in windows for v in lags]:
    print('--------------------------------------------------------------------')
    print('LSTM, window : {0}, lag : {1}'.format(window,lag))
    df_train_supervised = series_to_supervised(df_train, window=window, lag=lag)
    df_train_supervised = expand_df(df_train_supervised)
    df_train_supervised = drop_columns(df_train_supervised,lag,window)
    X,y = convert_xy(df_train_supervised,lag)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_test,y_test, test_size=0.5)

    print('Train set shape', X_train.shape)
    print('Test set shape', X_test.shape)
    print('Validation set shape', X_val.shape)

    X_train=to_tensor(X_train)
    X_test=to_tensor(X_test)
    X_val=to_tensor(X_val)

    model = lstm_model(X_train)
    history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=epochs, verbose=0)
    res=model.evaluate(X_test, y_test)
    print('LSTM, window : {0}, lag : {1}, MSE : {2}, MAPE : {3}'.format(window,lag,res[0],res[1]))
    model.save('models/lstm_simple_'+str(window)+'_'+str(lag)+'.h5')

