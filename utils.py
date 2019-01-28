import pandas as pd
import numpy as  np
from sklearn.preprocessing import OneHotEncoder

def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def drop_columns(df,lag,window):
    data=df.copy()
    columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['item', 'store']]
    for i in range(window, 0, -1):
        columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['item', 'store']]
    data.drop(labels=columns_to_drop, inplace=True, axis=1)
    data.rename({'store(t)':'store', 'item(t)':'item'}, inplace=True,axis='columns')
    return data

def expand_df(df):
    data = df.copy()
    data['weekend'] = np.int32(data.index.dayofweek > 3)
    return data

def convert_xy(df,lag):
    labels_col = 'sales(t+%d)' % lag
    X = df.drop(labels_col, axis=1)
    y = df[labels_col]
    store_ohe = OneHotEncoder()
    item_ohe = OneHotEncoder()
    X_store = pd.DataFrame(store_ohe.fit_transform(X.store.values.reshape(-1,1)).toarray())
    X_items = pd.DataFrame(item_ohe.fit_transform(X.item.values.reshape(-1,1)).toarray())
    X = X.drop(['store','item'], axis=1)
    X = np.concatenate([X,X_items,X_store],axis=1)
    y = y.values
    return X,y

def to_tensor(X):
    return X.reshape((X.shape[0],1,X.shape[1]))