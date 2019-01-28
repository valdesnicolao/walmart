import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM


def lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer="rmsprop", metrics=['mape'])
    return model

def regression_model(X_train):
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer="rmsprop", metrics=['mape'])
    return model