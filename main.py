import pandas as pd
import numpy as np
from modData import mod_data, min_max_normalize, extract_date
from stateful_model import train_stateful_model
from predict import prediction

train_param = dict(
    norm=True,
)


train = pd.read_csv("./bike-sharing-demand/train.csv")
test = pd.read_csv("./bike-sharing-demand/test.csv")

train_date = extract_date(train)
test_date = extract_date(test)

train = mod_data(train)
test = mod_data(test)

std_cols = ['temp', 'atemp', 'humidity', 'windspeed']
pred_cols = ['season', 'holiday', 'workingday', 'weather',
             'temp', 'atemp', 'humidity', 'windspeed', 'year',
             'month', 'day', 'hour']
target = ['count']

if train_param['norm']:
    train = min_max_normalize(train, std_cols, -1, 1)
    test = min_max_normalize(test, std_cols, -1, 1)

X_train = np.asarray(train[pred_cols])
y_train = np.asarray(train[target])
X_train = np.expand_dims(X_train, axis=0)
y_train = np.expand_dims(y_train, axis=0)

X_test = np.asarray(test[pred_cols])
X_test = np.expand_dims(X_test, axis = 0)
history, model = train_stateful_model(X_train, y_train)

prediction(test_date, X_test, model)
