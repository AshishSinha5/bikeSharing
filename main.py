import pandas as pd
from modData import mod_data, min_max_normalize

train = pd.read_csv("./bike-sharing-demand/train.csv")
test = pd.read_csv("./bike-sharing-demand/test.csv")


train = mod_data(train)
test = mod_data(test)

std_cols = ['temp', 'atemp', 'humidity', 'windspeed']
pred_cols = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity',
             'windspeed', 'year', 'month', 'day', 'hour']
target = ['count']


train = min_max_normalize(train, std_cols, -1, 1)
test = min_max_normalize(test, std_cols, -1, 1)



