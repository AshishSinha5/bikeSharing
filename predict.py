import keras
import pandas as pd
import numpy as np
import time

predict_param = dict(
    result_file='./results/',
    model_name='stateful_gru',
)


def prediction(date_time, x, model):
    y = model.predict(x)
    y = y[0, :, 0]

    res = pd.DataFrame(columns=['datetime', 'count'])
    res['datetime'] = date_time
    res['count'] = y

    res.to_csv(predict_param['result_file'] + predict_param['model_name'] + time.strftime("%Y%m%d-%H%M%S"))
