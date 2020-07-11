import keras
import numpy as np

param = dict(units=16,
             return_seq=True,
             return_state=False,
             stateful=True,
             dropout=0.2,
             lr=0.005,
             epoch=300,
             num_dev=1,
             time_divs=200,
             batch_size=1)


def rmsle_error(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def stateful_model(x):
    f = x.shape[-1]

    input_tensor = keras.layers.Input(batch_shape=(param['num_dev'], None, f), name='input')
    rnn = keras.layers.GRU(units=param['units'],
                           dropout=param['dropout'],
                           stateful=param['stateful'],
                           return_state=param['return_state'],
                           return_sequences=param['return_seq'],
                           name='rnn'
                           )(input_tensor)
    dense = keras.layers.Dense(units=1, activation='sigmoid')
    count_out = keras.layers.TimeDistributed(dense, name='output')(rnn)

    count_model = keras.Model(inputs=input_tensor, outputs=count_out)

    return count_model


def compile_model(x):
    model = stateful_model(x)
    model.compile(loss=rmsle_error,
                  optimizer=keras.optimizers.Adam(learning_rate=param['lr']))
    return model


class DataGenerator(keras.utils.Sequence):

    def __init__(self, x, y, batch_size=1, time_divs=100):
        self.x = x
        self.y = y
        self.time_divs = time_divs
        self.batch_size = batch_size
        self.seq_start = 0
        self.seq_end = self.x.shape[1] // self.time_divs

    def __len__(self):
        return int((np.ceil(self.x.shape[0] // float(self.batch_size))) * self.time_divs)

    def __getitem__(self, idx):
        tidx = idx % self.time_divs
        bidx = idx // self.time_divs

        start_bidx = bidx * self.batch_size
        end_bidx = (bidx + 1) * self.batch_size

        time_div_size = (self.seq_end - self.seq_start) // self.time_divs

        start_tidx = tidx * time_div_size + self.seq_start
        end_tidx = (tidx + 1) * time_div_size + self.seq_start

        if tidx == self.time_divs - 1:
            end_tidx = self.seq_end
            time_div_size = end_tidx - start_tidx

        x_out = self.x[start_bidx:end_bidx, :, :]
        x_out = x_out[:, start_tidx:end_tidx, :]

        y_out = self.y[start_bidx:end_bidx, :]
        y_out = y_out[:, start_tidx:end_tidx]

        self.seq_start = self.seq_start + time_div_size
        self.seq_end = self.seq_end + time_div_size

        return x_out, y_out


def train_stateful_model(x, y):
    model = compile_model(x)

    train_generator = DataGenerator(x=x,
                                    y=y,
                                    batch_size=param['batch_size'],
                                    time_divs=param['time_divs'])



    history = model.fit_generator(train_generator,
                                  epochs=param['epoch'],
                                  callbacks=[tensorboard_callback, batch_end_callback])
