import keras
import numpy as np
import tensorflow as tf
import time
param = dict(units=32,
             return_seq=True,
             return_state=False,
             stateful=True,
             dropout=0.2,
             lr=0.005,
             epoch=50,
             num_dev=1,
             time_divs=100,
             batch_size=1,
             log_dir='./logs/',
             verbose=1,
             rnn_activation='tanh',
             dense_activation='relu',
             model_name='statefulGRU',
             model_save_dir='./models/'
             )


def rmsle_error(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    return tf.math.sqrt(tf.math.reduce_mean((tf.math.log1p(y_pred) - tf.math.log1p(y_true))**2, axis=1))


def stateful_model(x):
    f = x.shape[-1]

    input_tensor = keras.layers.Input(batch_shape=(param['num_dev'], None, f), name='input')
    rnn = keras.layers.GRU(units=param['units'],
                           dropout=param['dropout'],
                           stateful=param['stateful'],
                           return_state=param['return_state'],
                           return_sequences=param['return_seq'],
                           activation=param['rnn_activation'],
                           name='rnn'
                           )(input_tensor)
    dense = keras.layers.Dense(units=1, activation=param['dense_activation'])
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
        self.seq_end = self.x.shape[1]

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

        y_out = self.y[start_bidx:end_bidx, :, :]
        y_out = y_out[:, start_tidx:end_tidx, :]

        return x_out, y_out


def train_stateful_model(x, y):
    model = compile_model(x)
    keras.utils.plot_model(model,show_shapes=True)
    train_generator = DataGenerator(x=x,
                                    y=y,
                                    batch_size=param['batch_size'],
                                    time_divs=param['time_divs'])

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=param['log_dir']+ param['model_name'] + time.strftime("%Y%m%d-%H%M%S"))

    class CustomCallback(keras.callbacks.Callback):

        def __init__(self, time_divs):
            super().__init__()
            self.train_time_divs = time_divs
            self.batch_idx = 0

        def on_batch_begin(self, batch, logs=None):
            if self.batch_idx % self.train_time_divs == 0:
                print("Resetting States")
                self.model.reset_states()
            self.batch_idx = self.batch_idx + 1

    batch_end_callback = CustomCallback(param['time_divs'])
    
    history = model.fit_generator(train_generator,
                                  epochs=param['epoch'],
                                  callbacks=[tensorboard_callback, batch_end_callback],
                                  verbose=param['verbose'])

    model.save_weights(param['model_save_dir']+param['model_name']+time.strftime("%Y%m%d-%H%M%S"))

    return history, model
