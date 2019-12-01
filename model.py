from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers

def trim_dataset(data, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = data.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return data[:-no_of_rows_drop]
    else:
        return data

BATCH_SIZE = 100
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
x_temp, y_temp = get_timeseries(X_test, 3, 3)
x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)

lstm_model = Sequential()
lstm_model.add(LSTM(100, batch_input_shape=(100, 3, x_t.shape[2])))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(20,activation='relu'))
lstm_model.add(Dense(1,activation='sigmoid'))
optimizer = optimizers.RMSprop(lr=0.1)
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)

history = lstm_model.fit(x_t, y_t, epochs=50, verbose=2, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                    trim_dataset(y_val, 100)))

