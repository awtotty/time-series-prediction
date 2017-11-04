import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras

import string


# Fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
    
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    total_windows = len(series)-window_size

    for i in range(total_windows):
        X.append(series[i:i+window_size])

    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))

    return model


### return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    letters = string.ascii_letters
    punctuation = ['!', ',', '.', ':', ';', '?']

    text = [c for c in text if c in letters or c in punctuation or c is ' ']
    text = ''.join(text) # convert to string

    return text

### fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    i = 0

    while (i + window_size <= len(text)): # while window is within the text
        inputs.append(text[i:i+window_size]) 
        outputs.append(text[i+window_size]) 
        i += step_size

    return inputs,outputs

# build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))

    return model
