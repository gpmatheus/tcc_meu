
from tensorflow import keras

def build_model(input_shape, strides=(2, 2)):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(input_shape))
    model.add(keras.layers.Conv2D(16, (4, 4), strides=strides, activation='relu'))
    model.add(keras.layers.Conv2D(32, (3, 3), strides=strides, activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), strides=strides, activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), strides=strides, activation='relu'))
    
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model