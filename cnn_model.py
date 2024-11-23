
from tensorflow import keras

def build_model(input_shape, strides=(2, 2)):
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    model = keras.models.Sequential()
    model.add(keras.layers.Input(input_shape))
    model.add(keras.layers.Conv2D(16, (4, 4), strides=strides, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
    model.add(keras.layers.Conv2D(32, (3, 3), strides=strides, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
    model.add(keras.layers.Conv2D(64, (3, 3), strides=strides, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
    model.add(keras.layers.Conv2D(128, (3, 3), strides=strides, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
    
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(256, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
    model.add(keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
    model.add(keras.layers.Dense(1, activation='linear', kernel_initializer=initializer, bias_initializer=initializer))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.0005), loss='mse', metrics=['mse'])
    return model