
from tensorflow import keras

def get_2d_cnn(hp):
    #usedict argument for manually set, otherwise use hp
    if isinstance(hp, dict):
        learning_rate = hp['learning_rate']
    else:
        learning_rate = hp.Choice("learning_rate", values=[0.001, 0.0001])
    
    model = keras.Sequential(layers=[
        #referenced from https://www.youtube.com/watch?v=dOG-HxpbMSw
        #convulution 1
        keras.layers.Conv2D(32, (3,3), activation=keras.activations.relu),
        keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
        keras.layers.BatchNormalization(),
        #convolution 2
        keras.layers.Conv2D(32, (3,3), activation=keras.activations.relu),
        keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
        keras.layers.BatchNormalization(),
        #convolution 3
        keras.layers.Conv2D(32, (2,2), activation=keras.activations.relu),
        keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same'),
        keras.layers.BatchNormalization(),
        #flatten and feed into dense layer with dropiut
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dropout(0.3),
        #predictions
        keras.layers.Dense(6, activation=keras.activations.softmax)
        
    ])
    cce = keras.losses.categorical_crossentropy

    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=cce, metrics=['accuracy'])
    return model 