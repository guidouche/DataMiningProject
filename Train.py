import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from keras import backend as K

def train(nbEpoch):

    K.set_image_data_format('channels_last')

    # On charge MNIST
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32') / 255

    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test = X_test.astype('float32') / 255

    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test = np_utils.to_categorical(Y_test, 10)

    # Modèle : LENET
    # 1 couche cachée de 32 neurones (fully connected)
    model = Sequential()

    # passage de 32*32 en 28*28 avec 6 noyeau
    model.add(Convolution2D(6, (5, 5), padding='same', input_shape=(28, 28, 1)))
    # passage de 28*28 a 14*14
    model.add(MaxPooling2D())
    # passage de 14*14 a 10*10
    model.add(Convolution2D(16, (5, 5)))
    # passage de 10*10 a 5*5
    model.add(MaxPooling2D())

    model.add(Flatten())

    # Couche c5 120 noyeau
    model.add(Dense(120, activation='sigmoid'))
    # Couche c6 84 noyeau
    model.add(Dense(84, activation='sigmoid'))
    # Couche de sortie 10 noyeau
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

    model.summary()

    h = model.fit(X_train, Y_train,
                    batch_size=32, epochs=nbEpoch, verbose=1, validation_data=(X_test, Y_test))

    model.save('modele_leNet.h5')
    plt.plot(h.history['val_accuracy'])
    plt.plot(h.history['accuracy'])
    plt.show()

