import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from keras import backend as K

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
model1 = Sequential()

# passage de 32*32 en 28*28 avec 6 noyeau
model1.add(Convolution2D(6, (5, 5), padding='same', input_shape=(28, 28, 1)))
# passage de 28*28 a 14*14
model1.add(MaxPooling2D())
# passage de 14*14 a 10*10
model1.add(Convolution2D(16, (5, 5)))
# passage de 10*10 a 5*5
model1.add(MaxPooling2D())

model1.add(Flatten())

# Couche c5 120 noyeau
model1.add(Dense(120, activation='sigmoid'))
# Couche c6 84 noyeau
model1.add(Dense(84, activation='sigmoid'))
# Couche de sortie 10 noyeau
model1.add(Dense(10, activation='softmax'))

model1.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

model1.summary()

h1 = model1.fit(X_train, Y_train,
                batch_size=32, epochs=5, verbose=1, validation_data=(X_test, Y_test))
model1.save('modele_TP2_leNet.h5')
print(h1.history['val_accuracy'])
print(h1.history['accuracy'])
plt.plot(h1.history['val_accuracy'])
plt.plot(h1.history['accuracy'])
plt.show()
