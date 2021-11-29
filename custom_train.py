import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

NbEpoch = 15

model1 = Sequential()
model1.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Convolution2D(32, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Convolution2D(64, (3, 3), activation='relu'))
model1.add(Convolution2D(64, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(64, activation='sigmoid'))
model1.add(Dropout(0.2))
model1.add(Dense(5, activation='softmax'))
model1.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

model1.summary()

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=180,
    zoom_range=0.3,
    horizontal_flip=True,
    rescale=1 / 255,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    rescale=0.1,
    fill_mode='nearest')

# trainning
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical')

label_map = (train_generator.class_indices)
print(label_map)

h = model1.fit(
    train_generator,
    epochs=NbEpoch,
    validation_data=validation_generator)

model1.save('custom_modele.h5')

plt.plot(h.history['val_accuracy'])
plt.plot(h.history['accuracy'])
plt.show()
