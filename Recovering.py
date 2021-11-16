

from keras import backend as K
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model
import numpy as np

# Pour utiliser tensorflow
K.set_image_data_format('channels_last')

modele= load_model('modele_leNet.h5')


def reconize():
    img = load_img('img.jpeg', color_mode="grayscale", target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    return modele.predict(img)


