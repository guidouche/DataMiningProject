
from keras import backend as K
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model

# Pour utiliser tensorflow
from keras.utils import np_utils

K.set_image_data_format('channels_last')

modele= load_model('custom_modele.h5')


def reconize():
    img = load_img('img.jpeg', color_mode="rgb", target_size=(150, 150))
    #img = load_img('img.jpeg', color_mode="grayscale", target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    prediction = modele.predict(img)
    print(prediction)

    return prediction


