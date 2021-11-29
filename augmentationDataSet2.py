from os import listdir
from os.path import isfile, join
import random

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def EraseFile(repertoire):
    import os
    for filename in os.listdir(repertoire):
        os.remove(repertoire + "/" + filename)


def duplication(src, folder):
    img = load_img(src)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=folder,
                              save_prefix="_",
                              save_format='jpeg'):
        i += 1
        if i >= nbDoublons:
            break


monRepertoire = "./data/Source/"

fichiers = [f for f in listdir(monRepertoire)]

print(fichiers)
# Nombre de photos généreés pour chaque fichier source
nbDoublons = 30

datagen = ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=180, width_shift_range=0.1,
    height_shift_range=0.1, brightness_range=None, shear_range=0.0, zoom_range=0.3,
    channel_shift_range=0.0, fill_mode='nearest')

for f in fichiers:
    photos = [p for p in listdir(join(monRepertoire, f))]
    nbPhotos = photos.__len__()
    # On fait du 60% train 40% valdiation :
    nbtrain = round(60 * nbPhotos / 100)
    nbValidation = nbPhotos - nbtrain
    random.shuffle(photos)
    _save_to_dir = "./data/" + "train" + "/" + f
    EraseFile(_save_to_dir)
    for x in range(nbtrain):
        duplication(join(monRepertoire, f, photos[x]), _save_to_dir)
    print("Duplication du répertoire " + f + " vers train effectuée.")
    _save_to_dir = "./data/" + "validation" + "/" + f
    EraseFile(_save_to_dir)
    for x in range(nbtrain, nbtrain + nbValidation):
        duplication(join(monRepertoire, f, photos[x]), _save_to_dir)
    print("Duplication du répertoire " + f + " vers validation effectuée.")
