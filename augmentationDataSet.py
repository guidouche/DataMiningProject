
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


nImages = 30; # Nombre d'images générées
datagen = ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=180, width_shift_range=0.1,
    height_shift_range=0.1, brightness_range=None, shear_range=0.0, zoom_range=0.3,
    channel_shift_range=0.0, fill_mode='nearest')

img = load_img("./data/Source/lune/lune_1.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_1",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break


img = load_img("./data/Source/lune/lune_2.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_2",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_3.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_3",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_4.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_4",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_5.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_5",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_6.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_6",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_7.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_7",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_8.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_8",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_9.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_9",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_10.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_10",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_11.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_11",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_12.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_12",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_13.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_13",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_14.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_14",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_15.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_15",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_26.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_26",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_27.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_27",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_28.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_28",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_29.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_29",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break



img = load_img("./data/Source/lune/lune_30.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/train/lune",
                          save_prefix='_doublon_'+"lune_30",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break


img = load_img("./data/Source/lune/lune_31.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/validation/lune",
                          save_prefix='_doublon_'+"lune_31",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break


img = load_img("./data/Source/lune/lune_32.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/validation/lune",
                          save_prefix='_doublon_'+"lune_32",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break


img = load_img("./data/Source/lune/lune_33.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/validation/lune",
                          save_prefix='_doublon_'+"lune_33",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break

img = load_img("./data/Source/lune/lune_34.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/validation/lune",
                          save_prefix='_doublon_'+"lune_34",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break

img = load_img("./data/Source/lune/lune_35.png")
print(img)
# Mise en forme de l'image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Boucle de génération et d'enregistrement des images supplémentaires

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="./data/validation/lune",
                          save_prefix='_doublon_'+"lune_35",
                          save_format='jpeg'):
    i += 1
    print(i)
    if i >= nImages:
        break
