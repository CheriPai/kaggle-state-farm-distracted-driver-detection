import numpy as np
from vgg16 import VGG16
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

img_size = 224
batch_size = 64
dropout = 0.5

lr = 0.0001
decay = 0
epochs = 5

TRAIN_PATH = "data/train"
VALID_PATH = "data/valid"
TEST_PATH = "data/pseudo"
# TRAIN_PATH = "data/sample/train"
# VALID_PATH = "data/sample/valid"


gen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0,
    zoom_range=0.15,
    channel_shift_range=6,
)
train = gen.flow_from_directory(
    TRAIN_PATH,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True)
valid = gen.flow_from_directory(
    VALID_PATH,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False)
pseudo = gen.flow_from_directory(
    TEST_PATH,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    classes=["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"],
    class_mode="categorical",
    shuffle=False)

vgg = VGG16(dropout)
vgg.finetune_dense(10)

vgg.model.compile(
    optimizer=Adam(lr=lr, decay=decay),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

vgg.model.load_weights("data/weights.h5")

pseudo.__dict__["classes"] = np.argmax(vgg.model.predict_generator(pseudo, val_samples=pseudo.nb_sample), axis=-1)

for i in range(epochs):
    vgg.model.fit_generator(
        train,
        samples_per_epoch=train.nb_sample,
        nb_epoch=1,
        validation_data=valid,
        nb_val_samples=valid.nb_sample)

    vgg.model.fit_generator(
        pseudo,
        samples_per_epoch=pseudo.nb_sample,
        nb_epoch=1,
        validation_data=valid,
        nb_val_samples=valid.nb_sample)

vgg.model.save_weights("data/weights.h5")
