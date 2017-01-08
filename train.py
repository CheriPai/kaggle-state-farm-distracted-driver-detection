from vgg16 import VGG16
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

img_size = 224
batch_size = 64
dropout = 0.1

lr = 0.001
decay = 1e-6
epochs = 10

gen = ImageDataGenerator()
train = gen.flow_from_directory(
    "data/train",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True)
valid = gen.flow_from_directory(
    "data/valid",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False)

vgg = VGG16(dropout)
vgg.finetune_dense(10)

vgg.model.compile(
    optimizer=Adam(lr=lr, decay=decay),
    loss="categorical_crossentropy",
    metrics=["accuracy"])
vgg.model.fit_generator(
    train,
    samples_per_epoch=train.nb_sample,
    nb_epoch=epochs,
    validation_data=valid,
    nb_val_samples=valid.nb_sample)

vgg.model.save_weights("data/weights.h5")
