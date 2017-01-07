from vgg16 import VGG16
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

img_size = 224
epochs = 5

gen = ImageDataGenerator()
train = gen.flow_from_directory("data/valid", target_size=(img_size, img_size), batch_size=32, class_mode="categorical", shuffle=True)
valid = gen.flow_from_directory("data/sample", target_size=(img_size, img_size), batch_size=32, class_mode="categorical", shuffle=False)

vgg = VGG16(0.1)
vgg.finetune(10)
vgg.model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
vgg.model.fit_generator(train, samples_per_epoch=train.nb_sample, nb_epoch=epochs, validation_data=valid, nb_val_samples=valid.nb_sample)