import numpy as np
import os
from vgg16 import VGG16
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img

img_size = 224
batch_size = 64
dropout = 0.1
test_dir = "data/test"

vgg = VGG16(dropout)
vgg.finetune_dense(10)
vgg.model.load_weights("data/weights.h5")

vgg.model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

open("data/output", "w").close()
with open("data/output", "a") as f:
    f.write("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n")
    for i in os.listdir(test_dir):
        img_path = os.path.join(test_dir, i)
        img = img_to_array(load_img(img_path, target_size=((img_size, img_size))))
        img = np.array([img])
        prediction = vgg.model.predict(img)[0]
        prediction = np.clip(prediction, 0.004, 0.96)
        f.write(i + "," + ",".join(["%.5f" % n for n in prediction]) + "\n")
