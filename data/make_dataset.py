import numpy as np
import os
import pandas as pd
from shutil import copyfile

SAMPLE_TRAIN_SIZE = 150
SAMPLE_VALID_SIZE = 100

driver_img_paths = pd.read_csv("driver_imgs_list.csv")
valid_drivers = driver_img_paths["subject"].unique()
np.random.shuffle(valid_drivers)
valid_drivers = valid_drivers[:3]

valid_img_paths = driver_img_paths[driver_img_paths["subject"].isin(valid_drivers)]
valid_img_paths = valid_img_paths["classname"].str.cat(valid_img_paths["img"], sep="/")

for i in range(10):
    os.makedirs("valid/c{}".format(i))
    os.makedirs("sample/valid/c{}".format(i))
    os.makedirs("sample/train/c{}".format(i))

for p in valid_img_paths:
    os.rename("train/" + p, "valid/" + p)

for i in range(10):
    for n, f in enumerate(os.listdir("train/c{}".format(i))):
        if n >= SAMPLE_TRAIN_SIZE:
            break
        f_path = os.path.join("train/c{}".format(i), f)
        copyfile(f_path, "sample/train/c{}/{}".format(i, f))
    for n, f in enumerate(os.listdir("valid/c{}".format(i))):
        if n >= SAMPLE_VALID_SIZE:
            break
        f_path = os.path.join("valid/c{}".format(i), f)
        copyfile(f_path, "sample/valid/c{}/{}".format(i, f))
