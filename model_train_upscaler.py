from keras.layers import Conv2D,Dense,Dropout,BatchNormalization,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import os
import json
import numpy as np
from model import generate_upscale2,pixel_crop,number_of_crops,learning_rate
import cv2
from random import randint

input_small = "bo_small"
input_large = "bo_large"

sum = len(os.listdir(input_large))
print(sum)
batch_size = 16
load_weights = True
test_set = 100
def next_link(input_folder):
    while True:
        for path in os.listdir(input_folder):
            if path[-3:] == "jpg":
                try:
                    img = image.load_img(input_folder + "/" + path)
                    img = image.img_to_array(img)/255.*2-1
                    yield img
                except Exception as e:
                    print(e)
        print("all",str(input_folder))

gen_small = next_link(input_small)
gen_large = next_link(input_large)
r_noise = 0.004
def generator():
    batch_img = []
    batch_labels = []
    index = 0
    while True:
        index += 1

        img = cv2.resize(next(gen_small),(512,512),interpolation=cv2.INTER_CUBIC)

        #label = np.clip(next(gen_large)+np.random.random((512,512,3))*r_noise-np.ones((512,512,3))*(r_noise/2),-1,1)
        label =next(gen_large)

        shape = img.shape
        batch_img.append(img)
        batch_labels.append(label)

        if(len(batch_img) == batch_size):
            #noise = np.random.random((batch_size, 512, 512, 1)) * 0.1 - 0.05
            batch_img = np.array(batch_img)

            batch_labels = np.array(batch_labels)

            yield (batch_img,batch_labels)
            batch_img = []
            batch_labels = []

model = generate_upscale2()
model.compile(Adam(learning_rate=0.0003), "mean_squared_error", metrics=["accuracy"])
print(model.summary)
if load_weights:
    model.load_weights("weights/lastUpScale2.h5")
print(model.summary())
check1 = ModelCheckpoint(filepath="weights/lastUpScale2.h5",save_best_only=False, save_weights_only=True, verbose=1)
print(sum//batch_size)
model.fit_generator(generator=generator(),steps_per_epoch=sum//batch_size,epochs=100,callbacks=[check1])