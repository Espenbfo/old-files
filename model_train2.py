from keras.layers import Conv2D,Dense,Dropout,BatchNormalization,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import os
import json
import numpy as np
from model import generate_model2,pixel_crop,number_of_crops,learning_rate
import cv2
from random import randint
import json
input_folder = "resized2"
meta = "metadata.txt"
with open(meta, "r") as f:
    data = json.load(f)
print("data",len(data))
batch_size = 64
load_weights = True
test_set = 100
with open("shapes.json","r") as f:
    shapes = json.load(f)
def generator():
    maxframe = 32
    max_size = 128
    batch_img = []
    batch_labels = []
    stage = 1
    max_stage = 2
    while True:
        for d in data:
            if(d[1]==None):
                continue
            index = d[0]
            #print(index)
            img = image.load_img(input_folder + "\\" + "00000"[:5-len(str(index))] + str(index) + ".jpg")
            img = image.img_to_array(img)
            #print(index)
            shape = shapes[index]
            #print(shape)
            r = randint(0,maxframe)
            size = max_size-r
            left = randint(0,r)
            top = randint(0,r)
            right = r-left
            bot = r-top
            y_warp = shape[0] / size
            x_warp = shape[1] / size
            x1, y1, x2, y2 = left + d[1][0][0] / x_warp, top + d[1][0][1] / y_warp, left + d[1][1][0] / x_warp, top + d[1][1][1] / y_warp
            img = cv2.resize(img,(size,size))
            img = np.pad(img,((top,bot),(left,right),(0,0)),constant_values=((0,0),(0,0),(0,0)))
            #print(img.shape)
            if stage == 0:
                pass
            if stage == 1 or stage == 3:
                img = np.flip(img,axis=1)
                x1 = 128-x1
                x2 = 128-x2
                x1,x2 =  x2,x1
            if stage == 2 or stage == 3:
                img = np.flip(img,axis=0)
                y1 = 128-y1
                y2 = 128-y2
                y1,y2 = y2,y1
            x1,x2,y1,y2 = x1/128,x2/128,y1/128,y2/128
            #save = image.array_to_img(img)
            #save2 = img[int(y1):int(y2),int(x1):int(x2)]
            #save2 = image.array_to_img(save2)
            #save2.save("just_bordered/"+ str(index)+".jpg")
            batch_img.append(img)
            batch_labels.append([x1,y1,x2,y2])
            #print(x1,y1,x2,y2)
            if(len(batch_img) == batch_size):
                batch_img = np.array(batch_img)
                #print(batch_img.shape)
                batch_labels = np.array(batch_labels)
                #print(batch_labels.shape)
                yield (batch_img,batch_labels)
                batch_img = []
                batch_labels = []
        print("")
        print("all")
        stage += 1
        if stage==max_stage:
            stage = 0

model = generate_model2()
print(model.summary())
if load_weights:
    model.load_weights("weights/lastNoDropout.h5")
print(model.summary())

check1 = ModelCheckpoint(filepath="weights/lastNoDropout.h5",save_best_only=False, save_weights_only=True, verbose=1)
model.fit_generator(generator=generator(),steps_per_epoch=282//4,epochs=100,callbacks=[check1])