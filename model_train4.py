from keras.layers import Conv2D,Dense,Dropout,BatchNormalization,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import os
import json
import numpy as np
from model import generate_model,pixel_crop,number_of_crops,learning_rate
from random import randint
import cv2
input_folder = "resized"
input_folder2 = "resized5"
meta = "metadata.txt"
meta2  = "metadata2.txt"
with open(meta, "r") as f:
    data = json.load(f)
with open(meta2, "r") as f:
    data2 = json.load(f)
links = input_folder,input_folder2
metas = 2
print("data",len(data))
batch_size = 64
load_weights = True

def generator():
    batch_img = []
    batch_labels = []
    while True:
        for m in range(metas):
            if m == 1:
                datafile = data
                folder = input_folder
            elif m == 0:
                datafile = data2
                folder = input_folder2
            for d in datafile:
                if(d[1]==None):
                    continue
                index = d[0]
                #print(index)
                img = image.load_img(folder + "\\" + "00000"[:5-len(str(index))] + str(index) + ".jpg")
                img = image.img_to_array(img)
                #if randint(0, 3) == 3:
                #    img2 = image.img_to_array(img2)
                #else:
                img2 = img
                shape = img.shape
                #print(shape[0]/pixel_crop//number_of_crops)
                x1,y1,x2,y2 = d[1][0][0],d[1][0][1],d[1][1][0],d[1][1][1]
                if (x1 == x2 or y1 == y2):
                    continue
                #print(x1,y1,x2,y2)
                #iy1,iy2,ix1,ix2=pixel_crop*i,pixel_crop*(i+number_of_crops),j*pixel_crop,(j+number_of_crops)*pixel_crop
                #print(ix1,ix2,iy1,iy2)
                bh = img2.shape[0]
                bb = img2.shape[1]
                #print(bh)
                #print(bb)
                h = y2-y1
                b = x2-x1
                rh=randint(0,bh-h)
                rb = randint(0,bb-b)
                rx1 = rb
                rx2 = rb+b
                ry1 = rh
                ry2 = rh+h
                #if(x1 < ix1 < x2 or x1 < ix2 < x2 or y1 < iy1 < y2 or y1 < iy2 < y2):
                #    label = 1
                #else:
                #    label = 0
                try:
                    subimg = img[y1:y2,x1:x2]
                    subimg = cv2.resize(subimg,(128,128))
                    subwrong = img2[ry1:ry2,rx1:rx2]
                    if (subwrong is None):
                        print(ry1,ry2,rx1,rx2)
                        print(index)
                    subwrong = cv2.resize(subwrong,(128,128))
                except Exception as e:
                    print(index)
                    print(shape)
                    print(y1,y2,x1,x2)
                    print(ry1, ry2, rx1, rx2)
                    print(e)
                #print(subimg.shape)
                #if (folder == input_folder2):

                    #save = image.array_to_img(subimg)
                    #image.save_img("just_bordered/"+ str(index)+"-" +str(x1)+"-" +str(x2)+"-" +str(y1)+"-" +str(y2) + ".jpg",save)
                batch_img.append(subimg)
                batch_img.append(subwrong)
                batch_labels.append(1)
                batch_labels.append(0)

                if(len(batch_img) == batch_size):
                    batch_img = np.array(batch_img)
                    #print(batch_img.shape)
                    batch_labels = np.array(batch_labels)
                    #print(batch_labels.shape)
                    yield (batch_img,batch_labels)
                    batch_img = []
                    batch_labels = []
            print("all")

model = generate_model()
if load_weights:
    model.load_weights("weights/lastChecker3.h5")
print(model.summary())

check1 = ModelCheckpoint(filepath="weights/lastChecker3.h5",save_best_only=False, save_weights_only=True, verbose=1)
model.fit_generator(generator=generator(),steps_per_epoch=140,epochs=100,callbacks=[check1])