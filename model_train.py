from keras.layers import Conv2D,Dense,Dropout,BatchNormalization,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import os
import json
import numpy as np
from model import generate_model,pixel_crop,number_of_crops,learning_rate
input_folder = "resized"
meta = "metadata.txt"
with open(meta, "r") as f:
    data = json.load(f)
print("data",len(data))
batch_size = 16
load_weights = True

def generator():
    batch_img = []
    batch_labels = []
    while True:
        for d in data:
            if(d[1]==None):
                continue
            index = d[0]
            #print(index)
            img = image.load_img(input_folder + "\\" + "00000"[:5-len(str(index))] + str(index) + ".jpg")
            img = image.img_to_array(img)
            shape = img.shape
            #print(shape[0]/pixel_crop//number_of_crops)
            for i in range(shape[0]//(pixel_crop*number_of_crops)*2-1):#-number_of_crops/):
                i*=number_of_crops//2
                for j in range(shape[1] // (pixel_crop*number_of_crops)*2-1):#-number_of_crops):
                    j*=number_of_crops//2
                    x1,y1,x2,y2 = d[1][0][0],d[1][0][1],d[1][1][0],d[1][1][1]
                    #print(x1,y1,x2,y2)
                    iy1,iy2,ix1,ix2=pixel_crop*i,pixel_crop*(i+number_of_crops),j*pixel_crop,(j+number_of_crops)*pixel_crop
                    #print(ix1,ix2,iy1,iy2)
                    if(x1>ix2 or x2<ix1 or y1>iy2 or y2<iy1):
                        label = 0
                    else:
                        label = 1
                    #if(x1 < ix1 < x2 or x1 < ix2 < x2 or y1 < iy1 < y2 or y1 < iy2 < y2):
                    #    label = 1
                    #else:
                    #    label = 0
                    subimg = img[iy1:iy2,ix1:ix2]
                    #print(subimg.shape)
                    #save = image.array_to_img(subimg)
                    #image.save_img("just_bordered/"+ str(index)+"-" +str(ix1)+"-" +str(ix2)+"-" +str(iy1)+"-" +str(iy2) + "-" + str(label)+".jpg",save)
                    batch_img.append(subimg)
                    batch_labels.append(label)

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
    model.load_weights("weights/last32.h5")
print(model.summary())

check1 = ModelCheckpoint(filepath="weights/last32.h5",save_best_only=False, save_weights_only=True, verbose=1)
model.fit_generator(generator=generator(),steps_per_epoch=1300,epochs=10,callbacks=[check1])