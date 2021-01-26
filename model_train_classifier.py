from keras.layers import Conv2D,Dense,Dropout,BatchNormalization,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import os
import json
import numpy as np
from model import generate_classifier,pixel_crop,number_of_crops,learning_rate
import cv2
from random import randint
classes = 4
input_folders = [["col_bo","colfil_bo"],["colfil_bu","colfil_bu"],["col_pu","colfil_pu"],["col_no"]]
sum = 0
for i in input_folders:
    #print(i)
    for i2 in i:
        for p in os.listdir(i2):
            if p[-3:] == "jpg":
                sum += 1
print("sum:",sum)
meta = "metadata.txt"
with open(meta, "r") as f:
    data = json.load(f)
print("data",len(data))
batch_size = 64
load_weights = True
test_set = 100
def next_link(input_folders, class_index = 0):
    print("gen" + str(class_index))
    while True:
        for input_folder in input_folders[class_index]:
            #print(input_folder)
            for path in os.listdir(input_folder):
                if path[-3:] == "jpg":
                    try:
                        yield image.load_img(input_folder + "/" + path)
                    except:
                        pass
        print("all",str(input_folders))
gen = []
for i in range(classes):
  gen.append(next_link(input_folders,i))
def generator():
    maxframe = 32
    max_size = 128
    batch_img = []
    batch_labels = []
    current_class = 0
    index = 0
    while True:
        index += 1
        #print(index)
        img = next(gen[current_class])
        img = image.img_to_array(img)
        shape = img.shape
        #r = randint(0,maxframe)
        #size = max_size-r
        #left = randint(0,r)
        #top = randint(0,r)
        #right = r-left
        #bot = r-top
        #img = cv2.resize(img,(size,size))
        #img = np.pad(img,((top,bot),(left,right),(0,0)),constant_values=((0,0),(0,0),(0,0)))

        #save = image.array_to_img(img)
        #save2 = img[int(y1):int(y2),int(x1):int(x2)]
        #save2 = image.array_to_img(save2)
        #save.save("just_bordered/"+ str(index)+ "-" + str(int(use1)) + ".jpg")
        batch_img.append(img)
        label = np.zeros(classes)
        label[current_class] = 1
        batch_labels.append(label)
        #print(x1,y1,x2,y2)
        if(len(batch_img) == batch_size):
            batch_img = np.array(batch_img)
            #print(batch_img.shape)
            batch_labels = np.array(batch_labels)
            #print(batch_labels.shape)
            yield (batch_img,batch_labels)
            batch_img = []
            batch_labels = []
        current_class += 1
        if current_class == classes:
            current_class = 0

model = generate_classifier()
if load_weights:
    model.load_weights("weights/lastClassifier3.h5")
print(model.summary())
check1 = ModelCheckpoint(filepath="weights/lastClassifier3.h5",save_best_only=False, save_weights_only=True, verbose=1)
model.fit_generator(generator=generator(),steps_per_epoch=sum//batch_size,epochs=100,callbacks=[check1])