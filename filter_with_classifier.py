from model import generate_classifier
from keras.preprocessing import image
import numpy as np
import os
import cv2
import json
from filter_with_model import crop_double_check
input_dirs = []

save_dirs = ["col_pad_bo","col_pad_bu","col_pad_pu","col_pad_no"]
crop_dir = "col_pad_crop"
cutoff = 0.8
m = generate_classifier()
m.load_weights("weights/lastClassifier2.h5")
os.makedirs(crop_dir,exist_ok=True)
for save_dir in save_dirs:
    if not os.path.exists(save_dir):
        print("creating_save_dir")
        os.makedirs(save_dir)
def classify(imgname,cutoff=0.9):
    try:
        if imgname[-3:] != "jpg":
            return None, -1
        try:
            img = image.load_img(imgname)
        except:
            return None, -1

        img = image.img_to_array(img)

        img2 = cv2.resize(img, (128, 128))
        pred = m.predict(np.array([img2]))[0]
        rating = pred[0]
        current = -1
        j = 0
        for p in pred:
            if p > cutoff:
                if current == -1:
                    current = j
                else:
                    current = None
            j+=1
        if current is None or current == -1:
            return img, -1
        #print(pred)
        #print(rating)
        return img, current
    except Exception as e:
        print(imgname)
        print(e)
        return None, -1
i = 0
paths = []
for path in input_dirs:
    temppaths=[]
    for path2 in os.listdir(path):
        if not "." in path2:
            for path3 in os.listdir(path + "/" + path2):
                if path3[-3:] == "jpg":
                    paths.append(path + "/" + path2 + "/" + path3)
        if path2[-3:] == "jpg":
            paths.append(path + "/" + path2)
print(len(paths))
data = []
j = 0
tries = 0
currents = [0, 0, 0, 0]
crops = 0
for path in paths:
    img,current = classify(path)
    if current != -1:
        if current == 0:
            crop = crop_double_check(img)
            if not crop is None:
                crop = image.array_to_img(crop)
                crop_path = crop_dir + "/00000"[:6 - len(str(crops))] + str(crops) + ".jpg"
                crop.save(crop_path)
                crops += 1
        path = "00000"[:5-len(str(currents[current]))]+str(currents[current])+".jpg"
        img = image.array_to_img(img)
        img.save(os.path.join(save_dirs[current],path))
        currents[current] += 1
        i += 1
        print(i,currents,tries)
    tries += 1