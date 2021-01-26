import json
from keras.preprocessing import image
import numpy as np
import os
input_dir="resized"
meta="metadata.txt"
output_dir = "just_bordered"
use_model = False
if use_model:
    from model import generate_model2
    import cv2
with open(meta, "r") as f:
    data = json.load(f)
for d in data:
    index = d[0]
    if d[1] == None:
        continue
    img = image.load_img(input_dir+"\\" + "00000"[:5-len(str(index))] + str(index)+ ".jpg")
    img = image.img_to_array(img)
    if use_model:
        m = generate_model2()
        m.load_weights("weights/last2.h5")
        shape = img.shape
        #print(shape)
        img2 = cv2.resize(img, (128, 128))
        y_warp = shape[0] / 128
        x_warp = shape[1] / 128
        result = m.predict(np.array([img2]))[0]
        #print(result)
        if int(result[0]*x_warp)==int(result[2]*x_warp) or int(result[1]*y_warp)==int(result[3]*y_warp):
            continue
        cropped_1 = [int(result[0]*x_warp),int(result[1]*y_warp)]
        cropped_2 = [int(result[2]*x_warp), int(result[3]*y_warp)]
    else:
        if d[1][0][0] == d[1][1][0] or d[1][0][1] == d[1][1][1]:
            continue
        cropped_1 = d[1][0]
        cropped_2 = d[1][1]
    #print(cropped_1,cropped_2)
    #y,x = img.shape[:2]
    #cropped_1[1], cropped_2[1], cropped_1[0], cropped_2[0] = y - cropped_2[1], y - cropped_1[1], x - cropped_2[0], x - cropped_1[0]
    #img = np.flip(img,axis=1)
    #img = np.flip(img,axis=0)
    img = img[cropped_1[1]:cropped_2[1], cropped_1[0]:cropped_2[0]]
    img = image.array_to_img(img)
    img.save(os.path.join(output_dir, "00000"[:5 - len(str(index))] + str(index) + ".jpg"))

