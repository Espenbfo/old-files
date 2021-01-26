import os
import numpy as np
from keras.preprocessing import image
import cv2

input_dirs = []
pixel_crop = 8
save_dir = "results"

i = 0
for dir in input_dirs:
    for path in os.listdir(dir):
        if path[-3:] != "jpg":
            continue
        try:
            img = image.load_img(dir + "\\" + path)
        except Exception as e:
            print(e)
            continue
        img = image.img_to_array(img)

        if type(img) != np.ndarray:
            print(path + ".jpg")
            print("is " + str(type(img)))
            continue

        height,width,channels = img.shape[:3]
        if(height > 1080):
            img=cv2.resize(img, (int(1080*width/(height)),1080))
        height, width, channels = img.shape[:3]
        if height%pixel_crop != 0:
            img = img[:height-height%pixel_crop]
        if width%pixel_crop != 0:
            img = img[:,:width-width%pixel_crop]
        if channels != 3:
            img = np.concatenate(img.copy(),img.copy(),img)
            print(img.shape)
        img = image.array_to_img(img)
        img.save(os.path.join(save_dir, "00000"[:5-len(str(i))]+str(i) + ".jpg"))
        i+=1
    print("new_dir")
print(i)