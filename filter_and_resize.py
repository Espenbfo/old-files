import os
import numpy as np
from keras.preprocessing import image
import cv2
import json

input_folder="rips"
input_dirs = ["temp"]
input_dirs = list(set(input_dirs))

print(input_dirs)

pad_to_size=True
save_shapes=False
out_size = 256
save_dir = "rips/temp2"
i = 0
if not os.path.exists(save_dir):
    print("creating_save_dir")
    os.makedirs(save_dir)
sum = 0
for dir in input_dirs:
    for path in os.listdir(os.path.join(input_folder,dir)):
        sum += 1
print(sum)
if save_shapes:
    shapes = []
for dir in input_dirs:
    print(dir)
    for path in os.listdir(os.path.join(input_folder,dir)):
        try:
            if path[-3:] != "jpg":
                continue
            try:
                img = image.load_img(os.path.join(input_folder,dir,path))
            except Exception as e:
                print(e)
                continue
            img = image.img_to_array(img)

            if type(img) != np.ndarray:
                print(path + ".jpg")
                print("is " + str(type(img)))
                continue
            if save_shapes:
                shapes.append(img.shape)
            if not pad_to_size:
                img = cv2.resize(img,(128,128))
            else:
                shape = img.shape
                xy = shape[0] / shape[1]
                if (xy < 1):
                    y = out_size
                    x = int(y * xy)
                else:
                    x = out_size
                    y = int(x / xy)
                img = cv2.resize(img, (y, x),interpolation=cv2.INTER_AREA)
                pady = out_size - y
                padx = out_size - x
                img = np.pad(img, ((padx // 2, padx // 2 + padx % 2), (pady // 2, pady // 2 + pady % 2), (0, 0)),
                             constant_values=((0, 0), (0, 0), (0, 0)))
            img = image.array_to_img(img)
            img.save(os.path.join(save_dir, "000000"[:6-len(str(i))]+str(i) + ".jpg"))
            i+=1
        except Exception as e:
            print(e)
            print(dir)
            print(path)
    print(i)
    #print("new_dir")
if save_shapes:
    with open("shapes.json","w") as f:
        json.dump(shapes,f)
print(i)