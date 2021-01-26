from keras.preprocessing import image
import numpy as np
import os
from model import generate_upscale2
import cv2
input_dir = "C:\programmering\stylegan\imgs"

output_dir = "altered/scaled"

m = generate_upscale2()

m.load_weights("weights/lastUpScale2.h5")
os.makedirs(output_dir,exist_ok=True)

for path in os.listdir(input_dir):
    noise = np.random.random((1, 512, 512, 1)) * 0.1 - 0.05
    img = image.load_img(os.path.join(input_dir,path))
    img = image.img_to_array(img)/255.*2-1
    img = cv2.resize(img,(256,256))
    img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
    upscaled = m.predict(np.stack([img]))[0]
    scaled = image.array_to_img(upscaled)
    scaled.save(os.path.join(output_dir,path))
    print(path)
