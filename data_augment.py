import cv2
import numpy as np
import imutils
from math import sin,cos,pi
img = cv2.imread("col_pad_crop/00001.jpg")
from random import random
import os
def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),borderMode=cv2.BORDER_REPLICATE)
    return rotated_mat

def stretch(img,xs,ys):
    height, width = img.shape[:2]
    img = cv2.resize(img,(int(width*xs),int(height*ys)))
    return img

def contrast(img,c):
    return cv2.addWeighted( img, c, img, 0, 127*(1-c))

def brighten_darken(img,b):
    return cv2.addWeighted( img, 1, img, 0, b)
#x,y,c = img.shape
#angle=0
#rad = angle/180*pi
#M = cv2.getRotationMatrix2D((x/2,y/2),angle,1)
#img = cv2.warpAffine(img,M,(int(sin(pi/2-rad)*y+sin(rad)*x),int(cos(pi/2-rad)*y+cos(rad)*x)),borderMode=cv2.BORDER_REPLICATE)
def augment(img):
    img= rotate_image(img,random()*6-3)
    img = stretch(img,0.95+random()*0.1,0.95+random()*0.1)
    img=contrast(img,0.9+random()*0.4)
    img=brighten_darken(img,random()*30-15)
    return img

def resize_and_pad(img,out_size):
    shape = img.shape
    xy = shape[0] / shape[1]
    if (xy < 1):
        y = out_size
        x = int(y * xy)
    else:
        x = out_size
        y = int(x / xy)
    img = cv2.resize(img, (y, x))
    pady = out_size - y
    padx = out_size - x
    img = np.pad(img, ((padx // 2, padx // 2 + padx % 2), (pady // 2, pady // 2 + pady % 2), (0, 0)),
                 constant_values=((0, 0), (0, 0), (0, 0)))
    return img
#img = rotate_image(img,5)
#img = stretch(img,1,1.2)
#img = contrast(img,1.3)
#img = brighten_darken(img,-15)

input_folder="filer"
input_dirs = []
out_dir="altered/augmented"
os.makedirs(out_dir,exist_ok=True)

for path in os.listdir(input_folder):
    if path not in input_dirs:
        print(path)

sum = 0
for dir in input_dirs:
    for path in os.listdir(os.path.join(input_folder, dir)):
        if path[-4:] != ".jpg":
            continue
        sum += 1
print("sum",sum)

ind = 0
for dir in input_dirs:
    print(dir)
    print(ind)
    for path in os.listdir(os.path.join(input_folder,dir)):
        if path[-4:] != ".jpg":
            continue
        try:
            img = cv2.imread(os.path.join(input_folder,dir,path))
        except Exception as e:
            print(e)
            print(path)
            continue
        if img is None:
            print(path)
            continue
        for i in range(5):
            img2 = img.copy()
            img2 = augment(img2)
            img2 = resize_and_pad(img2,256)
            cv2.imwrite(os.path.join(out_dir,"00000"[:6-len(str(ind))] + str(ind)+".jpg"),img2)
            #print(img2.shape)
            ind += 1

#img = imutils.rotate(img,20)
#cv2.imshow("1",img)
#cv2.waitKey()