from keras.layers import Conv2D,Dense,Dropout,BatchNormalization,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from model import generate_model2,generate_classifier
import os
import json
import numpy as np
import pygame as pg
import sys
import cv2
number_of_crops = 32
pixel_crop = 8
image_index= 100
data_file = "metadata.txt"
with open(data_file, "r") as f:
    data  = json.load(f)

def generate_link(image_index=image_index):
    return "validation/00000"[:-len(str(image_index))] + str(image_index) +".jpg"

path = "folder" #folder
paths = os.listdir(path)
for i in range(len(paths)-1,-1,-1):
    if paths[i][-3:] != "jpg":
        paths.pop(i)
def generate_link_from_paths(image_index=image_index,paths=paths):
    print(path+ "/" +paths[image_index])
    return path+ "/" +paths[image_index]
class InputBox:

    def __init__(self, x, y, w, h, color = (0,0,0,255)):
        self.rect = pg.Rect(x, y, w, h)
        self.color = pg.Color(color[0],color[1],color[2],color[3])
        self.active = False
        self.surface = pg.Surface((w,h))
        self.surface.fill((color[0],color[1],color[2]))
        self.surface.set_alpha(color[3])

    def draw(self, screen):
        pg.draw.rect(screen, self.color, self.rect, 0)

    def getWidth(self):
        return self.rect.w

    def setAlpha(self, a):
        self.surface.set_alpha(128)

    def setColr(self,r,g,b):
        self.surface.fill((r,g,b))
model = generate_model2()
model2 = generate_classifier()
model2.load_weights("weights/lastClassifier3.h5")
model.load_weights("weights/lastNoDropout.h5")

def generate_predict(imgname,increase = 0.04):
    img = image.load_img(imgname)
    img = image.img_to_array(img)
    shape = img.shape
    print(shape)
    img = cv2.resize(img,(128,128))
    print(img.shape)
    y_warp = shape[0]
    x_warp = shape[1]

    predict = model.predict(np.array([img]))[0]
    rating = model2.predict(np.array([img]))[0]
    print("rating: ",rating)
    print(shape)
    print(y_warp)
    print(x_warp)
    print(predict)
    color = 1
    predict[0] -= increase
    predict[2]+= increase
    predict[1]-= increase/3
    predict[3]+= increase/3
    predict[0]*=x_warp
    predict[2]*=x_warp
    predict[1]*=y_warp
    predict[3]*=y_warp
    return InputBox(predict[0], predict[1], predict[2]-predict[0], predict[3]-predict[1],
                 (int(color * 255), int(color * 255), int(color * 255),200))
#for i in range(len(result)):
#    input_boxes.append(InputBox(boxdata[i][0],boxdata[i][1],boxdata[i][2]-boxdata[i][0],boxdata[i][3]-boxdata[i][1],(int(result[i]*255),int(result[i]*255),int(result[i]*255))))
input_boxes= []
input_boxes.append(generate_predict(imgname = generate_link_from_paths(image_index)))
#save_file = "metadata.txt"
pixel_crop = 8
load=True
if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))

pg.init()
pg.key.set_repeat(300, 25)
screen = pg.display.set_mode((1920, 1024))
COLOR_INACTIVE = pg.Color('black')
COLOR_ACTIVE = pg.Color('white')
FONT = pg.font.SysFont('arial', 20)




class Background(pg.sprite.Sprite):
    def __init__(self, image_file, location):
        pg.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pg.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

def main(image_index=image_index,input_boxes=input_boxes):
    clock = pg.time.Clock()
    #input_box1 = InputBox(15, 100, 70, 30)
    done = False

    BackGround = Background(generate_link_from_paths(image_index), [0, 0])
    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
            if event.type== pg.KEYDOWN:
                if event.key == pg.K_d:
                    image_index += 1
                    input_boxes=[generate_predict(imgname=generate_link_from_paths(image_index))]
                    #while data[image_index][1] is None:
                    #    #print(data[image_index][0])
                    #    image_index += 1
                    #p = data[image_index][1]
                    #print(p)
                    #input_boxes = [InputBox(p[0][0], p[0][1], p[1][0]-p[0][0], p[1][1]-p[0][1],
                    #(int(1 * 255), int(1 * 255), int(1 * 255),200))]
                    BackGround = Background(generate_link_from_paths(image_index),
                                            [0, 0])
                if event.key == pg.K_w:
                    done = True

        screen.fill((255, 255, 255))
        screen.blit(BackGround.image, BackGround.rect)
        #print text
        for box in input_boxes:
            screen.blit(box.surface,(box.rect.x,box.rect.y))
            #box.draw(screen)

        pg.display.flip()
        clock.tick(30)

main()
pg.quit()
