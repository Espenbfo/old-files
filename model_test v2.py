from keras.layers import Conv2D,Dense,Dropout,BatchNormalization,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from model import generate_model
import os
import json
import numpy as np
import pygame as pg
import sys
number_of_crops = 32
pixel_crop = 8

imgname="resized/02351.jpg"
img = image.load_img(imgname)
img = image.img_to_array(img)


class InputBox:

    def __init__(self, x, y, w, h, color = (0,0,0,255)):
        self.rect = pg.Rect(x, y, w, h)
        self.color = pg.Color(color[0],color[1],color[2],color[3])
        self.active = False
        self.surface = pg.Surface((pixel_crop,pixel_crop))
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
model = generate_model()
print(model.summary)
model.load_weights("weights/last32.h5")

shape = img.shape
boxdata = []
print(shape[0]//pixel_crop//8)
subimages = []
predictions = [[0]*(shape[0]//pixel_crop-1) for i in range(shape[1]//pixel_crop-1)]
for i in range(shape[0]//(pixel_crop*number_of_crops)-1):
    i *= number_of_crops
    for j in range(shape[1] // (pixel_crop*number_of_crops)-1):
        j*=number_of_crops
        iy1,iy2,ix1,ix2=pixel_crop*i,pixel_crop*(i+number_of_crops),j*pixel_crop,(j+number_of_crops)*pixel_crop

        subimg = img[iy1:iy2, ix1:ix2]
        #print(subimg.shape)
        subimages.append(subimg)
        boxdata.append([ix1,iy1,ix2, iy2])

result = model.predict(np.array(subimages))
for i in range(len(result)):
    for r1 in range(number_of_crops):
        for r2 in range(number_of_crops):
            l1 = (shape[1]//pixel_crop-number_of_crops)
            l2 = (shape[0]//pixel_crop-number_of_crops)
            predictions[boxdata[i][0]//pixel_crop+r1][boxdata[i][1]//pixel_crop +r2] += float(result[i])
#subimages = np.array(subimages)
#print(subimages.shape)
#result = model.predict(subimages)
print(result)
input_boxes = []
argmax = 0
for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        if predictions[i][j] > argmax:
            argmax = predictions[i][j]
print("argmax",argmax)
print(predictions)
for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        p = predictions[i][j]
        cutoff=0.5
        if (p > argmax*cutoff):
            color =(predictions[i][j]-argmax*cutoff)/(argmax*(1-cutoff))
        else:
            color = 0
        input_boxes.append(
            InputBox(i*pixel_crop, j*pixel_crop, pixel_crop, pixel_crop,
                     (int(color * 255), int(color * 255), int(color * 255),200)))
#for i in range(len(result)):
#    input_boxes.append(InputBox(boxdata[i][0],boxdata[i][1],boxdata[i][2]-boxdata[i][0],boxdata[i][3]-boxdata[i][1],(int(result[i]*255),int(result[i]*255),int(result[i]*255))))

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

def main():
    clock = pg.time.Clock()
    #input_box1 = InputBox(15, 100, 70, 30)
    done = False


    BackGround = Background(imgname, [0, 0])
    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
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
