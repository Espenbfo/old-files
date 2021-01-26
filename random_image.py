import pygame as pg
import numpy as np
import cv2
from useful_funcs import pad_to_size,list_dir_recursive, is_jpg,is_png, is_gif,\
    is_mp4,gif_to_numpy,mp4_to_numpy_generator, find_score
import random
import time
import gif2numpy
import os
import re
#average = np.random.RandomState().randn(1,Gs.input_shape[1])
#for i in range(1000):
#    average += np.random.RandomState().randn(1,Gs.input_shape[1])
#average/=1001
#average = Gs.components.mapping.run(average, None)[0]
#print(average.shape)
folder_list = list_dir_recursive("filer/",folderregex="", fileregex=".*(score){1,}.*")#,"filer/reddit_sub_pasta")
min_score= 100

titlewords = r"" #må skille flere uavhengige ord med |
user = r""

titles = {}

folder_list = list(filter(lambda x: find_score(x,titlewords,user) >= min_score, folder_list))

"""
print(titles)
for key in titles.keys():
    if len(titles[key])>=2:
        to_delete = titles[key][1:]
        for filename in to_delete:
            try:
                os.remove(filename)
            except Exception as e:
                print(e)
        print(titles[key])

for path in folder_list:
    try:
        splitpath = path.split("\\")
        splitpath2 = splitpath[1].rsplit(".",1)
        os.rename(path.replace("/","\\"),splitpath[0] + "\\" + splitpath2[0].replace("{","_",1).replace("}","_",1)+ "." + splitpath2[1])
    except:
        try:
            splitpath = path.split("\\")
            splitpath[0] = "\\\\?\\" + splitpath[0].replace("/","\\")
            splitpath2 = splitpath[1].rsplit(".", 1)
            os.rename(path.replace("/", "\\"),
                      splitpath[0] + "\\" + splitpath2[0].replace("_", "{", 1).replace("_", "}", 1) + "." + splitpath2[
                          1])
        except:
            print(path)
            """
pathsum = len(folder_list)
#folder_list = []
random_choice = True
if not random_choice:
    path_iter = iter(folder_list)
    i = 0
def new_folder_list():
    dir = "reddit_sub_pasta"
    global folder_list
    global path_iter
    global pathsum
    folder =".jpg"
    while not os.path.isdir(os.path.join("filer",dir, folder)):
        folder = random.choice(os.listdir(os.path.join("filer",dir)))
        print(folder)
    folder_list = list_dir_recursive(os.path.join("filer",dir, folder))
    random_choice = False
    if not random_choice:
        path_iter = iter(folder_list)
    pathsum = len(folder_list)
#new_folder_list()

#folder_list = list_dir_recursive("filer",regex="reddit.*")
print(len(folder_list))
pg.init()
display = pg.display.set_mode((1500, 1000))

running = True
clock = pg.time.Clock()
formats = "jpg"
only_video = False
movie = False

index = 0
if format == "gif":
    fileextention = is_gif
    movie = True
elif format == "mp4":
    fileextention = is_mp4
    movie = True
else:
    fileextention = is_jpg
time_max = 4
timeleft = 1
timeRunning = False
videoRunning = False
def new_surf():
    global movie
    global path_iter
    global index
    f = ""
    while True:
        if random_choice:
            path = random.choice(folder_list)
        else:
            try:
                path = next(path_iter)
                index += 1
                print(index,"of",pathsum)
            except StopIteration:
                index = 0
                path_iter = iter(folder_list)
                path = next(path_iter)
                index += 1
                print(index,"of",pathsum)
        print(path)
        if is_jpg(path) and not only_video:
            movie = False
            f = "jpg"
            break
        elif is_png(path) and not only_video:
            movie = False
            f = "png"
            break
        elif is_gif(path):
            movie = True
            f = "gif"
            break
        elif is_mp4(path):
            movie = True
            f = "mp4"
            break
    #while not fileextention(path):
    #    print(path)
    #    path = random.choice(folder_list)
    #print(path)
    try:
        if not movie:
            image = cv2.imread(path)
            image = cv2.flip(image,1)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = pad_to_size(image,1500,1000)
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #image *= 255.
            #image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            image =pg.surfarray.make_surface(image)
            return image
        else:
            if timeRunning:
                return video(path)
            return video(path)
    except Exception as e:
        print(e)
        return new_surf()

class video:
    def __init__(self, filename,frames_pr_second=24,format="mp4",seconds=-1):
        if format == "mp4":
            self.frames = mp4_to_numpy_generator(filename,seconds=seconds)
            self.frames_pr_second = self.frames.framePerSecond
        elif format == "gif":
            self.frames,self.frames_pr_second = gif_to_numpy(filename)
        #print(self.frames.shape)
        #self.frames = np.delete(self.frames,3,2)
        #print(self.frames.shape)
        #self.frames = [cv2.rotate(pad_to_size(frame,1500,1000), cv2.ROTATE_90_COUNTERCLOCKWISE) for frame in self.frames]
        #self.frames = [pg.surfarray.make_surface(frame) for frame in self.frames]
        self.nframes = self.frames.frameCount
        self.n = 0
        #self.frames_pr_second = frames_pr_second
        self.time = time.time()
        self.iterator=iter(self.frames)
    def __iter__(self):
        return self

    def start(self):
        self.time = time.time()

    def __next__(self):
        try:
            new_time =time.time()
            prev_frame=self.n
            self.n += (new_time-self.time)*self.frames_pr_second
            self.time = new_time
            if self.n >= self.nframes:
               self.n = 0
            #print(int(self.n))
            self.frames.cap.set(cv2.CAP_PROP_POS_FRAMES,int(self.n))
            frame = next(self.iterator)
            frame = cv2.rotate(pad_to_size(frame,1500,1000), cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = pg.surfarray.make_surface(frame)
            return frame
        except StopIteration:
            raise StopIteration
        #new_time =time.time()
        #self.n += (new_time-self.time)*self.frames_pr_second
        #self.time = new_time
        #if self.n >= self.nframes:
        #    self.n = 0
        #return self.frames[int(self.n)-1]

v = None
surf = new_surf()
if movie:
    v = iter(surf)
    surf = next(v)
else:
    v = None
last_time = 0


while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.KMOD_SHIFT:
                shift = True
            if event.key == pg.K_d:
                videoRunning = False
                while True:
                    try:
                        surf = new_surf()
                        if movie:
                            v = iter(surf)
                            surf = next(v)
                        else:
                            v = None
                        break
                    except:
                        print("Skipping")
                        pass

            if event.key == pg.K_s:
                timeRunning = not timeRunning
            if event.key == pg.K_t:
                new_folder_list()
                index = 0
                videoRunning = False
                surf = new_surf()
                if movie:
                    v = iter(surf)
                    surf = next(v)
                else:
                    v = None
            if event.key == pg.K_SPACE:
                videoRunning = not videoRunning
                if v is not None:
                    v.start()
        if event.type == pg.KEYUP:
            if event.key == pg.KMOD_SHIFT:
                shift = False
    if timeRunning:
        new_time = time.time()
        timeleft,last_time = timeleft-(new_time-last_time),new_time
        #print(timeleft)
        if timeleft <= 0:
            surf = new_surf()
            if movie:
                videoRunning = True
                lasttime = time.time()
                timeleft = surf.nframes//surf.frames_pr_second
                v = iter(surf)
                surf = next(v)
            else:
                v = None
                timeleft = time_max
    if videoRunning:
        if v is not None:
            surf = next(v)
    display.blit(surf, (0, 0))
    pg.display.update()
    #clock.tick(30)
pg.quit()