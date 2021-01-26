import cv2
import numpy as np
import os
import re
import gif2numpy


def pad_to_size(img, x_size, y_size):
    img_ratio = img.shape[0] / img.shape[1]
    ratio = y_size / x_size
    if (img_ratio > ratio):
        y = y_size
        x = int(y / img_ratio)
    else:
        x = x_size
        y = int(x * img_ratio)
    img = cv2.resize(img, (x, y), interpolation=cv2.INTER_AREA)
    pady = y_size - y
    padx = x_size - x
    img = np.pad(img, ((pady // 2, pady // 2 + pady % 2), (padx // 2, padx // 2 + padx % 2), (0, 0)),
                 constant_values=((0, 0), (0, 0), (1, 1)))
    return img


def list_dir_recursive(*dirs, folderregex=".*", fileregex=".*"):
    fopattern = re.compile(folderregex)
    fipattern = re.compile(fileregex)
    files = []
    for dir in dirs:
        temp_files = os.listdir(dir)
        for file in temp_files:
            fullfile = os.path.join(dir, file)
            if not os.path.isdir(fullfile):
                if not fipattern.match(file):
                    print(file)
                else:
                    files.append(fullfile)
            else:
                if not fopattern.match(file):
                    print(file)
                else:
                    files += list_dir_recursive(fullfile)
    return files


def get_latest_file(*dirs, regex=".*"):
    files = list_dir_recursive(*dirs)
    filedates = {}

    pattern = re.compile(regex)
    for file in files:
        if not pattern.match(file):
            print(file)
            continue
        else:
            filedates[file] = os.path.getmtime(file)
    files_sorted = sorted(filedates, key=filedates.get, reverse=True)
    print(files_sorted)
    return files_sorted


# get_latest_file('C:/programmering/stylegan2/results',regex=".*[0-9][0-9][0-9][0-9][0-9][0-9]\.pkl$")
def is_jpg(file):
    if file[-4:] == ".jpg":
        return True


def is_png(file):
    if file[-4:] == ".png":
        return True


def is_gif(file):
    if file[-4:] == ".gif":
        return True


def is_mp4(file):
    if file[-4:] == ".mp4":
        return True


def mp4_to_numpy(file, seconds=-1):
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framePerSecond = int(cap.get(cv2.CAP_PROP_FPS))

    fc = 0
    ret = True

    print(framePerSecond, seconds)
    if seconds != -1:
        frameCount = min(frameCount, int(framePerSecond * seconds))
    if frameCount * frameWidth * frameHeight > 1000 * 500 * 500:
        print("b/a", frameCount)
        frameCount = int(1000 * 500 * 500 / frameWidth / frameHeight)
        print("b/a", frameCount)

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        buf[fc] = cv2.cvtColor(buf[fc], cv2.COLOR_BGR2RGB)
        fc += 1

    return buf, framePerSecond


class mp4_to_numpy_generator():
    def __init__(self, file, seconds=-1):
        cap = cv2.VideoCapture(file)
        self.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framePerSecond = int(cap.get(cv2.CAP_PROP_FPS))

        self.fc = 0
        ret = True

        if seconds != -1:
            self.frameCount = min(self.frameCount, int(self.framePerSecond * seconds))
        # if self.frameCount*self.frameWidth*self.frameHeight > 1000*500*500:
        #    print("b/a",self.frameCount)
        #    self.frameCount = int(1000*500*500/self.frameWidth/self.frameHeight)
        #    print("b/a", self.frameCount)

        self.cap = cap

    def __iter__(self):
        return self

    def __next__(self):
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.frameCount:
            self.fc = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, buf = self.cap.read()
        # print(buf.shape)
        buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
        self.fc += 1
        return buf


def gif_to_numpy(file):
    return mp4_to_numpy(file)


def find_score(filename, titlewords="", user=""):
    matchscore = re.search(r"(?<=score-)[0-9]{1,}(?=-)", filename)
    if len(titlewords) > 0:
        matchtitle = re.search(r"(?<=title-).{1,}(?=-)", filename)
        if matchtitle:
            if not re.search(titlewords.lower(), matchtitle.group().lower()):
                return -1
        else:
            matchtitle = re.search(r"(?<=title-).{1,}", filename)
            if matchtitle:
                if not re.search(titlewords.lower(), matchtitle.group().lower()):
                    return -1
            else:
                return -1
    if len(user) > 0:
        matchuser = re.search(r"(?<=user-).{1,}?(?=-)", filename)
        if matchuser:
            if not re.search(user.lower(), matchuser.group().lower()):
                return -1
        else:
            return -1
    if matchscore:
        return int(matchscore.group())
    else:
        return -1
