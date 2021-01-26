import cv2
import numpy as np
import os
i = 0

os.makedirs("altered/video_td",exist_ok=True)
def generate_link(image_index=i):
    return "altered/video_td/00000"[:-len(str(image_index))] + str(image_index) +".jpg"
input_dir = "filer"
for path in os.listdir(input_dir):
    try:
        if path[-3:] == "mp4":
            try:
                cap = cv2.VideoCapture(os.path.join(input_dir,path))
            except Exception as e:
                print(e)
                continue
        else:
            continue
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

        fc = 0
        ret = True

        while (fc < frameCount  and ret):
            ret, buf[fc] = cap.read()
            fc += 1
        #print(buf)
        cap.release()


        for b in buf[::5]:
            cv2.imwrite(generate_link(i),b)
            i += 1
            if i % 1000 == 0:
                print(i)
    except Exception as e:
        print(e)