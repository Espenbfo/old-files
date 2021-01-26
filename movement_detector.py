import numpy as np
import cv2
import scipy.stats as st
from push_notification import pushbullet_message
import time

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
fgbg.setHistory(2000)


numlastimages = 10
lastimages = []
threshold = 0.6
writing = False

prevmask = None
capture_time = 10
capture_start = time.time()
last_time = time.time()

print(fgbg.getBackgroundRatio())
while(1):
    ret, frame = cap.read()
    if frame is None:
        break
    mask = fgbg.apply(frame)

    if not prevmask is None:

        fgmask = mask
    else:
        fgmask = mask
        prevmask = mask

    fgmask = cv2.GaussianBlur(fgmask,(41,41),0)
    fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)[1]
    #fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,(20,20))
    #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))#np.ones((30,30))
    kernel1 = np.ones((50,50))
    kernel2 = np.ones((100,100))#cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))#np.ones((60,60))
    #kernel3 = np.ones((30,30))
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel3)
    #fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_DILATE,kernel1)
    h, w = fgmask.shape[:2]

    #mask = np.zeros((h + 2, w + 2), np.uint8)

    #fgmask = np.uint8(fgmask)
    #mask = diff=cv2.absdiff(frame,fgbg.getBackgroundImage())
    #print(mask.shape)
    #mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    #fgmask = cv2.threshold(fgmask,10,255,cv2.THRESH_BINARY)[1]
    #cv2.floodFill(fgmask, mask, (0,0), 255);
    #print(mask)
    #fgmask += mask
    #fgmask = fillhole(fgmask)
    lastimages.append(fgmask)
    if len(lastimages) > numlastimages:
        lastimages.pop(0)
    fgmask = lastimages[0]
    for i in lastimages[1:]:
        fgmask+=i
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel2)
    thesum = fgmask.sum()/(fgmask.shape[0]*fgmask.shape[1])/255.
    if(thesum > threshold):
        bevegelse = False
        if not writing:
            writing=True
            print("bevegelsessensor\\"+str(time.time())[4:10]+".avi")
            print((frame.shape[1], frame.shape[0]))
            writer = cv2.VideoWriter("bevegelsessensor\\"+str(time.time())[4:10]+".avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, (frame.shape[1], frame.shape[0]))
        print("Bevegelse!")
        cv2.imshow('frame', cv2.bitwise_and(frame, frame, mask=fgmask))
        last_time = capture_start = time.time()
    else:
        cv2.destroyWindow("frame")
    if writing:
        writer.write(frame)
        if time.time()-capture_start > capture_time:
            print("stop_record")
            writer.release()
            writing=False
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    prevmask += mask-3

cv2.imshow('frame', fgbg.getBackgroundImage())

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
