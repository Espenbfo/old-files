from model import generate_model3,generate_model2,generate_model,generate_classifier
from keras.preprocessing import image
import numpy as np
import os
import cv2
import json
input_dirs = []
save_dir = "collection_cut3_2"
save_orig = False
full_save_dir = "full_save_dir"
cutoff = 0.8
m2 = generate_model2()
m = generate_model3()
m3 = generate_model()
m.load_weights("weights/lastBry.h5")
m2.load_weights("weights/lastNoDropout.h5")
m3.load_weights("weights/lastChecker3.h5")
saving_data = False
save_file = "metadata2.txt"
if not os.path.exists(save_dir):
    print("creating_save_dir")
    os.makedirs(save_dir)
if not os.path.exists(full_save_dir):
    print("creating_save_dir")
    os.makedirs(full_save_dir)
def use_cutoff(imgname,cutoff=0.5):
    if imgname[-3:] != "jpg":
        return None, False
    try:
        img = image.load_img(imgname)
    except:
        return None, False
    img = image.img_to_array(img)

    img2 = cv2.resize(img, (128, 128))
    rating = m.predict(np.array([img2]))[0]
    print(rating)
    if (rating < cutoff):
        return img, False
    return img, True

def extract_box(img, increase = 0.04):
    try:
        #if imgname[-3:] != "jpg":
        #    return None,None,None,None,None,None
        #try:
        #    img = image.load_img(imgname)
        #except:
        #    return None,None,None,None,None,None
        #img = image.img_to_array(img)
        shape = img.shape
        img2 = cv2.resize(img,(128,128))
        #print(img.shape)
        y_warp = shape[0]
        x_warp = shape[1]

        predict = m2.predict(np.array([img2]))[0]
        #print(predict)
        color = 1
        predict[0] -= increase
        predict[2]+= increase
        predict[1]-= increase/3
        predict[3]+= increase/3
        predict[0] = max(0,min(1,predict[0]))
        predict[2] = max(0,min(1,predict[2]))
        predict[1] = max(0,min(1,predict[1]))
        predict[3] = max(0,min(1,predict[3]))
        predict[0]*=x_warp
        predict[2]*=x_warp
        predict[1]*=y_warp
        predict[3]*=y_warp
        #print(predict)
        #print(imgname)
        img3 = img[int(predict[1]):int(predict[3]),int(predict[0]):int(predict[2])]
        #print(img3,int(predict[0]),int(predict[1]),int(predict[2]),int(predict[3]),img)
        return img3,int(predict[0]),int(predict[1]),int(predict[2]),int(predict[3]),img
    except Exception as e:
        print(e)
        print(imgname)
        return None,None,None,None,None,None

def double_check(img):
    img2 = cv2.resize(img, (128, 128))
    predict = m3.predict(np.array([img2]))[0]
    #print(predict)
    if predict < 0.9:
        #print("nope")
        return img, False
    #print("yup")
    return img, True

def classify(img,cutoff=0.9):
    img2 = cv2.resize(img, (128, 128))
    pred = m.predict(np.array([img2]))[0]
    rating = pred[0]
    if (np.argmax(pred) != 0 or rating < cutoff):
        return img, False
    print(pred)
    print(rating)
    return img, True

def crop_double_check(img):
    img, x1, y1, x2, y2, original_img = extract_box(img)
    if img is None:
        return None
    if (len(img) == 0):
        return None
    img, cut = double_check(img)
    if cut:
        return img
if __name__ == "__main__":
    i = 0
    paths = []
    for path in input_dirs:
        temppaths=[]
        for path2 in os.listdir(path):
            if not "." in path2:
                for path3 in os.listdir(path + "/" + path2):
                    if path3[-3:] == "jpg":
                        paths.append(path + "/" + path2 + "/" + path3)
            if path2[-3:] == "jpg":
                paths.append(path + "/" + path2)
    print(len(paths))
    data = []
    j = 0
    for path in paths:
        #img,passed = use_cutoff(path)
        passed = True

        img,x1,y1,x2,y2,original_img = extract_box(path)
        if img is None:
            continue
        if (len(img) == 0):
            continue
        img, cut = double_check(img)
        if not passed:
            if not img is None:
                j += 1
                img = image.array_to_img(img)
                img.save(os.path.join("bry_filtered", str(j) + ".jpg"))
            continue
        if cut:
            path = "00000"[:5-len(str(i))]+str(i)+".jpg"
            img = image.array_to_img(img)
            img.save(os.path.join(save_dir,path))
            if save_orig:
                original_img = image.array_to_img(original_img)
                original_img.save(os.path.join(full_save_dir,path))
            if (saving_data):
                data.append((i,((x1,y1),(x2,y2))))
            i += 1
            print(i)
            if saving_data:
                if i % 1000 == 0:
                    print("saving",i)
                    with open(save_file, "w") as f:
                        json.dump(data, f)
        else:
            pass
            #img = image.array_to_img(img)
            #img.save(os.path.join("resized4",path))