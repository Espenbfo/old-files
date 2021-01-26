import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
input_image = ""

def kernelize(img,kern,padding=False):
    print(img)
    kern_dim = kern.shape[0]

    new_img = np.zeros((img.shape[0]-kern_dim+1,img.shape[1]-kern_dim+1,3))
    print(new_img.shape)
    for i in range(len(new_img)):
        for i2 in range(len(new_img[0])):
            for j in range(kern_dim):
                for k in range(kern_dim):
                    #try:
                    new_img[i][i2] += img[i+j][i2+k]*kern[j][k]
                    #except:
                    #    pass
                        #print(i,i2,j,k)
    return new_img

kernel_dim = 11
kernel = np.array([[abs(-kernel_dim/2 + i) - abs(-kernel_dim/2 + j) for i in range(kernel_dim)] for j in range(kernel_dim)])#np.array([[1,2,3],[4,5,6],[7,8,9]])
print(kernel)
#kernel = kernel/kernel.sum()

image = cv2.imread(input_image)/255.

#image = kernelize(image,kernel)
image = cv2.filter2D(image,-1,kernel)
#print(image)
cv2.imshow("1",image)

cv2.waitKey(0)
