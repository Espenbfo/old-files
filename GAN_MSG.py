import numpy as np
import pickle
import cv2

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,BatchNormalization,LeakyReLU,Reshape,Conv2DTranspose,Dropout, Embedding, Concatenate, Activation, AveragePooling2D, ReLU, UpSampling2D
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop, Adam
import os
from tensorflow.keras.preprocessing import image
import random



IMAGE_DIR="result_MSG"
INPUT_DIRS=["just_bordered","legal_cut","legal_cut2","collection_cut"] #"legal_cut3"
NUM_IMAGES = 100000
NUM_SHIFT = 100000
LOAD_IMAGES = True

LOAD_WEIGHTS=False
INPUT_MODEL="gan70step_3000.h5"
LATENT_DIM=512
MODEL_SAVE="weights/MSG.h5"

depth=1
steps_pr_depth = [500,1000,2000,4000,100000]
max_depth = len(steps_pr_depth)
print(max_depth)
ITERATIONS=200000
BATCH_SIZE=32

HEIGHT=64
WIDTH=64
CHANNELS=3

LEARNING_RATE=0.0001
TRAIN_ADV=True
LEARNING_RELATION_ADV=1
TRAIN_DISC=True
LEARNING_RELATION_DISC=2
TRAIN_GEN=False
LEARNING_RELATION_GEN = 1


if LOAD_IMAGES:
    data = []
    i = 0
    for i in range(max_depth):
        data.append([])
    for input_dir in INPUT_DIRS:
        for path in os.listdir(input_dir):
            img = image.load_img(os.path.join(input_dir,path))
            img = image.img_to_array(img)/(255./2.)-1
            shape = img.shape
            for j in range(max_depth):
                size = 4 * (2 ** j)
                xy = shape[0]/shape[1]
                if (xy < 1):
                    y = size
                    x = round(y*xy)
                else:
                    x = size
                    y = round(x / xy)
                img2 = cv2.resize(img,(y,x))
                pady = size-y
                padx = size-x
                img2 = np.pad(img2,((padx//2,padx//2+padx%2),(pady//2,pady//2+pady%2),(0,0)),constant_values=((0,0),(0,0),(0,0)))
                data[j].append(img2)

            if (i+1)%10000==0:
                print(str(i) + " images loaded")
            i += 1

        if (i+1)%10000==0:
            print(str(i) + " images loaded")

data = [np.array(data[i]) for i in range(max_depth)]
print(len(data))
print(len(data[0]))
print(len(data[max_depth-1]))
#print(data[0][:10])
#print(data[max_depth-1][:10])
#print(data.shape[0])
#print(data.shape[max_depth-1])
def generator(latent_dim=512):

    outputs = []
    input_latent = Input((latent_dim))

    gen = Reshape((4,4,latent_dim//16))(input_latent)

    gen = Conv2D(256,3,padding="same",name="first_conv")(gen)
    gen = LeakyReLU()(gen)

    gen = Conv2D(256,3,padding="same",name="second_conv")(gen)
    gen = LeakyReLU()(gen)

    #outputs.append(Conv2D(3,3,padding="same",activation="tanh")(gen))

    for i in range(depth-1):
        gen = UpSampling2D(2)(gen)
        scale = 1
        if (i > 3):
            scale = i-3
        gen = Conv2D(256//(scale**2),3,padding="same",name="conv" + str(i*2))(gen)
        gen = LeakyReLU()(gen)

        gen = Conv2D(256//(scale**2),3,padding="same",name="conv" + str(i*2+1))(gen)
        gen = LeakyReLU()(gen)

    out = Conv2D(3,3,padding="same",activation="tanh")(gen)



    gen = Model(input_latent,out)

    #plot_model(gen, to_file="model.png")
    return gen


def discriminator():

    inp = Input((2 * (2 ** depth), 2 * (2 ** depth), 3))
    disc = Conv2D(256,3,padding="same",name="first_conv")(inp)
    disc = LeakyReLU(alpha=0.2)(disc)
    disc = Conv2D(256,3,padding="same",name="second_conv")(disc)
    disc = LeakyReLU(alpha=0.2)(disc)
    disc = AveragePooling2D()(disc)
    print(disc.shape)

    for i in range(depth-2,-1,-1):

        disc = Conv2D(256//((i+1)**2),3,padding="same",name="conv" + str(i*2))(disc)
        disc = LeakyReLU(alpha=0.2)(disc)
        disc = Conv2D(256//((i+1)**2),3,padding="same",name="conv" + str(i*2+1))(disc)
        disc = LeakyReLU(alpha=0.2)(disc)
        disc = AveragePooling2D()(disc)

    disc = Flatten()(disc)
    disc = Dense(1,activation="sigmoid")(disc)

    disc = Model(inp,disc)

    return disc


def cGan(generator_model,discriminator_model,learning_rate):
    discriminator_model.trainable=False
    input_latent = generator_model.input

    gen_outputs = generator_model.output

    generator_output = generator_model(input_latent)
    gan_output = discriminator_model(generator_output)

    gan = Model(input_latent, gan_output)

    gan_optimizer = RMSprop(lr=learning_rate * LEARNING_RELATION_ADV)
    gan.compile(optimizer=gan_optimizer, loss="binary_crossentropy")

    return gan


for steps in steps_pr_depth:
    gen = generator()
    disc = discriminator()
    discriminator_optimizer = RMSprop(lr=LEARNING_RATE)

    disc.compile(discriminator_optimizer, loss="binary_crossentropy")
    gan = cGan(gen,disc,LEARNING_RATE)

    print(gen.summary())
    print(disc.summary())
    print(gan.summary())

    start = 0

    if LOAD_WEIGHTS:
        gan.load_weights(MODEL_SAVE)
    if (depth > 1):
        gan.load_weights(MODEL_SAVE,by_name=True)
        print("loaded")
    for step in range(1,steps+1):

        stop=start+BATCH_SIZE
        real_images=data[depth-1][start:stop]


        latent_space = np.random.normal(size=(BATCH_SIZE,LATENT_DIM))
        generated_images=gen.predict(latent_space)



        combined_images=[np.concatenate((generated_images,real_images),axis=0)]
        #print(len(combined_images[0]))
        labels=np.concatenate([np.ones((BATCH_SIZE))*0.92-0.05*np.random.random((BATCH_SIZE)),
                               np.zeros((BATCH_SIZE))+0.08+0.05*np.random.random((BATCH_SIZE))])


        d_loss=disc.train_on_batch(combined_images,labels)


        misleading_targets= np.zeros((BATCH_SIZE))

        latent_space = np.random.normal(size=(BATCH_SIZE,LATENT_DIM))

        a_loss = gan.train_on_batch(latent_space,misleading_targets)


        start += BATCH_SIZE
        if start > len(data[depth]) - BATCH_SIZE:
            start=0

        if step%10==0:
            print("step: ", step)
            print("discriminator loss:", d_loss)
            print("adversial loss", a_loss)
        if step%100 == 0:
            img= image.array_to_img((generated_images[-1]+1)*(255./2), scale=False)
            img.save(os.path.join(IMAGE_DIR,"generated" + str(len(os.listdir(IMAGE_DIR))) + "step_" + str(step)  +  ".jpg"))
            img= image.array_to_img((generated_images[-2]+1)*(255./2), scale=False)
            img.save(os.path.join(IMAGE_DIR,"generated" + str(len(os.listdir(IMAGE_DIR))) + "step_" + str(step)  +  ".jpg"))
            img= image.array_to_img((real_images[-1]+1)*(255./2), scale=False)
            img.save(os.path.join(IMAGE_DIR,"real" + str(len(os.listdir(IMAGE_DIR))) + "step_" + str(step)  +  ".jpg"))
    gan.save_weights(MODEL_SAVE)
    depth += 1
