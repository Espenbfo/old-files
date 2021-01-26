import numpy as np
import pickle
import cv2

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,BatchNormalization,LeakyReLU,Reshape,Conv2DTranspose,Dropout, Embedding, Concatenate, Activation, MaxPooling2D, ReLU
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop, Adam
import os
from tensorflow.keras.preprocessing import image
import random



IMAGE_DIR="result_cGAN"
INPUT_DIRS=["just_bordered","legal_cut","legal_cut2","collection_cut"] #"legal_cut3"
NUM_IMAGES = 100
NUM_SHIFT = 0
LOAD_IMAGES = True
SAVE_DIR = "gan.h5"

LOAD_WEIGHTS=True
INPUT_MODEL=SAVE_DIR
LATENT_DIM=122
MODEL_SAVE="weights"

#Ordrenummer: 83217134
ITERATIONS=200000
BATCH_SIZE=32

SIZE=64
HEIGHT=64
WIDTH=64
CHANNELS=3

LEARNING_RATE=0.00005
TRAIN_ADV=True
LEARNING_RELATION_ADV=1
TRAIN_DISC=True
LEARNING_RELATION_DISC=2
TRAIN_GEN=False
LEARNING_RELATION_GEN = 1

if LOAD_IMAGES:
    data = []
    i = 0
    for input_dir in INPUT_DIRS:
        for path in os.listdir(input_dir):
            img = image.load_img(os.path.join(input_dir,path))
            img = image.img_to_array(img)/(255./2.)-1
            shape = img.shape
            xy = shape[0]/shape[1]
            if (xy < 1):
                y = SIZE
                x = int(y*xy)
            else:
                x = SIZE
                y = int(x / xy)
            img = cv2.resize(img,(y,x))
            pady = SIZE-y
            padx = SIZE-x
            img = np.pad(img,((padx//2,padx//2+padx%2),(pady//2,pady//2+pady%2),(0,0)),constant_values=((0,0),(0,0),(0,0)))
            #print(img.shape)
            if type(img) != np.ndarray:
                print("wiki_resized\\" + str(i) + ".jpg")
                print("is " + str(type(img)))
                continue
            data.append(img)

            if (i+1)%10000==0:
                print(str(i) + " images loaded")
            i += 1




def generator(latent_dim=122):

    input_latent = Input((latent_dim,))

    gen = Dense(1024*4*4)(input_latent)
    gen = ReLU()(gen)
    gen = Reshape((4,4,1024))(gen)

    gen = Conv2DTranspose(512,4,strides=2,padding="same")(gen) #8,8
    gen = ReLU()(gen)

    gen = Conv2DTranspose(256,4,strides=2,padding="same")(gen) #8,8
    #gen = BatchNormalization()(gen)
    gen = ReLU()(gen)

    #gen = Conv2DTranspose(256,4,strides=2,padding="same")(gen) #8,8
    #gen = ReLU()(gen)

    gen = Conv2DTranspose(128,4,strides=2,padding="same")(gen) #8,8
    gen = ReLU()(gen)

    gen = Conv2DTranspose(CHANNELS,4,strides=2,padding="same")(gen) #8,8
    gen = Activation("tanh")(gen)

    gen = Model(input_latent,gen)

    #plot_model(gen, to_file="model.png")
    return gen


def discriminator(height,width,channels,learning_rate,embedding_dim=64):

    input_image = Input((height,width,channels))

    disc = Conv2D(128, 4,strides=2,padding="same")(input_image)
    disc = LeakyReLU()(disc)

    disc = Conv2D(256, 4,strides=2,padding="same")(disc)
    disc = LeakyReLU()(disc)

    #disc = Conv2D(256, 4,strides=2,padding="same")(disc)
    #disc = LeakyReLU()(disc)

    disc = Conv2D(512, 4,strides=2,padding="same")(disc)
    #disc = BatchNormalization()(disc)
    disc = LeakyReLU()(disc)

    disc = Conv2D(1024, 4,strides=2,padding="same")(disc)
    disc = LeakyReLU()(disc)

    disc = Flatten()(disc)

    disc = Dropout(0.4)(disc)

    disc = Dense(1, activation="sigmoid")(disc)

    disc = Model(input_image,disc)

    discriminator_optimizer = Adam(lr=LEARNING_RATE,beta_1=0.5)

    disc.compile(discriminator_optimizer, loss="binary_crossentropy",metrics=['binary_accuracy'])

    return disc


def cGan(generator_model,discriminator_model,learning_rate):
    discriminator_model.trainable=False
    input_latent = generator_model.input

    print(input_latent.shape)

    generator_output = generator_model(input_latent)
    gan_output = discriminator_model(generator_output)

    gan = Model(input_latent, gan_output)

    gan_optimizer = Adam(lr=learning_rate * LEARNING_RELATION_ADV,beta_1=0.5)
    gan.compile(optimizer=gan_optimizer, loss="binary_crossentropy")

    return gan


def encoder(height,width,channels,label_dim,latent_dim,generator_model):
    input_image = Input((height,width,channels))

    enc = Conv2D(128, 3,padding="same")(input_image)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU()(enc)

    enc = Conv2D(256, 4, strides=2,padding="same")(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU()(enc)

    enc = Conv2D(512, 4, strides=2,padding="same")(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU()(enc)

    enc = Conv2D(128, 4, strides=2,padding="same")(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU()(enc)

    enc = Flatten()(enc)
    enc = Dense((latent_dim))(enc)
    enc = Activation("tanh")(enc)

    enc = Model(input_image,enc)
    return enc


def fullEnkoder(encoder_model,generator_model,learning_rate):
    input_latent, input_label = generator_model.input

    input_image = encoder_model.input
    print(input_image.shape)
    generator_model.trainable = False
    encoder_output=encoder_model(input_image)
    output= generator_model([encoder_output, input_label])
    enc_gen = Model([input_image, input_label], output)

    generator_optimizer = Adam(lr=learning_rate)
    enc_gen.compile(generator_optimizer, loss="binary_crossentropy")

    return enc_gen




gen = generator()
disc = discriminator(HEIGHT,WIDTH,CHANNELS,LEARNING_RATE)

gan = cGan(gen,disc,LEARNING_RATE)
print(gan.summary())

print(gen.summary())
print(disc.summary())
start = 0

if LOAD_WEIGHTS:
    gan.load_weights(MODEL_SAVE+"\\"+INPUT_MODEL)

perm_latent = np.random.normal(size=(8,LATENT_DIM))
for step in range(ITERATIONS):

    stop=start+BATCH_SIZE
    real_images=data[start:stop]


    latent_space = np.random.normal(size=(BATCH_SIZE,LATENT_DIM))
    generated_images=gen.predict(latent_space)

    combined_images=np.concatenate([generated_images,real_images])

    labels=np.concatenate([np.ones((BATCH_SIZE))-0.05*np.random.random((BATCH_SIZE)),
                           np.zeros((BATCH_SIZE))+0.05*np.random.random((BATCH_SIZE))])



    if TRAIN_GEN:
        g_loss = enc_gen.train_on_batch(real_images,real_images)
    else:
        g_loss = None

    if TRAIN_DISC:
        d_loss=disc.train_on_batch(combined_images,labels)
    else:
        d_loss=None

    misleading_targets= np.zeros((BATCH_SIZE))

    if TRAIN_ADV:
        a_loss = gan.train_on_batch(latent_space,misleading_targets)
    else:
        a_loss = None


    start += BATCH_SIZE
    if start > len(data) - BATCH_SIZE:
        print("epoooooch")
        start=0

    if step%10==0:
        print("step: ", step)
        print("discriminator loss:", d_loss)
        print("adversial loss", a_loss)
        print("generator_loss", g_loss)
    if step%200 == 0:
        if(step%1000==0):
            gan.save_weights(MODEL_SAVE+"\\" + SAVE_DIR)
            same_images = gen.predict(perm_latent)
            for savimg in same_images:
                img = image.array_to_img((savimg + 1) * (255. / 2), scale=False)
                img.save(
                    os.path.join(IMAGE_DIR, "same" + str(len(os.listdir(IMAGE_DIR))) + "step_" + str(step) + ".jpg"))
            #img = image.array_to_img((real_images[0] + 1) * (255. / 2), scale=False)
            #img.save(os.path.join(IMAGE_DIR,
            #                      "real" + str(len(os.listdir(IMAGE_DIR))) + "step_" + str(step) + ".jpg"))

        img= image.array_to_img((generated_images[0]+1)*(255./2), scale=False)
        img.save(os.path.join(IMAGE_DIR,"generated" + str(len(os.listdir(IMAGE_DIR))) + "step_" + str(step)  +   ".jpg"))

