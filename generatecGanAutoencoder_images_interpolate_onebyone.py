import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation,BatchNormalization,LeakyReLU,Reshape,Conv2DTranspose,Dropout, Embedding, Concatenate,ReLU
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical
import cv2


import os
from tensorflow.keras.preprocessing import image

LATENT_DIM=122 #Størrelsen på noisen
#Bilde størrelse:
SIZE=64
HEIGHT=64
WIDTH=64
CHANNELS=3

LEARNING_RATE=0.0002
TRAIN_ADV=True
LEARNING_RELATION_ADV=1
TRAIN_DISC=True
LEARNING_RELATION_DISC=2
TRAIN_GEN=False
LEARNING_RELATION_GEN = 1
BATCH_SIZE=20
INTERPOLATION_LENGTH=25

#model

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





gen = generator()
disc = discriminator(HEIGHT,WIDTH,CHANNELS,6,LEARNING_RATE)

gan = cGan(gen,disc,LEARNING_RATE)

gan.load_weights("weights\\gan.h5")

INPUT_DIR="imdb_resized"


latent_vectors=np.random.normal(size=(BATCH_SIZE,LATENT_DIM))
data = []
for i in range(INTERPOLATION_LENGTH*BATCH_SIZE-INTERPOLATION_LENGTH):
    data.append(latent_vectors[(i//INTERPOLATION_LENGTH)]*(INTERPOLATION_LENGTH-1-i%INTERPOLATION_LENGTH)/(INTERPOLATION_LENGTH-1)+latent_vectors[1+i//INTERPOLATION_LENGTH]*((i%INTERPOLATION_LENGTH)/(INTERPOLATION_LENGTH-1)))
data = np.array(data)
print(data)
print(data.shape)
generated_images=gen.predict(data)


save_dir="result_cGAN_generation\\"

x= 0
for i in range(len(generated_images)):
    x+=1

    i = generated_images[i]
    i=(i+1)/2.
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)

    i*=255
    i = cv2.resize(i,(512,512))
    cv2.imwrite(save_dir+str(x)+".jpg", i)

import imageio

images = []
for filename in range(INTERPOLATION_LENGTH*BATCH_SIZE-INTERPOLATION_LENGTH):
    images.append(imageio.imread(save_dir+str(filename+1)+".jpg"))

imageio.mimsave(save_dir+'movie.gif', images)


