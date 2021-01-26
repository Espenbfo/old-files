from keras.layers import Conv2D,Dense,Dropout,BatchNormalization,Flatten,MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
number_of_crops = 32
learning_rate = 0.00025
pixel_crop = 8
def generate_model(size=128):
    x,y = size,size#pixel_crop*number_of_crops,pixel_crop*number_of_crops
    m = Sequential()
    m.add(Conv2D(64,kernel_size=(3,3),padding="same",input_shape=(x,y,3),activation="relu"))
    m.add(BatchNormalization())
    m.add(Conv2D(128, kernel_size=(3, 3), strides=(2,2), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(Flatten())
    m.add(Dropout(0.2))
    m.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(learning_rate=learning_rate)
    m.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])
    return m

def generate_model2(size=128):
    x,y = size,size#pixel_crop*number_of_crops,pixel_crop*number_of_crops
    m = Sequential()
    m.add(Conv2D(64,kernel_size=(3,3),padding="same",input_shape=(x,y,3),activation="relu"))
    m.add(BatchNormalization())
    m.add(Conv2D(64, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(128, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(128, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(Conv2D(256, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(256, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(256, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(Flatten())
    #m.add(Dropout(0.2))
    m.add(Dense(4, activation="sigmoid"))
    optimizer = Adam(learning_rate=learning_rate)
    m.compile(optimizer=optimizer,loss="mean_squared_error",metrics=["accuracy"])
    return m
def generate_model3(size=128):
    x,y = size,size#pixel_crop*number_of_crops,pixel_crop*number_of_crops
    m = Sequential()
    m.add(Conv2D(64,kernel_size=(3,3),padding="same",input_shape=(x,y,3),activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(64, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(128, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(256, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(512, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(Flatten())
    m.add(Dropout(0.2))
    m.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(learning_rate=learning_rate)
    m.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])
    return m
def generate_classifier(size=128,classes = 4):
    x,y = size,size#pixel_crop*number_of_crops,pixel_crop*number_of_crops
    m = Sequential()
    m.add(Conv2D(64,kernel_size=(3,3),padding="same",input_shape=(x,y,3),activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(64, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(128, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(256, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(MaxPool2D())
    m.add(Conv2D(512, kernel_size=(3, 3), padding="same", input_shape=(x, y, 3), activation="relu"))
    m.add(BatchNormalization())
    m.add(Flatten())
    m.add(Dense(classes, activation="softmax"))
    optimizer = Adam(learning_rate=learning_rate)
    m.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
    return m

def generate_vgg(size=128):
    from keras.applications import VGG19
    vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = Sequential()
    model.add(vgg19)
    model.add(Flatten())
    model.add(Dense(512,activation="sigmoid"))
    model.add(Dropout(0.3))
    model.add(Dense(4,activation="sigmoid"))
    model.compile("adam","mean_squared_error",metrics=["accuracy"])
    print(model.summary())
    return model

def generate_upscale2():
    m = Sequential()
    m.add(Conv2D(64,(11,11),padding="same",input_shape=(512,512,3),activation="relu"))
    m.add(Conv2D(32,(1,1),padding="same",activation="relu"))
    m.add(Conv2D(3,(5,5),padding="same",activation="tanh"))
    return m
