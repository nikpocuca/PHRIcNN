#!/bin/env python

from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import keras 


imageInput = layers.Input(shape = (200,200,1),name = "imageInput")

L1_1 = layers.Conv2D(32,(3,3), activation ='relu',input_shape = (200,200,1))(imageInput)
L1_2 = layers.MaxPooling2D((2,2))(L1_1)
L1_3 = layers.Conv2D(64,(3,3),activation = 'relu')(L1_2)
L1_4 = layers.MaxPooling2D((2,2))(L1_3)
LF = layers.Flatten()(L1_4)
LD = layers.Dense(1024,activation='relu')(LF)

R1_1 = layers.Conv2D(32,(3,3), activation ='relu',input_shape = (200,200,1))(imageInput)
R1_2 = layers.MaxPooling2D((2,2))(R1_1)
R1_3 = layers.Conv2D(64,(3,3),activation = 'relu')(R1_2)
R1_4 = layers.MaxPooling2D((2,2))(R1_3)
R1_5 = layers.Conv2D(128, (3, 3), activation='relu')(R1_4)
R1_6 = layers.MaxPooling2D((2,2))(R1_5) 
R1_7 = layers.Conv2D(128, (3, 3), activation='relu')(R1_6)
R1_8 = layers.MaxPooling2D((2,2))(R1_7) 
R1_9 = layers.Conv2D(128, (3, 3), activation='relu')(R1_8)
R1_10 = layers.MaxPooling2D((2,2))(R1_9) 
RF = layers.Flatten()(R1_10)
RD = layers.Dense(1024,activation='relu')(RF)


CLayer = layers.concatenate([LD,RD])
C1 = layers.Dense(2048,activation='relu')(CLayer)
C2 = layers.Dense(256,activation='relu')(C1)
C3 = layers.Dense(1,activation='sigmoid')(C2)

model = keras.models.Model(inputs = [imageInput], outputs = [C3])


#plot_model(model,to_file = 'split.png',show_shapes = True)

model.summary()

from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adadelta(), #optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

train_dir = 'data/train'
validation_dir = 'data/test'

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 156, 126. Size set manually, 
        #depends on data.
        target_size=(200, 200),
        color_mode="grayscale",
        batch_size=40,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(200, 200),
        color_mode="grayscale",
        batch_size=20,
        class_mode='binary')


for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

stop_early = EarlyStopping(monitor="val_loss",
                            min_delta=0,
                            patience=4,
                            verbose=0,
                            mode="auto")
                            #baseline=None)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=188,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=32.6,
      callbacks=[stop_early])



