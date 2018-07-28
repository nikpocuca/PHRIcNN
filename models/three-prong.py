
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import keras 


imageInput = layers.Input(shape = (200,200,1),name = "imageInput")



#  Base Convolutional Layer
B_1 = layers.Conv2D(32,(3,3), activation ='relu',input_shape = (200,200,1))(imageInput)
B_2 = layers.MaxPooling2D((2,2))(B_1)
B_3 = layers.Conv2D(64,(3,3),activation = 'relu')(B_2)
B_4 = layers.MaxPooling2D((2,2))(B_3)
B_5 = layers.Conv2D(128, (3, 3), activation='relu')(B_4)
B_6 = layers.MaxPooling2D((2,2))(B_5) 
 

# Split layer. 
C_1 = layers.Conv2D(128,(3,3), activation ='relu')(B_6)
N_1 = layers.Conv2D(128,(3,3), activation ='relu')(B_6)
H_1 = layers.Conv2D(128,(3,3), activation ='relu')(B_6)


# Contours 
C_2 = layers.MaxPooling2D((2,2))(C_1)
C_3 = layers.Conv2D(128,(3,3), activation ='relu')(C_2) 
C_4 = layers.MaxPooling2D((2,2))(C_3)
C_5 = layers.Conv2D(128,(3,3), activation ='relu')(C_4)
C_6 = layers.MaxPooling2D((2,2))(C_5)
CF =  layers.Flatten()(C_6)
CD_1 = layers.Dense(1024,activation = 'relu')(CF)
CD_2 = layers.Dense(256, activation = 'relu')(CD_1)
CS = layers.Dense(1,activation='sigmoid')(CD_2)



# Numbers
N_2 = layers.MaxPooling2D((2,2))(N_1)
N_3 = layers.Conv2D(128,(3,3), activation ='relu')(N_2) 
N_4 = layers.MaxPooling2D((2,2))(N_3)
N_5 = layers.Conv2D(128,(3,3), activation ='relu')(N_4)
N_6 = layers.MaxPooling2D((2,2))(N_5)
NF =  layers.Flatten()(N_6)
ND_1 = layers.Dense(1024,activation = 'relu')(NF)
ND_2 = layers.Dense(256, activation = 'relu')(ND_1)
NS = layers.Dense(1,activation='sigmoid')(ND_2)


# Hands 
H_2 = layers.MaxPooling2D((2,2))(H_1)
H_3 = layers.Conv2D(128,(3,3), activation ='relu')(H_2) 
H_4 = layers.MaxPooling2D((2,2))(H_3)
H_5 = layers.Conv2D(128,(3,3), activation ='relu')(H_4)
H_6 = layers.MaxPooling2D((2,2))(H_5)
HF =  layers.Flatten()(H_6)
HD_1 = layers.Dense(1024,activation = 'relu')(HF)
HD_2 = layers.Dense(256, activation = 'relu')(HD_1)
HS = layers.Dense(1,activation='sigmoid')(HD_2)

model = keras.models.Model(inputs = [imageInput], outputs = [CS,NS,HS])


plot_model(model,to_file = 'split.png',show_shapes = True)

model.summary()

#from keras import optimizers

#model.compile(loss='binary_crossentropy',
#              optimizer=keras.optimizers.Adadelta(), #optimizers.RMSprop(lr=1e-4),
#              metrics=['accuracy'])

