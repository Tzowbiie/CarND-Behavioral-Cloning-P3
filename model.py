import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randrange

lines = []
with open('/root/Desktop/Data/driving_log.csv') as csvfile:
#with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # not considering data with throttle below 0.2
        #if line['throttle']>0.2:
        lines.append(line)

""" left, right and center camera images are loaded and added to a list.
    
    The steering angle of the Left and right camera images are corrected with a hyperparameter.
"""
images = []
measurements = []
#Steering hyperparameter
correction = 0.21 # this is a parameter to tune
curvature_limit = 0.1
for line in lines:
    source_path = line[0] #center image
    source_pathl = line[1] #left image
    source_pathr = line[2] #right image
    filename = source_path.split('/')[-1]
    filenamel = source_pathl.split('/')[-1]
    filenamer = source_pathr.split('/')[-1]
    current_path = '/root/Desktop/Data/IMG/' + filename
    current_pathl = '/root/Desktop/Data/IMG/' + filenamel
    current_pathr = '/root/Desktop/Data/IMG/' + filenamer
    #current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/data' + filename
    #current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/data' + filenamel
    #current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/data' + filenamer
    image = cv2.imread(current_path,1)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    imagel = cv2.imread(current_pathl,1)
    imagel = cv2.cvtColor(imagel,cv2.COLOR_BGR2RGB)
    imager = cv2.imread(current_pathr,1)
    imager = cv2.cvtColor(imager,cv2.COLOR_BGR2RGB)
    #cropping images randomly to improve robustness
    rand_crop = randrange(15)
    #print('Random ',rand_crop)
    rand_crop1 = randrange(15)
    # Remove the top 20 and bottom 20 pixels of 160x320x3 images
    image = image[20+rand_crop:135+rand_crop1, :, :]
    # Resize the image to match input layer of the model
    resize = (200, 66)
    image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    rand_crop = randrange(15)
    rand_crop1 = randrange(15)
    #left image
    imagel = imagel[20+rand_crop:135+rand_crop1, :, :]
    # Resize the image to match input layer of the model
    imagel = cv2.resize(imagel, resize, interpolation=cv2.INTER_AREA)   
    #right image
    rand_crop = randrange(15)
    rand_crop1 = randrange(15)
    # Remove the top 20 and bottom 20 pixels of 160x320x3 images
    imager = imager[20+rand_crop:135+rand_crop1, :, :]
    # Resize the image to match input layer of the model
    imager = cv2.resize(imager, resize, interpolation=cv2.INTER_AREA) 
    measurement = float(line[3])
    #adding image to images container
    images.append(image)
    measurements.append(measurement)
    images.append(imagel)
    measurements.append(measurement+correction)
    images.append(imager) 
    measurements.append(measurement-correction)
    #flipping images and steering angle to balance right and left curves
aug_images, aug_measurements, bright_image=[],[],[]
for image, measurement in zip(images, measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    aug_images.append(cv2.flip(image,1))
    aug_measurements.append(measurement*-1.0)

#plt.imsave('/home/workspace/CarND-Behavioral-Cloning-P3/TestImages/test0.jpg',aug_images[0],format="jpg")
#plt.imsave('/home/workspace/CarND-Behavioral-Cloning-P3/TestImages/test1.jpg',aug_images[1],format="jpg")
#plt.imsave('/home/workspace/CarND-Behavioral-Cloning-P3/TestImages/test2.jpg',aug_images[2],format="jpg")
#plt.imsave('/home/workspace/CarND-Behavioral-Cloning-P3/TestImages/test3.jpg',aug_images[3],format="jpg")
#plt.imsave('/home/workspace/CarND-Behavioral-Cloning-P3/TestImages/test4.jpg',aug_images[4],format="jpg")
#plt.imsave('/home/workspace/CarND-Behavioral-Cloning-P3/TestImages/test5.jpg',aug_images[5],format="jpg")

X_train = np.array(aug_images)
y_train = np.array(aug_measurements)
                              
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras import optimizers

model = Sequential()
#adding a Lambda Layer for Normalisation and Mean Centering
model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(66,200,3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu', border_mode="valid", name='conv1'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu', border_mode="valid", name='conv2'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu', border_mode="valid", name='conv3'))
model.add(Convolution2D(64,3,3, activation='relu', border_mode="valid", name='conv4'))
model.add(Convolution2D(64,3,3, activation='relu', border_mode="valid", name='conv5'))
#model.add(Dropout(0.7, name='drop1'))
model.add(Flatten())
model.add(Dense(100, name='fullc1'))
#model.add(Dropout(0.5, name='drop2'))
model.add(Dense(50, name='fullc2'))
model.add(Dropout(0.5, name='drop3'))
model.add(Dense(10, name='fullc3'))
model.add(Dropout(0.5, name='drop4'))
model.add(Dense(1,name='output'))

print(model.summary())

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)      
model.compile(loss='mse', optimizer=adam)
model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=9, verbose = 1)               

model.save('model_final.h5')

#with open('model.json', 'w') as outfile:
#    json.dump(model.to_json(), outfile)
exit()