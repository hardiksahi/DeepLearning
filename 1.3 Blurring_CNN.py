## Hardik Sahi
## ID: 20743327
## Intro to ML
## Section 002 

# Importing the required libraries
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFilter
from keras.datasets import mnist

## Using 6000 training examples and 1000 test examples instead of complete training and test set
dataCountTrain = 6000
dataCountTest = 1000
class_num = 10

## Loading MNIST data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

xTrainTemp = []

### DATA PREPROCESSING STARTS
## Converting the data into 32*32 from 28*28
for i in range(dataCountTrain):
	X_trainImage = Image.fromarray(x_train[i])
	new_image = X_trainImage.resize((32,32), Image.HAMMING)
	img_array = new_image.convert('L')
	img_array = np.array(img_array)
	xTrainTemp.append(img_array)
    
x_train = np.array(xTrainTemp)
y_train = y_train[0:dataCountTrain]
y_test = y_test[0:dataCountTest]

img_row_count = img_column_count = 32

# Because MNIST data consists of grayscale images, we add depth dimension to it by reshaping depending on imageFormat
if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_row_count, img_column_count)
	input_shape = (1, img_row_count, img_column_count)
else:
	x_train = x_train.reshape(x_train.shape[0], img_row_count, img_column_count, 1)
	input_shape = (img_row_count, img_column_count, 1)
    
#Convert class vectors to binary class matrices  and convert them to float32 and normalize them 
y_train = keras.utils.to_categorical(y_train, class_num)
y_test = keras.utils.to_categorical(y_test, class_num)
x_train = x_train.astype('float32')
x_train /= 255
### DATA PREPROCESSING ENDS

### CREATING CNN LAYERS 
#Step1: Initialise the CNN
classifierModel = Sequential()

#Adding different layers in the CNN

#Layer 1 Conv layer
classifierModel.add(Conv2D(64, (3,3), input_shape = input_shape, activation='relu',padding='same'))

#Layer2 MaxPooling
classifierModel.add(MaxPooling2D(pool_size=(2,2)))

#Layer3 Conv Layer
classifierModel.add(Conv2D(128, (3,3), activation='relu',padding='same'))

#Layer4 MaxPooling
classifierModel.add(MaxPooling2D(pool_size=(2,2)))

#Layer5 Conv Layer
classifierModel.add(Conv2D(256, (3,3), activation='relu',padding='same'))

## Commenting the layers 6 through 12 because system was not able to compute with many layers
#Layer6 Conv Layer
#classifierModel.add(Conv2D(256, (3,3), activation='relu',padding='same'))#(2,2,256)

#Layer7 MaxPooling
#classifierModel.add(MaxPooling2D(pool_size=(2,2))) #(1,1,256)

#Layer8 Conv Layer
#classifierModel.add(Conv2D(512, (3,3), activation='relu',padding='same'))

#Layer9 Conv Layer
#classifierModel.add(Conv2D(512, (3,3), activation='relu',padding='same'))

#Layer10 MaxPooling
#classifierModel.add(MaxPooling2D(pool_size=(2,2)))

#Layer11 Conv Layer
#classifierModel.add(Conv2D(512, (3,3), activation='relu',padding='same'))

#Layer12 Conv Layer
#classifierModel.add(Conv2D(512, (3,3), activation='relu',padding='same'))

#Layer13 MaxPooling
classifierModel.add(MaxPooling2D(pool_size=(2,2)))

#Flattening step
classifierModel.add(Flatten())

#Layer 14 FC
classifierModel.add(Dense(units = 4096, activation = 'relu'))

#Layer 15 FC
classifierModel.add(Dense(units = 4096, activation = 'relu'))

#Layer 16 FC
classifierModel.add(Dense(units = 512, activation = 'relu'))

#Layer 17 FC
classifierModel.add(Dense(units = 10, activation = 'softmax'))

# Compiling CNN
#optimizer = 'adam' :: use SGD for optimization
#loss='binary_cross_entropy' :: bin, for >2 categorical use categorical_crossentropy
classifierModel.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Fitting the CNN model  
history = classifierModel.fit(x_train, y_train, batch_size=32,epochs=3, verbose=1)


scoresList = []
blurList = [0, 1, 2, 3, 4, 5, 6]

## Looping over the entire test set to blur images
for blur in blurList:
    x_test_temp_blur = []
    for i in range(dataCountTest):
        X_testImage = Image.fromarray(x_test[i])
        new_image = X_testImage.resize((32,32), Image.HAMMING)
        # Now blur image
        image_blur = new_image.filter(ImageFilter.GaussianBlur(radius=blur))
        img_array_blur = image_blur.convert('L')
        x_test_temp_blur.append(np.array(img_array_blur))
    x_test_rotated = np.array(x_test_temp_blur)
    
    if K.image_data_format() == 'channels_first':
        x_test_rotated = x_test_rotated.reshape(x_test_rotated.shape[0], 1, img_row_count, img_column_count)
    else:
        x_test_rotated = x_test_rotated.reshape(x_test_rotated.shape[0], img_row_count, img_column_count,1)
        
    x_test_rotated = x_test_rotated.astype('float32')
    x_test_rotated /= 255

    score = classifierModel.evaluate(x_test_rotated, y_test, batch_size=32)
    scoresList.append(score)

scoresList = np.array(scoresList)
        
labels_blur = ['0', '1', '2', '3', '4', '5', '6']

# Plotting history for accuracy
plt.plot(blurList, scoresList[:,1])
plt.xticks(blurList, labels_blur)
plt.title('Accuracy Against Gaussian Blurring')
plt.ylabel('Accuracy on test')
plt.xlabel('Blur extent')
plt.grid()
plt.show()

# Plotting history for loss
plt.plot(blurList, scoresList[:,0])
plt.xticks(blurList, labels_blur)
plt.title('Loss Against Gaussian Blurring')
plt.ylabel('Loss on test')
plt.xlabel('Blur Extent')
plt.grid()
plt.show()