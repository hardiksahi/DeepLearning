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
from keras import regularizers
from keras.datasets import mnist


## Using 6000 training examples and 1000 test examples instead of complete training and test set
dataCountTrain = 6000
dataCountTest = 1000
class_num = 10

## Loading MNIST data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

xTrainTemp = []

### DATA PREPROCESSING STARTS

## Converting the data into 32*32 from 28*28 and retreiving 6000 examples from training and 1000 from test set
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

# Add dimensions to the dataset so it is readable by the CNN 
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
classifierModel = Sequential()


#Adding different layers in the CNN

#Layer 1 Conv layer
classifierModel.add(Conv2D(64, (3,3), input_shape = input_shape, activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.001)))#(30,30,64)

#Layer2 MaxPooling
classifierModel.add(MaxPooling2D(pool_size=(2,2)))#(15,15,64)

#Layer3 Conv Layer
classifierModel.add(Conv2D(128, (3,3), activation='relu',padding='same'))#(13,13,128)

#Layer4 MaxPooling
classifierModel.add(MaxPooling2D(pool_size=(2,2))) #(6,6,128)

#Layer5 Conv Layer
classifierModel.add(Conv2D(256, (3,3), activation='relu',padding='same'))#(4,4,256)

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

print(classifierModel.output_shape)


# Fits the model 
history = classifierModel.fit(x_train, y_train, batch_size=32,epochs=3, verbose=1)

scoresList = []
degreesList = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

## Looping over all the entire test set to rotate all images by specific degrees

x_test_temp_rotate = []
for degree in degreesList:
    x_test_temp_rotate = []
    for i in range(dataCountTest):
        X_testImage = Image.fromarray(x_test[i])
        new_image = X_testImage.resize((32,32), Image.HAMMING)
        # Now rotate image
        rotate_image = new_image.rotate(degree)
        img_array_rot = rotate_image.convert('L')
        x_test_temp_rotate.append(np.array(img_array_rot))
    x_test_rotated = np.array(x_test_temp_rotate)
    
    if K.image_data_format() == 'channels_first':
        x_test_rotated = x_test_rotated.reshape(x_test_rotated.shape[0], 1, img_row_count, img_column_count)
    else:
        x_test_rotated = x_test_rotated.reshape(x_test_rotated.shape[0], img_row_count, img_column_count,1)
        
    x_test_rotated = x_test_rotated.astype('float32')
    x_test_rotated /= 255

    score = classifierModel.evaluate(x_test_rotated, y_test, batch_size=32)
    scoresList.append(score)

scoresList = np.array(scoresList)
        
    

labels_rotation = ['-45', '-40', '-35', '-30', '-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40', '45']

# Plotting history for accuracy
plt.plot(degreesList, scoresList[:,1])
plt.xticks(degreesList, labels_rotation)
plt.title('TestSet: Accuracy vs Rotation')
plt.ylabel('Accuracy on test')
plt.xlabel('Degrees of rotation')
plt.grid()
plt.show()

# Plotting history for loss
plt.plot(degreesList, scoresList[:,0])
plt.xticks(degreesList, labels_rotation)
plt.title('TestSet: Loss vs Rotation')
plt.ylabel('Loss on test')
plt.xlabel('Degrees of rotation')
plt.grid()
plt.show()