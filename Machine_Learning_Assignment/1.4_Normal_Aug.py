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
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist


## Using 6000 training examples and 1000 test examples instead of complete training and test set
dataCountTrain = 6000
dataCountTest = 1000
class_num = 10

## Loading MNIST data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
xTrainTemp = []
xTestTemp = []

### DATA PREPROCESSING STARTS

## Converting the data into 32*32 from 28*28 and retreiving 6000 examples from training and 1000 from test set
for i in range(dataCountTrain):
	X_trainImage = Image.fromarray(x_train[i])
	new_image = X_trainImage.resize((32,32), Image.HAMMING)
	img_array = new_image.convert('L')
	img_array = np.array(img_array)
	xTrainTemp.append(img_array)
    
for i in range(dataCountTest):
	X_trainImage = Image.fromarray(x_test[i])
	new_image = X_trainImage.resize((32,32), Image.HAMMING)
	img_array = new_image.convert('L')
	img_array = np.array(img_array)
	xTestTemp.append(img_array)

x_train = np.array(xTrainTemp)
x_test = np.array(xTestTemp)
y_train = y_train[0:dataCountTrain]
y_test = y_test[0:dataCountTest]


img_row_count = img_column_count = 32
# Add dimensions to the dataset so it is readable by the CNN 
if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_row_count, img_column_count)
	x_test = x_test.reshape(x_test.shape[0], 1, img_row_count, img_column_count)
	input_shape = (1, img_row_count, img_column_count)
else:
	x_train = x_train.reshape(x_train.shape[0], img_row_count, img_column_count, 1)
	x_test = x_test.reshape(x_test.shape[0], img_row_count, img_column_count, 1)
	input_shape = (img_row_count, img_column_count, 1)
    
#Convert class vectors to binary class matrices  and convert them to float32 and normalize them
y_train = keras.utils.to_categorical(y_train, class_num)
y_test = keras.utils.to_categorical(y_test, class_num)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

train_datagen = ImageDataGenerator(horizontal_flip=True)
test_datagen = ImageDataGenerator()


### DATA PREPROCESSING ENDS

### CREATING CNN LAYERS 
classifierModel = Sequential()


#Adding different layers in the CNN

#Layer 1 Conv layer
classifierModel.add(Conv2D(64, (3,3), input_shape = input_shape, activation='relu',padding='same'))#(30,30,64)

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
test_datagen.fit(x_test)

train_datagen.fit(x_train)

history = classifierModel.fit_generator(train_datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch = len(x_train), 
					epochs = 2, 
					validation_data = test_datagen.flow(x_test, y_test, batch_size = 32),
					validation_steps = len(x_test))


# Plotting history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy (Train and Test) vs Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()

# Plotting history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss (Train and Test) vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()