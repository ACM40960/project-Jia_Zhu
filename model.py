from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from PIL import Image
import keras
from tensorflow.keras.layers import Conv2D, Input, GlobalAveragePooling2D, ZeroPadding2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense,Dropout
from tensorflow.keras.models import Model, load_model, save_model

from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import backend

from tensorflow.keras import applications
from keras.models import Sequential

import numpy as np


class CNN1(tf.keras.Model):
  
  """ a CNN model with 5 convolutional layers, 2 max pooling layers followed by 4 dense layers with some regularization.
  
  To initialize an instance, input_shape is passed where the dimensions of the input should be specified.
  
  example: CNN1(input_shape = (256, 256, 1)), indicating images of size 256*256 with 1 channel are accepted
  
  """

  def __init__(self, input_shape):
    super().__init__()
    self.conv1 = Conv2D(16,kernel_size=(3,3),activation='relu',input_shape = input_shape)
    self.conv2 = Conv2D(32,kernel_size=(3,3),activation='relu')
    self.mp1 = MaxPool2D(2,2)
    self.conv3 = Conv2D(32,kernel_size=(3,3),activation='relu')  
    self.conv4 = Conv2D(32,kernel_size=(3,3),activation='relu')
    self.conv5 = Conv2D(64,kernel_size=(3,3),activation='relu')
    self.mp2 = MaxPool2D(4,4)
    self.flatten1 = Flatten()  
    self.dense1 = Dense(64,activation='relu')      
    self.dense2 = Dense(32,activation='relu') 
    self.dense3 = Dense(16,activation='relu')
    self.dropout1 = Dropout(rate=0.5)          
    self.dense4 = Dense(4,activation='softmax') 


  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.conv5(self.conv4(self.conv3(self.mp1(self.conv2(x)))))
    x = self.dense4(self.dropout1(self.dense3(self.dense2(self.dense1(self.flatten1(self.mp2(x)))))))
    return x

class CNN2(tf.keras.Model):
  
  """ another version of CNN model with pooling and batch normalization layers following each convolutional layer.

  To initialize an instance, input_shape is passed where the dimensions of the input should be specified.
  
  example: CNN2(input_shape = (256, 256, 1)), indicating images of size 256*256 with 1 channel are accepted
  """

  def __init__(self, input_shape):
    super().__init__()
    self.conv1 = Conv2D(64,kernel_size=(22,22), strides = 2,input_shape=input_shape)
    self.pooling1 = MaxPool2D(4,4)
    self.bn1 = BatchNormalization()

    self.conv2 = Conv2D(128,kernel_size=(11,11), strides = 2, padding = "same") 
    self.pooling2 = MaxPool2D(2,2)
    self.bn2 = BatchNormalization()

    self.conv3 = Conv2D(256,kernel_size=(7,7), strides = 2, padding = "same")
    self.pooling3 = MaxPool2D(2,2)
    self.bn3 = BatchNormalization()


    self.flatten1 = Flatten()  
    self.act1 = Activation("relu")
    self.dense1 = Dense(1024,activation='relu')  
    self.dropout1 = Dropout(rate=0.4)    
    self.dense2 = Dense(256,activation='relu') 
    self.dropout2 = Dropout(rate=0.4)          
    self.dense3 = Dense(4,activation='softmax') 

  def call(self, inputs):
    x = self.bn1(self.pooling1(self.conv1(inputs)))
    x = self.bn2(self.pooling2(self.conv2(x)))
    x = self.bn3(self.pooling3(self.conv3(x)))
    x = self.dropout1(self.dense1(self.act1(self.flatten1(x))))
    x = self.dropout2(self.dense2(x)) 
    return self.dense3(x)




class Finetuning(tf.keras.Model):

  """
  The keras subclass model version of Transfer Learning. In this implementation, to handle
  the dimension mismatch issue for training grayscale image with pre-trained models with three channels,
  we need to change the configuration of the first conv2d layer of the pre-trained model, then average over
  all three channels of the weight arrays in that layer, to accomodate single-channel images.

  The initialization of the class takes two inputs: the name of the pre-trained model and the input shape.

  All weights in the pre-trained model will be frozen and only the parameters in the following fully-connected 
  layers will be trainable.


  """

  def __init__(self, transfer, input_shape):
    super().__init__()

    self.img_size = input_shape[0]  

    if transfer == "VGG19":

        # Load the VGG19 model with pretrained weights and exclude the top (classification) layers
        self.base_model = applications.VGG19(weights='imagenet', 
                                        include_top=False, 
                                        input_shape=(self.img_size, self.img_size, 3))
        # Set all layers in VGG19 as non-trainable
        for layer in self.base_model.layers:
            layer.trainable = False

        ##  since VGG19 model accepts RGB images with 3 channels as inputs, we need to change the configuration
        ## on the first conv2d layer of VGG19 model to accept (512, 512, 1) shaped grayscale images


        cfg = self.base_model.get_config()
        cfg['layers'][0]['config']['batch_input_shape'] = (None, self.img_size, self.img_size, 1)
        self.base_model = Model.from_config(cfg)

        ## accordingly, we need to change the shape of weights array in the first layer of the VGG, 
        # by averaging over the channel dimension to produce filters with shape (3, 3, 1) as opposed to (3, 3, 3).

        new_weights = np.reshape(self.base_model.get_weights()[0].sum(axis=2)/3,(3,3,1,64))
        weights = self.base_model.get_weights()
        weights[0] = new_weights
        self.base_model.set_weights(weights)

    elif transfer == "inceptionv3":
        ## load the inceptionv3 model  with pretrained weights and exclude the top (classification) layers
        self.base_model = applications.InceptionV3(weights='imagenet', 
                                        include_top=False, 
                                        input_shape = (self.img_size, self.img_size, 3))
        
        # Set all layers in inceptionv3 as non-trainable
        for layer in self.base_model.layers:
            layer.trainable = False

        cfg = self.base_model.get_config()
        cfg['layers'][0]['config']['batch_input_shape'] = (None, self.img_size, self.img_size, 1)
        self.base_model = Model.from_config(cfg)

        ## accordingly, we need to change the shape of weights array in the first layer of the inceptionV3, 
        # by averaging over the channel dimension to produce filters with shape (3, 3, 1) as opposed to (3, 3, 3).
        ## the remainder of the weights remain the same

        new_weights = np.reshape(self.base_model.get_weights()[0].sum(axis=2)/3,(3,3,1,32))
        weights = self.base_model.get_weights()
        weights[0] = new_weights
        self.base_model.set_weights(weights)

    ## add a global average pooling layer to flatten out, then apply drop out and dense layers to fine-tune on our MRI datasets
    self.pooling1 = GlobalAveragePooling2D()   
    self.dense1 = Dense(512,activation='relu') 
    self.dense2 = Dense(4,activation='softmax')    
 
  def call(self, inputs):

    x = self.base_model(inputs)
    x = self.pooling1(x)
    x = self.dense1(x)
    output = self.dense2(x)

    return(output)
  
def Finetuning_V2(transfer, img_size):

  """"Sequential version of implementation for transfer learning
  Similiar to Finetuning(), In this implementation we keep all weights in the pre-trained models fixed,
  But this time around we add a convolutional layer before the inceptionv3 or VGG19 input layer to expand the 
  last dimension(number of channels) to 3 instead of 1.
  
  We replace the final classification layer of the pre-trained models with some fully-connected layers to 
  fine tune on the brain tumor datasets"""


  if transfer == "inceptionv3":

    base_model = applications.InceptionV3(weights='imagenet', 
                                  include_top=False, 
                                  input_shape=(img_size, img_size,3))

  elif transfer == "VGG19":
    base_model = applications.VGG19(weights='imagenet', 
                                  include_top=False, 
                                  input_shape=(img_size, img_size,3))
    

  ## make the parameters in the inceptionv3 model untrainable
  for layer in base_model.layers:
    layer.trainable = False

  model = Sequential()
  ## add a conv layer before the pre-trained model to make the number of input channels in agreement
  model.add(Conv2D(3,kernel_size=(1,1),input_shape=(img_size, img_size, 1)))
  model.add(base_model)

  ## add a global average pooling layer to flatten out, then apply drop out and dense layers to fine-tune on our MRI datasets
  model.add(GlobalAveragePooling2D())
  model.add(Dropout(0.5))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4, activation='softmax'))
  return (model)
  