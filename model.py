from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from PIL import Image
import keras
from tensorflow.keras.layers import Conv2D, Input, GlobalAveragePooling2D, ZeroPadding2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense,Dropout
from tensorflow.keras.models import Model, load_model, save_model

from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import backend

from keras import applications
from keras.models import Sequential




class CNN1(tf.keras.Model):

  def __init__(self, input_shape =(512,512, 1)):
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

  def __init__(self, input_shape =(512,512, 1)):
    super().__init__()
    self.conv1 = Conv2D(64,kernel_size=(22,22), strides = 2,input_shape=input_shape)
    self.pooling1 = MaxPool2D(4,4)
    # self.bn1 = BatchNormalization()

    self.conv2 = Conv2D(128,kernel_size=(11,11), strides = 2, padding = "same") 
    self.pooling2 = MaxPool2D(2,2)
    # self.bn2 = BatchNormalization()

    self.conv3 = Conv2D(256,kernel_size=(7,7), strides = 2, padding = "same")
    self.pooling3 = MaxPool2D(2,2)
    # self.bn3 = BatchNormalization()


    self.flatten1 = Flatten()  
    self.act1 = Activation("relu")
    self.dense1 = Dense(1024,activation='relu')  
    self.dropout1 = Dropout(rate=0.4)    
    self.dense2 = Dense(256,activation='relu') 
    self.dropout2 = Dropout(rate=0.4)          
    self.dense3 = Dense(4,activation='softmax') 


  def call(self, inputs):
    x = self.pooling1(self.conv1(inputs))
    x = self.pooling2(self.conv2(x))
    x = self.pooling3(self.conv3(x))
    x = self.dropout1(self.dense1(self.act1(self.flatten1(x))))
    x = self.dropout2(self.dense2(x)) 
                      
    return self.dense3(x)


class Finetuning(tf.keras.Model):

  def __init__(self, transfer, input_shape =(512,512, 1)):
    super().__init__()

    if transfer == "VGG16":

        self.base_model = applications.VGG16(weights='imagenet', 
                                        include_top=False, 
                                        input_shape=(512,512, 3))

    elif transfer == "inceptionv3":

        self.base_model = applications.InceptionV3(weights='imagenet', 
                                        include_top=False, 
                                        input_shape = (512,512, 3))
        

    ## make the parameters in the inceptionv3 model untrainable
    for layer in self.base_model.layers:
        layer.trainable = False

    ## add a conv layer before the inceptionv3 to make the number of input channels in agreement
    self.conv1 = Conv2D(3,kernel_size=(1,1),input_shape=input_shape)

    ## add a global average pooling layer to flatten out, then apply drop out and dense layers to fine-tune on our MRI datasets
    self.pooling1 = GlobalAveragePooling2D()
    self.dropout1 = Dropout(rate=0.5)    
    self.dense2 = Dense(256,activation='relu') 
    self.dropout2 = Dropout(rate=0.5)   
    self.dense3 = Dense(4,activation='softmax')    
 
  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.base_model(x)
    x = self.dropout1(self.pooling1(x))
    x = self.dropout2(self.dense2(x))

    return(self.dense3(x))
  
