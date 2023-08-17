import numpy as np
import pandas as pd
from data_loader import data_generation, test_generation

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input

from PIL import Image
import keras
from tensorflow.keras.layers import Conv2D, Input, GlobalAveragePooling2D, ZeroPadding2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense,Dropout
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import backend

import pickle
import os
import json
from model import CNN1, CNN2, Finetuning, Finetuning_V2
import yaml
from utils import best_model_finder
import numpy as np
import argparse

"""
Script for training a model.

Potentially 6 arguments can be passed through command line when calling train.py. All default values
are in config.yml.

--model: type of model to be trained. CNN1, CNN2, VGG19 or inceptionv3
--epoch: number of epochs
--bs: batch size
--lr: learning rate for adam optimizer
--aug: if data augmentation is used
--c: if contunue training mode is on. If set to true, the script will look for the best performing model
    in the corresponding folder, load the weights and continue training that model.

"""




def model_train(batch_size, nb_epochs, model, model_name, lr, continue_training):

    """Fit a model on training data generator and evaluate it on validation generator.
    Monitoring the accuracy on validation set after each epoch and employ early stopping.
    Model weights are saved upon completion of each epoch.
    
    After completing the training, a json file would be saved in the same folder as the .h5 files,
    to record the traing and validation loss/accuracy.
    
    If continue training mode is on, the history.json won't overwrite the original file but will add
    new data onto the existing file."""
    folderpath = "models/%s/"%model_name
    CHECK_FOLDER = os.path.isdir(folderpath)
    if not CHECK_FOLDER:
        os.makedirs(folderpath)
    filepath = folderpath + "weights-{epoch:02d}-{val_accuracy:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_weights_only = True, save_best_only=True, mode='max')

    ## set the callback function with early stopping and monitering the validation accuracy
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3) 
    callbacks_list = [checkpoint, early_stopping]

    ## use adam optimizer with customized learning rate
    optimizer = Adam(learning_rate = lr)

    ## compile the model.
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    ## fit the model on train_generator and use validation generator as validation data
    history = model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // batch_size,
        epochs = nb_epochs,
        callbacks = callbacks_list)
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 


    # save the history to json:  
    hist_json_file = folderpath + 'history.json' 

    ## if in cotinue training mode and history.json file already exists, we want to append new
    ## evaluation data into the existing file instead of completely overwriting it

    if (continue_training == True) and (os.path.exists(hist_json_file) == True):
        df = pd.read_json(hist_json_file)
        hist_df = pd.concat([df, hist_df])
        hist_df = hist_df.reset_index(drop=True)

        # Set the index to start from 1
        hist_df.index = range(0, len(hist_df))

    ## save the json file.
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    return model





if __name__ == "__main__":

    ## load the hyperparameters and other configurations
    with open("config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    eps = cfg["epochs"]
    augmentation = cfg["augmentation"]
    model_type = cfg["model_type"]
    img_size = cfg["resized_dim"]
    continue_training = cfg["continue_training"]
    masked = cfg["masked"]
    bs = cfg["batch_size"]
    processed = cfg["processed"]
    seed = cfg["seed"]
    lr = cfg["learning_rate"]

    
    ##arguments to be passed through command line that can overwrite the .yml config

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default= model_type, type=str)
    parser.add_argument("--epoch", default= eps, type=int)
    parser.add_argument("--lr", default= lr, type=float)
    parser.add_argument("--bs", default= bs, type=int)
    parser.add_argument("--aug", default= augmentation, type=bool)
    parser.add_argument("--c", default= continue_training, type=str)
    parser.add_argument("--masking", default=masked, type=bool)

    args = parser.parse_args()
    model_type = args.model
    eps = args.epoch
    lr = args.lr
    bs = args.bs
    augmentation = args.aug
    continue_training = args.c
    masked = args.masking

    """
        saved directory for models will change according to model type and specifications"""
    if augmentation == True:
        mname = model_type + "_aug"
    else:
        mname = model_type + "_aug"
    if masked == True:
        mname = "masked_" + mname


    print("Now training a %s model: , with batch size of %s, maximum %s epochs."%(model_type, bs, eps))



    ## generate the train and validation generator
    (train_generator,validation_generator) = data_generation(augmentation=augmentation, processed = processed, masked = masked, bs = bs, seed = seed)

    ## if we are runnin the cnn architecture, we simply call a subclass model isntance
    if "CNN" in model_type:
        if model_type == "CNN1":

            model = CNN1(input_shape =(img_size,img_size, 1)) 
        elif model_type == "CNN2":
            model = CNN2(input_shape =(img_size,img_size, 1))

        ## When continue training mode is on, after building the model, it will try to  search for all
        ## model files in the folder and find the one that reaches the highest accuracy on validation set,
        ## then this .h5 saved weights will be loaded into the model instance.
        if continue_training == True:
            model.build(input_shape = (bs, img_size, img_size, 1))
            best_model_path = best_model_finder(mname)  ## find the model with the highest performance in the folder
            model.load_weights(best_model_path, skip_mismatch=False, by_name=False, options=None)  ## load the saved weights
            print("████████████model weights successfully loaded, Now training...")
        
        ## training
        model = model_train(batch_size = bs, nb_epochs = eps, model = model, model_name = mname, lr = lr,continue_training = continue_training)




    ## for transfer learning(Fine-tuning)
    else: 
        ## call Finetuning_V2 function to initialize a sequential model of transfer learning
        model = Finetuning_V2(model_type, img_size)
        # model = Finetuning(model_type, input_shape =(img_size, img_size, 1))

        if continue_training == True:
            model.build(input_shape = (bs, img_size, img_size, 1))
            best_model_path = best_model_finder(mname)
            model.load_weights(best_model_path, skip_mismatch=False, by_name=False, options=None)
            print("████████████model weights successfully loaded, Now training...")
        model = model_train(batch_size = bs, nb_epochs = eps, model = model, model_name = mname, lr = lr, continue_training= continue_training)
    






