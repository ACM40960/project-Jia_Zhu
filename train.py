import numpy as np
import pandas as pd
from data_loader import data_generation

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

from keras import applications
from keras.models import Sequential
import pickle
import os
from model import CNN1, CNN2, Finetuning
import yaml

# checkpoint file and early stopping




def model_train(batch_size, nb_epochs, model, model_name, lr):

    """train the """
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
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

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
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    return model


def best_model_finder(mname):
    folderpath = "models/%s/"%mname
    max_performance = 0
    best_model = ""
    for f in os.listdir(folderpath):
        if "history" in f: continue
        perf = float(f.split("-")[2][:4])
        if perf > max_performance:
            max_performance = perf
            best_model = f
    print("The current best performing model: %s is loaded"%best_model)
    return(folderpath+best_model)



if __name__ == "__main__":

    ## load the hyperparameters and other configurations

    with open("config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    eps = cfg["epochs"]
    augmentation = cfg["augmentation"]
    model_type = cfg["model_type"]
    if augmentation == True:
        mname = model_type + "_aug"
    else:
        mname = model_type + "_aug"

    continue_training = cfg["continue_training"]
    masked = cfg["masked"]
    bs = cfg["batch_size"]
    processed = cfg["processed"]
    seed = cfg["seed"]
    lr = cfg["learning_rate"]
    (train_generator,validation_generator) = data_generation(augmentation=augmentation, processed = processed, masked = masked, bs = bs, seed = seed)
    if "CNN" in model_type:
        if model_type == "CNN1":

            model = CNN1() ##  construct the cnn model structure
        elif model_type == "CNN2":
            model = CNN2()
        if continue_training == True:
            model.build(input_shape = (bs, 512, 512, 1))
            best_model_path = best_model_finder(mname)
            model.load_weights(best_model_path, skip_mismatch=False, by_name=False, options=None)
        model = model_train(batch_size = bs, nb_epochs = eps, model = model, model_name = mname, lr = lr)


    else: 
        model = Finetuning(model_type)


        if continue_training == True:
            model.build(input_shape = (bs, 512, 512, 1))
            best_model_path = best_model_finder(mname)
            model.load_weights(best_model_path, skip_mismatch=False, by_name=False, options=None)
        model = model_train(batch_size = bs, nb_epochs = eps, model = model, model_name = mname, lr = lr)




