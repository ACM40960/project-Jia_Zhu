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

from keras import applications
from keras.models import Sequential
import pickle
import os
import json
from model import CNN1, CNN2, Finetuning
import yaml
from utils import best_model_finder
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

# checkpoint file and early stopping




def model_train(batch_size, nb_epochs, model, model_name, lr, continue_training):

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
        validation_data = test_generator, 
        validation_steps = test_generator.samples // batch_size,
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
    if augmentation == True:
        mname = model_type + "_aug"
    else:
        mname = model_type + "_aug"

    img_size = cfg["resized_dim"]

    continue_training = cfg["continue_training"]
    masked = cfg["masked"]
    bs = cfg["batch_size"]
    processed = cfg["processed"]
    seed = cfg["seed"]
    lr = cfg["learning_rate"]
    (train_generator,validation_generator) = data_generation(augmentation=augmentation, processed = processed, masked = masked, bs = bs, seed = seed)
    test_generator = test_generation(masked, bs)
    if "CNN" in model_type:
        if model_type == "CNN1":

            model = CNN1(input_shape =(img_size,img_size, 1)) ##  construct the cnn model structure
        elif model_type == "CNN2":
            model = CNN2(input_shape =(img_size,img_size, 1))
        if continue_training == True:
            model.build(input_shape = (bs, img_size, img_size, 1))
            best_model_path = best_model_finder(mname)
            model.load_weights(best_model_path, skip_mismatch=False, by_name=False, options=None)
            print("████████████model weights successfully loaded, Now training...")
            
        model = model_train(batch_size = bs, nb_epochs = eps, model = model, model_name = mname, lr = lr,continue_training = continue_training)


        ######FOR TESTING PURPOSE ONLY!!!!!!!!!!
        predictions = model.predict(validation_generator,
                                        steps=validation_generator.samples/bs,
                                        workers = 0,
                                        verbose=1)
        

        # # Evaluate predictions, first get the predicted labels and the true labels
        predictedClass = np.argmax(predictions, axis=1)
        trueClass = validation_generator.classes[validation_generator.index_array]
        classLabels = list(validation_generator.class_indices.keys())

        # Create confusion matrix
        confusionMatrix = confusion_matrix(
            y_true=trueClass, # ground truth (correct) target values
            y_pred=predictedClass) # estimated targets as returned by a classifier
        confusionMatrix = pd.DataFrame(confusionMatrix, columns = classLabels, index = classLabels)
        accuracy = accuracy_score(trueClass, predictedClass)
        print("The confusion Matrix on testing set for model is: \n", confusionMatrix)

        print("\nAccuracy is %f"%accuracy)

        test_generator = test_generation(masked, bs)
        predictions = model.predict(test_generator,
                                        steps=test_generator.samples/bs,
                                        workers = 0,
                                        verbose=1)
        

        # # Evaluate predictions, first get the predicted labels and the true labels
        predictedClass = np.argmax(predictions, axis=1)
        trueClass = test_generator.classes[test_generator.index_array]
        classLabels = list(test_generator.class_indices.keys())

        # Create confusion matrix
        confusionMatrix = confusion_matrix(
            y_true=trueClass, # ground truth (correct) target values
            y_pred=predictedClass) # estimated targets as returned by a classifier
        confusionMatrix = pd.DataFrame(confusionMatrix, columns = classLabels, index = classLabels)
        accuracy = accuracy_score(trueClass, predictedClass)
        print("The confusion Matrix on testing set for model is: \n", confusionMatrix)

        print("\nAccuracy is %f"%accuracy)
        model.save("models/CNN1_aug/model.tf", save_format="tf")
        #########################END

    else: 
        model = Finetuning(model_type, input_shape =(img_size, img_size, 1))

        if continue_training == True:
            
            model.build(input_shape = (bs, img_size, img_size, 1))
            best_model_path = best_model_finder(mname)
            model.load_weights(best_model_path, skip_mismatch=False, by_name=False, options=None)
            print("████████████model weights successfully loaded, Now training...")
        model = model_train(batch_size = bs, nb_epochs = eps, model = model, model_name = mname, lr = lr, continue_training= continue_training)
    






