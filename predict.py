from model import CNN1, CNN2, Finetuning
import yaml
from data_loader import test_generation
from model import CNN1, CNN2, Finetuning, Finetuning_V2
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from preprocessing import blur_and_crop
import os
from PIL import Image
import cv2



"""
This script would load the model weights stored in model_path,
and use it to predict all images stored in predict_folder.

Each file would also be saved with the predicted label on it, for instance from "1.jpg" to "1_glioma.jpg".

"""
if __name__ == "__main__":

    ## load the hyperparameters and other configurations
    with open("config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    eps = cfg["epochs"]
    masked = cfg["masked"]
    bs = cfg["batch_size"]
    classes = cfg['classes']

    augmentation = cfg["augmentation"]
    # lr = cfg["learning_rate"]

    ## the path of the model to be evaluated is stored in model_path. We do not need to 
    ## further specify the model type as it weill be detected automatically from model_path
    model_path = cfg["model_path"]
    predict_path = cfg["predict_folder"]
    blur = cfg["blur"]

    img_size = cfg["resized_dim"]
    shape = (img_size,img_size, 1)
    if "CNN1" in model_path:
        model = CNN1(input_shape = shape) ##  construct the cnn model structure
    elif "CNN2" in model_path:
        model = CNN2(input_shape = shape) ##  construct the cnn model structure
    elif "VGG19" in model_path:
        model = Finetuning_V2("VGG19", img_size)
    elif "inceptionv3" in model_path:
        model = Finetuning_V2("inceptionv3", img_size)

    ## test data generator
    model.build(input_shape = (bs, img_size, img_size, 1))

    print("Model successfully built...")

    # model.compile(optimizer= Adam(learning_rate = lr),
    #           loss='categorical_crossentropy',
    #           metrics=['accuracy'])
    model.load_weights(model_path, skip_mismatch=False, by_name=False, options=None)
    print("Weights have been loaded, now predicting...")

    for f in os.listdir(predict_path):

        ## converting the image into array, preprocess and rescale it
        img_path = os.path.join(predict_path, f)
        img_o = cv2.imread(img_path)
        img = blur_and_crop(img_o, blur, cropping= False, kernel = 3, masking = masked, plot=False)
        img = cv2.resize(img, (img_size, img_size))
        img = img/255

        ## expand the image dimensions to (1, dim, dim, 1)
        img_exp = np.expand_dims(img, axis=-1)
        img_exp = np.expand_dims(img_exp, axis=0 )

        ## predict the image, get the class type
        res = model.predict(img_exp, verbose = False)
        class_num = np.argmax(res)
        class_name = classes[class_num] ## get the name of the classification
        prob = np.max(res)

        model.predict(img_exp)
        print("@@@@@@@Image: %s, Predicted Class:%s with probability %s\n"%(f, class_name, prob))

        ### save the image to a different name with predicted label in the file name.

        f_new = "_".join([f.split(".")[0], class_name]) + ".jpg"
        # os.rename(img_path, os.path.join(predict_path, f))
        image = Image.fromarray(img_o)
        image.save(os.path.join(predict_path, f_new))
