from model import CNN1, CNN2, Finetuning
import yaml
from data_loader import test_generation
from model import CNN1, CNN2, Finetuning, Finetuning_V2
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import argparse


""""
This script will take the value of "model_path" stored in config.yml and
evaluate the performance of the model on test dataset.

First a model instance will be initialized and the weights will be loaded,
then it will predict the data on testing set, calculate the confusion matrix, 
accuracy, F1 score, precision, recall and AUC.

we can pass the path of the model to be evaluated as command-line argument named 'path'. 
This is a string argument,
the value of this argument passed through command line when running the script would overwrite
the model_path stored in .yml file.


"""
if __name__ == "__main__":

    ## load the hyperparameters and other configurations
    with open("config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    eps = cfg["epochs"]
    bs = cfg["batch_size"]
    img_size = cfg["resized_dim"]

    # lr = cfg["learning_rate"]

    ## the path of the model to be evaluated is stored in model_path. We do not need to 
    ## further specify the model type as it weill be detected automatically from model_path
    model_path = cfg["model_path"]
    img_size = cfg["resized_dim"]
    shape = (img_size,img_size, 1)


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=model_path,
        type=str,
    )
    args = parser.parse_args()
    model_path = args.path
    print("The model to be evaluated is: %s"%model_path)

    if "CNN1" in model_path:
        model = CNN1(input_shape = shape) ##  construct the cnn model structure
    elif "CNN2" in model_path:
        model = CNN2(input_shape = shape) ##  construct the cnn model structure
    elif "VGG19" in model_path:
        model = Finetuning_V2("VGG19", img_size)

        # model = Finetuning("VGG19", shape) ## for the subclass implementation

    elif "inceptionv3" in model_path:
        model = Finetuning_V2("inceptionv3", img_size)
        # model = Finetuning("inceptionv3", shape) ## for the subclass implementation

    ## detect if masking was used for training this model.
    masked = False
    if "masked" in model_path:
        masked = True

    

    ## test data generator
    test_generator = test_generation(masked, bs)
    model.build(input_shape = (bs, img_size, img_size, 1))
    print("Model successfully built...")

    # model.compile(optimizer= Adam(learning_rate = lr),
    #           loss='categorical_crossentropy',
    #           metrics=['accuracy'])
    model.load_weights(model_path, skip_mismatch=False, by_name=False, options=None)
    print("Weights have been loaded, now predicting...")

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
    print("The confusion Matrix on testing set for model is: \n", confusionMatrix)

    accuracy = accuracy_score(trueClass, predictedClass)
    precision = precision_score(trueClass, predictedClass, average = "macro")
    recall = recall_score(trueClass, predictedClass, average = "macro")

    print("Precision,",precision, "\n")
    print("Recall,",recall, "\n")

    # f1 = f1_score(trueClass, predictedClass, average = "macro)

    print("\nAccuracy is %f"%accuracy)