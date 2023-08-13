from model import CNN1, CNN2, Finetuning
import yaml
from data_loader import test_generation
from model import CNN1, CNN2, Finetuning
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

if __name__ == "__main__":

    ## load the hyperparameters and other configurations
    with open("config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    eps = cfg["epochs"]
    masked = cfg["masked"]
    bs = cfg["batch_size"]
    img_size = cfg["resized_dim"]
    augmentation = cfg["augmentation"]
    # lr = cfg["learning_rate"]

    ## the path of the model to be evaluated is stored in model_path. We do not need to 
    ## further specify the model type as it weill be detected automatically from model_path
    model_path = cfg["model_path"]
    img_size = cfg["resized_dim"]
    if "CNN1" in model_path:
        model = CNN1(input_shape =(img_size,img_size, 1)) ##  construct the cnn model structure
    elif "CNN2" in model_path:
        model = CNN2(input_shape =(img_size,img_size, 1)) ##  construct the cnn model structure
    elif "VGG19" in model_path:
        model = Finetuning("VGG19", input_shape =(img_size,img_size, 1))
    elif "inceptionv3" in model_path:
        model = Finetuning("inceptionv3", input_shape =(img_size,img_size, 1))

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
    accuracy = accuracy_score(trueClass, predictedClass)
    print("The confusion Matrix on testing set for model is: \n", confusionMatrix)

    print("\nAccuracy is %f"%accuracy)