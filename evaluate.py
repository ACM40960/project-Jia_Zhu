from model import CNN1, CNN2, Finetuning
import yaml
from data_loader import test_generation
from model import CNN1, CNN2, Finetuning
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score

if __name__ == "__main__":

    ## load the hyperparameters and other configurations

    with open("config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    eps = cfg["epochs"]
    masked = cfg["masked"]
    bs = cfg["batch_size"]

    model_type = cfg["model_type"]
    augmentation = cfg["augmentation"]
    lr = cfg["learning_rate"]


    model_path = cfg["model_path"]

    test_generator = test_generation(masked, bs)
    if model_type == "CNN1":
        model = CNN1() ##  construct the cnn model structure
    elif model_type == "CNN2":
        model = CNN2() ##  construct the cnn model structure
    else: 
        model = Finetuning(model_type)

    model.build(input_shape = (bs, 512, 512, 1))
    model.compile(optimizer= Adam(learning_rate = lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.load_weights(model_path, skip_mismatch=False, by_name=False, options=None)

    predictions = model.predict(test_generator,
                                    steps=test_generator.samples/bs,
                                    workers = 0,
                                    verbose=1)
    
    predict = model.evaluate(test_generator,verbose = 1)

    # # Evaluate predictions
    predictedClass = np.argmax(predictions, axis=1)
    trueClass = test_generator.classes[test_generator.index_array]
    classLabels = list(test_generator.class_indices.keys())

    # Create confusion matrix
    confusionMatrix = (confusion_matrix(
        y_true=trueClass, # ground truth (correct) target values
        y_pred=predictedClass)) # estimated targets as returned by a classifier
    accuracy = accuracy_score(trueClass, predictedClass)
    print(confusionMatrix)
    print("accuracy is %f"%accuracy)