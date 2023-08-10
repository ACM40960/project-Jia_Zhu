from model import CNN1, CNN2, Finetuning
import yaml
from data_loader import test_generation
from model import CNN1, CNN2, Finetuning
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

if __name__ == "__main__":

    ## load the hyperparameters and other configurations

    with open("config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    eps = cfg["epochs"]
    masked = cfg["masked"]
    bs = cfg["batch_size"]
    model_path = cfg["model_path"]
    transfer = cfg["transfer_learning"]
    model_path = cfg["model_path"]
    test_generator = test_generation(masked, bs)
    if transfer == "None":
        model = CNN2() ##  construct the cnn model structure
        model.build(input_shape = (bs, 512, 512, 1))

    else: 
        model = Finetuning(transfer)
        model.build(input_shape = (bs, 512, 512, 3))
    model.load_weights(model_path, skip_mismatch=False, by_name=False, options=None)

    predictions = model.predict_generator(generator = test_generator,
                                    steps=test_generator.samples/bs,
                                    workers = 0,
                                    verbose=1)

    # Evaluate predictions
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