
# Brain Tumor Classification

According to American Association of Neurological Surgeons, brain tumor is an abnormal mass of tissue in which ceels grow and multiply uncontrollably, seemlingly unchecked by the mechanisms that control normal cells. A major challenge for brain tumor detection arises from the variations in tumor location, size, shape, and the imaging quality of MRI. 
In this project, we aim to distinguish 3 different brain tumor types: glioma(malignant), pituitary and meningioma (both benign), and no tumor.  

## Table Of Content

- [Dataset](#dataset)

- [How to Use](#how-to-use)
    - [Getting Started](#getting-started)
    - [Preprocessing](#preprocessing)
    - [Model Training](#model-training)
    - [Evaluation](#evaluation)
    - [Rrediction](#prediction)
- [Image Processing](#page-setup)


- [Model Structures](#model-structure)
    - [CNN models](#cnn-model)
    - [Fine-tuning models](#cnn-model)    
- [Performance](#performance)


- [References](#References)


## Dataset


## How to use
### Getting Started

Install the required packages for python in your current environment or create a new environment for this program.

    ```
    pip install -r requirements.txt
    ```


### Preprocessing

Run the following script

    ```
    python preprosessing.py --masking 0
    ```

This command will take the data under data/Training and data/Testing as original data and apply some blurring, masking, cropping and resizing to the images and store the resulting images in the generated folders look like Processed_* or Unmasked_Processed_* depending if masking is used. 

--masking would specify is masking is used. If set to True, a kapur thresholding method will be applied on the images and mask the umimport parts.


### Model Training

    
     python train.py --model CNN1 --bs 64 --epoch 10 --aug False --c False --lr 0.00001
    
This piece of code would train a model of type "CNN1" (--model argument), with learning rate (lr) of 0.00001. --bs, --epoch with specify the batch size and epoch number for training. --aug indicates if data augmentation is used, and -c will instruct if the model would continue training from a previously trained model by loading weights from that model(computer will look for the model with the best performance from the folder).

After training, the weights of the model will be stored in the corresponding subfolder under models folder, and a history.json file will also be generated to record the model performance and loss throughout the training process.


### Evaluation

To evaluate a specific saved model(weights), simply use:

    
     python evaluate.py --path models/CNN1_aug/weights-10-0.65.h5
    

where --path specify the path of the saved weights to be evaluated. This code will evaluate this model on the test dataset, print out the confusion matrix and accuracy, precision and recall.

### Prediction

To classify new images in a specific folder (default folder is data/Predict, can be set in config.yml), use the code

    ```
     python predict.py
    ```

prediction results will be printed and all images will be stored under new names where their predicted tumor type will be included in their new file names.

