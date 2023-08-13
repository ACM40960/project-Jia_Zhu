import os
from keras.preprocessing.image import ImageDataGenerator
import yaml

def data_generation(augmentation,  processed = True, masked = False, bs = 32, seed = 123):
    """
    input parameters 
    

    1. augmentation: bool, if data augmentation is implemented during data generation
    2. processed: If th processed or original imgages are used for training. 
    3. masked: bool, if the images were masked using Kapur thresholding. When True, the data generator would read the
    images from masked folder instead of the unmasked ones
    4. bs: batch size of the data generator
    5. seed: seed state for the data generator
    
    
    """
    with open("config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    img_size = cfg["resized_dim"] ## get the image size 
    classes = cfg['classes'] ## get the class names from config
    current_directory = os.getcwd()
    
    ### depending on if masking and other preprocessing techniques are used, we change the directory for training/validation/testing 
    ## set accordingly.
    
    masked_ind = ""
    if masked == False:
        masked_ind = "Unmasked_"
    if processed == True:
        processed_path = os.path.join(current_directory, "data", masked_ind + "Processed_Training")
        color = "grayscale"
    else:
        processed_path = os.path.join(current_directory, "data", masked_ind + "Training") 
        color = 'rgb'
    print("The folder for training data is: %s"%processed_path,"\n", "Color channels: %s"%color)

    ## Data are rescaled, if data augmentation is implemented, we create the tensor image data with real time data-augmentation.
    ## some augmentation strategies used are: zooming, rotation, brightness change, flipping...

    if augmentation == True:
        train_datagen = ImageDataGenerator(rescale=1./255,
            rotation_range = 90, shear_range = 0.4,zoom_range = 0, samplewise_center=True, brightness_range=[0.1, 0.7],
            vertical_flip = True, horizontal_flip = True, 
            validation_split=0.15) # set validation split
    else:
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15) # set validation split

    ## data flow are generated from directory. each subfolder contains only images of the class of the folder name.

    train_generator = train_datagen.flow_from_directory(
        processed_path,
        target_size=(img_size, img_size),
        color_mode=color,
        classes=classes,
        class_mode = "categorical",
        batch_size=bs,
        shuffle=True,
        seed=seed,
        save_to_dir=None,
        save_prefix='',
        save_format='jpg',
        follow_links=False,
        interpolation='nearest',
        keep_aspect_ratio=False,
        subset = "training"
    ) # set as training set
    validation_generator = train_datagen.flow_from_directory(
        processed_path,
        target_size=(img_size, img_size),
        color_mode=color,
        classes=classes,
        class_mode = "categorical",
        batch_size=bs,
        shuffle=False,
        seed=seed,
        save_to_dir=None,
        save_prefix='',
        save_format='jpg',
        follow_links=False,
        interpolation='nearest',
        keep_aspect_ratio=False,
        subset = "validation"
    
    ) # set as validation data
    return (train_generator,validation_generator)


""""
For test data generation. Shuffle is set to  false for testing data.
"""

def test_generation(masked = False, bs = 32):
    with open("config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    img_size = cfg["resized_dim"]
    classes = cfg['classes']  
    current_directory = os.getcwd()
    
    masked_ind = ""
    if masked == False:
        masked_ind = "Unmasked_"
    processed_path = os.path.join(current_directory, "data", masked_ind + "Processed_Testing")
    print("the test data used are stored in path: %s"%processed_path)
    classes = os.listdir(processed_path)
 ## data generator with data augmentation
    test_datagen = ImageDataGenerator(rescale=1./255) # set validation split

    test_generator = test_datagen.flow_from_directory(
        processed_path,
        target_size=(img_size, img_size),
        color_mode="grayscale",
        classes=classes,
        class_mode = "categorical",
        batch_size = bs,
        shuffle=False,
        save_to_dir = None,
        save_prefix = '',
        save_format = 'jpg',
        follow_links=False,
        interpolation='nearest',
        keep_aspect_ratio=False
    ) # set as training set
    return (test_generator)