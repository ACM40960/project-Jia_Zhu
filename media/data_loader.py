import os
from keras.preprocessing.image import ImageDataGenerator
import yaml
from matplotlib import pyplot as plt

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
        interpolation='nearest'
    ) # set as training set
    return (test_generator)



# The following 2 functions are just to visualize the data augmentation
def data_augmentation(refresh=True, num=5):
    """
    refresh: whether to replace current augmented data and generate new ones
    num: number of augmented data per image
    """

    training_path = "data\\visualization"
    ## destination parent folder for augmented data
    augmented_path = "data\\aug_visualization"
    current_directory = os.getcwd()
    original_path = os.path.join(current_directory, training_path)
    augmented_path = os.path.join(current_directory, augmented_path)

    ## augmented data generator
    image_generator = ImageDataGenerator(rotation_range=90, shear_range=0.4, zoom_range=0, samplewise_center=True,
                                         vertical_flip=True, horizontal_flip=True, samplewise_std_normalization=True)

    for subf in os.listdir(original_path):
        new_dir = os.path.join(augmented_path, subf)
        create_dir(new_dir, empty=refresh)

        for f in os.listdir(os.path.join(original_path, subf)):
            image_path = os.path.join(original_path, subf, f)
            img = load_img(image_path)
            x = np.array(img)
            x = x.reshape((1,) + x.shape)  # reshape to (1, height, width, channels)

            base_name = os.path.splitext(f)[0]  # extract name without extension
            ext = os.path.splitext(f)[1]  # extract file extension

            i = 0
            img.save(os.path.join(new_dir, f))  # save original image

            for batch in image_generator.flow(x, batch_size=1):
                # Construct the new filename
                new_filename = f"{base_name}_{i + 1}{ext}"
                new_image_path = os.path.join(new_dir, new_filename)

                # Save the image
                batch_image = tf.keras.preprocessing.image.array_to_img(batch[0])
                batch_image.save(new_image_path)

                i += 1
                if i >= num:
                    break

def visualize_augmentation(original_path, augmented_path, sample_name="Tr-gl_0010", num_augmentations=5):
    """
    Visualize the effect of data augmentation for a specific image.

    Parameters:
    - original_path: Path to the directory containing original images.
    - augmented_path: Path to the directory containing augmented images.
    - sample_name: Name of the sample image (without extension) to visualize.
    - num_augmentations: Number of augmented images to display for visualization.
    """
    original_img_name = sample_name + ".jpg"
    original_image_path = os.path.join(original_path, original_img_name)

    # Load and display original image
    original_img = plt.imread(original_image_path)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, num_augmentations + 1, 1)
    plt.imshow(original_img)
    plt.title("Original")

    # Display the augmented images
    for i in range(1, num_augmentations + 1):
        augmented_img_name = f"{sample_name}_{i}.jpg"
        augmented_image_path = os.path.join(augmented_path, augmented_img_name)

        augmented_img = plt.imread(augmented_image_path)

        plt.subplot(1, num_augmentations + 1, i + 1)
        plt.imshow(augmented_img)
        plt.title(f"Aug_{i}")

    plt.tight_layout()
    plt.show()