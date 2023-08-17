import numpy as np
import tensorflow as tf
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from PIL import Image
import imutils
from matplotlib import pyplot as plt
from kapur import kapur_threshold
from utils import create_dir
import yaml


## save the augmented data and the original ones in new folders
def data_augmentation(refresh=True, num=5):
    """
    refresh: whether to replace current augmented data and generate new ones
    num: number of augmented data per image
    """

    training_path = "data\\visualization"
    ## destination parent folder for augmented data
    augmented_path = "data\\augmentation_v"
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

def blur_and_crop(image, blur, cropping= False, kernel = 3, masking = True, plot=False):
    """
    preprocessing:
    1. convert to grayscale and blur the image using median or gaussian filter
    2. (optional)apply kapur thresholding to create a mask, mask the blurred image
    3. crop the image to contain only the brain image, leaving the blank around surrounding the brain out.
    """
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur == "median":
        blurred = cv2.medianBlur(gray, kernel)
    elif blur == "gaussian":
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise

    if masking == True:
   ## creating mask with kapur thresholding
        threshold = kapur_threshold(blurred)
        binr = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]
        masked_image = cv2.bitwise_and(blurred, blurred, mask=binr)
    else:
        masked_image = blurred
    if cropping == True:

        thresh = cv2.threshold(masked_image, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)


        # Find the extreme points for cropping
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # crop new image out of the original image using the four extreme points (left, right, top, bottom)
        cropped_image = masked_image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]      
    else:
        cropped_image = masked_image

    if plot:
        plt.figure(figsize=(10, 10))
        plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
        plt.subplot(132), plt.imshow(masked_image, cmap='gray'), plt.title('Masked Image')
        plt.subplot(133), plt.imshow(cropped_image, cmap='gray'), plt.title('Cropped and Masked Image')
        
        plt.show()
    

    return cropped_image


def preprocessing(training_path, blur_method, masking = False, crop = False):
    """
    preprocess the images in training_path parent folder
    1. create a destination folder for preprocessed images
    2. blur, mask and (crop) the iamges, masking is optional.
    3. store the processed images in new folder
    
    parameter: 
    training_path: the folder name for the original images to be processed
    masking: if masking is applied in the processing
    """
    
    current_directory = os.getcwd()
    ## destination parent folder for processed data
    if masking == True:
        processed_path = "\\Processed_".join(training_path.split("\\"))
    else:
        processed_path = "\\Unmasked_Processed_".join(training_path.split("\\"))        
    
    processed_path = os.path.join(current_directory, processed_path)
    original_path = os.path.join(current_directory, training_path)
    for subf in os.listdir(original_path):
        new_dir = os.path.join(processed_path, subf)
        create_dir(new_dir, empty = True)
        for f in os.listdir(os.path.join(original_path, subf)):
            image_path = os.path.join(original_path, subf, f)
            img = cv2.imread(image_path)


            ## apply image transformation
            new_img = blur_and_crop(img, blur = blur_method, cropping = crop, kernel = 3, masking = masking, plot=False)
            img = cv2.resize(img, (new_size, new_size)) ## resize the image as they are not of the same sizes
            image = Image.fromarray(new_img)
            image.save(os.path.join(new_dir, f))

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


def visualize_preprocessing_steps(image, blur="median", kernel=5, masking=True, cropping=True, plot=True):
    """
    Show each step in preprocessing.
    """
    images = []  # List to store images at each step
    titles = []  # Titles for each step

    # Original Image
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    titles.append('Original Image')

    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur == "median":
        blurred = cv2.medianBlur(gray, kernel)
    elif blur == "gaussian":
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    images.append(blurred)
    titles.append('Blurred Image')

    if masking == True:
        # Kapur Thresholding
        threshold = kapur_threshold(blurred)
        binr = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]
        images.append(binr)
        titles.append('Kapur Thresholding')

        # Mask the Blurred Image
        masked_image = cv2.bitwise_and(blurred, blurred, mask=binr)
        images.append(masked_image)
        titles.append('Masked Image')

    # Cropping
    if cropping == True:
        thresh = cv2.threshold(masked_image if masking else blurred, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        cropped_image = (masked_image if masking else blurred)[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        images.append(cropped_image)
        titles.append('Cropped Image')

    if plot:
        plt.figure(figsize=(20, 10))
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, len(images), i + 1)
            cmap = 'gray' if i > 0 else None
            plt.imshow(img, cmap=cmap)
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    return cropped_image


def visualize_preprocessing(original_directory, processed_directory, masking=True, cropping=True):
    """
    Visualize each step in the preprocessing of a random image.

    Parameters:
    - original_directory: The path to the original images.
    - processed_directory: The path to the processed images.
    - masking: If the images have been masked during preprocessing.
    - cropping: If the images have been cropped during preprocessing.
    """

    # Choose a random sub-folder (class) from the original directory
    random_class = np.random.choice(os.listdir(original_directory))
    original_class_path = os.path.join(original_directory, random_class)

    # Choose a random image from the selected class
    random_image_name = np.random.choice(os.listdir(original_class_path))
    original_image_path = os.path.join(original_class_path, random_image_name)

    # Read the chosen image
    img = cv2.imread(original_image_path)

    visualize_preprocessing_steps(img, blur="median", masking=masking, cropping=cropping) 

if __name__ == "__main__":
    with open("config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    blur_method = cfg["blur"]
    masked = cfg["masked"]
    new_size = cfg["resized_dim"]
    ## preprocess the training data,
    preprocessing(training_path = "data\\Training", blur_method = blur_method, masking = masked)
    #preprocessing(training_path = "data\\Training", blur_method = blur_method, masking = masked)
    ## preprocessing the testing data, masked and unmasked
    #preprocessing(training_path = "data\\Testing", blur_method = blur_method, masking = masked)
    preprocessing(training_path = "data\\Testing", blur_method = blur_method, masking = masked)