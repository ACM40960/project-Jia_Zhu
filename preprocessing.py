import numpy as np
import keras
import tensorflow as tf
import cv2
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from PIL import Image
import imutils
from matplotlib import pyplot as plt
from kapur import kapur_threshold

def create_dir(newdir, empty = True):
    """
    create new folder if the target folder doesnt exist
    """
    CHECK_FOLDER = os.path.isdir(newdir)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(newdir)
        print("created folder : ", newdir)

    else:
        if empty == True:
            ## whether to remove all contents in the current augmented data folder and generate new ones
            shutil.rmtree(newdir)
            print("current augmented data removed")
            os.makedirs(newdir)
            
        print(newdir, "folder already exists.")
    
## save the augmented data and the original ones in new folders
def data_augmentation(refresh = True, num = 5):
    """refresh: whether to replace current augmented data and generate new ones
    num: number of augmented data per image"""

    
    training_path = "data\\Training"
    ## destination parent folder for augmented data
    augmented_path = "data\\augmentation_training"
    current_directory = os.getcwd()
    original_path = os.path.join(current_directory,training_path)
    augmented_path = os.path.join(current_directory,augmented_path)

    ## augmented data generator
    image_generator = ImageDataGenerator(rotation_range = 90, shear_range = 0.4,zoom_range = 0, samplewise_center=True, 
                                         vertical_flip = True, horizontal_flip = True, samplewise_std_normalization= True)
    for subf in  os.listdir(original_path):
        
        new_dir = os.path.join(augmented_path, subf)
        create_dir(new_dir, empty = refresh)
        for f in os.listdir(os.path.join(original_path, subf)):
            image_path = os.path.join(original_path, subf,f)
            img = load_img(image_path)  
            i = 1
            img.save(os.path.join(augmented_path, subf, f))
            for batch in image_generator.flow(x, batch_size = 1, 
                          save_to_dir = new_dir,  
                          save_prefix = f.split(".")[0], save_format ='jpg'):
                i += 1
                if i > num: 
                    break

def blur_and_crop(image, blur = "median", cropping= False, kernel = 5, masking = True, plot=False):
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


def preprocessing(training_path, masking = False, crop = False):
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
            new_img = blur_and_crop(img, blur = "median", cropping = crop, kernel = 5, masking = masking, plot=False)
            image = Image.fromarray(new_img)
            image.save(os.path.join(new_dir, f))
        

if __name__ == "__main__":
    ## preprocess the training data,
    preprocessing(training_path = "data\\Training", masking = False)
    preprocessing(training_path = "data\\Training", masking = True)
    ## preprocessing the testing data, masked and unmasked
    preprocessing(training_path = "data\\Testing", masking = True)
    preprocessing(training_path = "data\\Testing", masking = False)