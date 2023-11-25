import os
import pandas as pd
import random
import nibabel as nib
import matplotlib.pyplot as plt
import shutil
import numpy as np
import cv2
from skimage import color


def image_two_show(image1, title1, image2, title2):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image1.copy(), cmap='gray')
    axs[0].set_title(title1)
    axs[1].imshow(image2.copy(), cmap='gray')
    axs[1].set_title(title2)
    plt.show()


def image_mean_func(image):
    average_color = np.mean(image, axis=(0, 1))
    return average_color


def remove_image_function(segmented_binary_image, flair_image):
    seg_image = segmented_binary_image
    flair_image_copy1 = flair_image.copy()
    flair_image_copy2 = flair_image.copy()
    image_height, image_width = flair_image_copy1.copy().shape
    # removing the tumor from the flair image
    result_flair_picture = flair_image_copy1.copy() 
    for x in range(image_width):
        for y in range(image_height):
            if seg_image[x][y] == 1:
                result_flair_picture[x][y] = 0 
    for x in range(image_width):
        for y in range(image_height):
            if result_flair_picture[x][y] == 0:
                flair_image_copy2[x][y] = 0 
    flair_image_copy2_8bit = cv2.convertScaleAbs(flair_image_copy2.copy())
    return flair_image_copy2_8bit


def comparison_function_otsu(segmented_binary_image, flair_image):
    flair_image_8bit = remove_image_function(segmented_binary_image, flair_image)
    otsu_threshold_value = cv2.threshold(flair_image_8bit.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    return otsu_threshold_value


def comparison_function_mean(segmented_binary_image, flair_image):
    flair_image_8bit = remove_image_function(segmented_binary_image, flair_image)
    mean_value = image_mean_func(flair_image_8bit)
    return mean_value





#* init variables
slide_number_variable = 0



#* _Reading Cases from File_
# Define the folder path
folder_path = r'G:\Academic\Master\Term Fall 2023-Carleton University\Medical Image Processing\Final Project\Brat Dataset\BRaTS 2021 Task 1 Dataset\BraTS2021_Training_Data'
# List the contents of the folder
folder_contents = os.listdir(folder_path)

#* _Selecting Cases Randomly in two Sets_
cases_number = len(folder_contents)
# Defining the number of random cases for each set
random_number = [25, 30]
# Defining the output array that contain the random cases for each set

random_array = [[] for x in range(len(random_number))]

# Choosing Random Cases and Putting them in the first set array
for array_index in range(len(random_number)):
    for random_counter in range(random_number[array_index]):
        random_case_name = ''
        # Check to see if the File is not in the array
        Flag_unique_random_case_name = False
        while Flag_unique_random_case_name is not True:
            # Getting a random case
            random_case_name = random.choice(folder_contents)  
            #check the first time

            if len(random_array[array_index]) == 0:
                Flag_unique_random_case_name = True
            else:
                if random_case_name not in random_array[array_index]:
                    Flag_unique_random_case_name = True

        random_array[array_index].append(random_case_name)


#! Shuffle the Cases
Random_ste1 = folder_contents
np.random.shuffle(Random_ste1)
Random_ste2 = folder_contents
np.random.shuffle(Random_ste2)

Random_array_v2 = np.vstack((Random_ste1, Random_ste2))


#* _Creat a Folder Called "Random_Cases" for Setes of Random Cases
# _One Folder Coming Back to Creat the "Random_Cases" Folder_
# Split the path into its components
folder_components = folder_path.split(os.path.sep)
# Remove the last component (folder)
Random_Cases_folder_path = os.path.sep.join(folder_components[:-1])
# Adding the Folder name to the end of the adderess
Random_Cases_folder_path = Random_Cases_folder_path + '\\' + 'Random_Cases'
# Checking to see if the "Random_Cases" folder exist
if not os.path.exists(Random_Cases_folder_path):
    os.makedirs(Random_Cases_folder_path)
# Extracting the Content inside the "Random_Cases" folder
Random_Cases_folder_contents = os.listdir(Random_Cases_folder_path)


#* _Creating "Cases_x" Folder_
Cases_folder_path = Random_Cases_folder_path = Random_Cases_folder_path + '\\' + 'Cases_'
# Check to see if there is any "Cases_x" folder. if not, creat "Cases_1" folder. if yes, creat a folder with uper x 
if not Random_Cases_folder_contents:
    Cases_folder_path = Cases_folder_path + '1'
    os.makedirs(Cases_folder_path)
else:
    # Initialize an empty list to store the last numbers
    last_numbers = []
    # Iterate through the file names
    for file_name in Random_Cases_folder_contents:
        # Split the file name by underscores to separate 'Cases' and the number
        parts = file_name.split('_')
        # Extract the last part, which is the number, and convert it to an integer
        last_part = parts[-1]
        
        try:
            last_number = int(last_part)
            last_numbers.append(last_number)
        except ValueError:
            # Handle the case where the last part is not a valid integer
            pass
    # Finding the last x of "Cases_x" folder
    max_cases_number = max(last_numbers)
    # Creating "Cases_x" folder with uper x
    Cases_folder_path = Cases_folder_path + str(max_cases_number+1)
    os.makedirs(Cases_folder_path)


#* _Creating "Setx" folders and Writing data_
for set_folder_counter in range(len(random_number)):
    # Creating "Setx" folder
    Set_folder_path = Cases_folder_path + '\\' + 'Set' + str(set_folder_counter+1)
    os.makedirs(Set_folder_path)

    # Case that are used and should not be in the next seg file finding
    case_unsed_number = 0

    # Creat case folders "x" from 1 to random_number
    for random_number_counter in range(random_number[set_folder_counter]):

        #Creating destination each case folder
        destination_each_case_folder_path = Set_folder_path + '\\' + str(random_number_counter+1)
        os.makedirs(destination_each_case_folder_path)

        Flag_segmentation_and_flair_ok = False
        while Flag_segmentation_and_flair_ok is False: 
            
            #Each case source path
            source_case_folder_path = folder_path + '\\' + Random_array_v2[set_folder_counter][case_unsed_number]
            source_case_folder_contents = os.listdir(source_case_folder_path)
            
            #Finding the segmentation metadata name in the source file
            segmentation_metadata_path = ''
            segmentation_metadata_name = ''
            for folder_content in source_case_folder_contents:
                if 'seg' in folder_content:
                    segmentation_metadata_name = folder_content
                    # Segmentation metadata path 
                    segmentation_metadata_path = source_case_folder_path + '\\' + folder_content
                    # Getting segmentation metadata data
                    segmentation_metadata = nib.load(segmentation_metadata_path).get_fdata()

            # Finding the flair image path
            flair_image_path = ''
            for folder_content in source_case_folder_contents:
                if 'flair' in folder_content:
                    flair_image_path = source_case_folder_path + '\\' + folder_content
                    # extract the flair image slice from the metadata
                    flair_image_metadata = nib.load(flair_image_path).get_fdata()

            for slice_number in range(60, 100):
                
                # The data of slide of random number only
                segmentation_slice_data = segmentation_metadata[:, :, slice_number]
                flair_image_slice = flair_image_metadata[:, :, slice_number]
                # Checking the sumation of the image to see if the segmentation is empty or not
                threshold = 0.5
                # Convert to binary image
                binary_slice_data = (segmentation_slice_data > threshold).astype(np.uint8)
                # the area of the segmentation
                sumation_variable = 0
                for i in range(240):
                    for j in range(240):
                        sumation_variable =  sumation_variable + binary_slice_data[i][j]

                seg_image_mean = image_mean_func(segmentation_slice_data)
                #! Defining the tumor region and flair
                if (sumation_variable > 500) and (seg_image_mean >= 0.082):
                    # Compare the segmentation and flair image to see if it is a good image 
                    comparision_result = comparison_function_otsu(binary_slice_data, flair_image_slice)
                    flair_image_mean = comparison_function_mean(binary_slice_data, flair_image_slice)
                    if flair_image_mean <= 42:
                        image_two_show(segmentation_slice_data, 'seg', flair_image_slice, 'flair')
                        print('floar image mean: ', flair_image_mean)
                        print('seg image mean: ', seg_image_mean)
                        print(sumation_variable, (random_number_counter+1))
                        print('slice number: ', slide_number_variable)
                        
                        #* Flages
                        Flag_segmentation_and_flair_ok = True
                        slide_number_variable = slice_number
                        # Defining the file to save a ".png"
                        # Saving the Black and White Image of the 'seg'
                        segmentation_image_path = destination_each_case_folder_path + '\\' + random_array[set_folder_counter][random_number_counter] + '_' + str((slide_number_variable+1)) + '_' + 'seg.png'
                        plt.imsave(segmentation_image_path, binary_slice_data, cmap='gray')
                        # Defining the file to save a ".png"
                        segmentation_image_path = destination_each_case_folder_path + '\\' + random_array[set_folder_counter][random_number_counter] + '_' + str((slide_number_variable+1)) + '_' + 'seg_original.png'
                        plt.imsave(segmentation_image_path, segmentation_slice_data, cmap='gray')
                        # Copying the 'seg' Metadata
                        case_metadata_path = destination_each_case_folder_path + '\\' + segmentation_metadata_name
                        shutil.copyfile(segmentation_metadata_path, case_metadata_path)
                        break
            
            case_unsed_number = case_unsed_number + 1
            print('case number: ', case_unsed_number)



        #* Writing {'flair', 't1ce', 't1', 't2'}
        segmentation_metadata_name = ''
        file_name_type = ''
        # Check the folder file, which is going to be written, is not junk or 'seg' file
        for folder_content in source_case_folder_contents:
            Flag_writting_is_ok = False
            if 'flair' in folder_content:
                Flag_writting_is_ok = True
                file_name_type = 'flair.png'
            elif 't1ce' in folder_content: # First should check 't1ce' to avoid wrong segmentation with 't1'
                Flag_writting_is_ok = True
                file_name_type = 't1ce.png'
            elif 't1' in folder_content:
                Flag_writting_is_ok = True
                file_name_type = 't1.png'
            elif 't2' in folder_content:
                Flag_writting_is_ok = True
                file_name_type = 't2.png'
            # The file is one of the {'flair', 't1ce', 't1', 't2'}
            if Flag_writting_is_ok is True:
                metadata_name = folder_content
                metadata_path = source_case_folder_path + '\\' + metadata_name
                # Getting segmentation metadata data
                metadata = nib.load(metadata_path).get_fdata()
                # The data of slide of random number only
                slice_data = metadata[:, :, slide_number_variable]
                # Copying the file to the pass
                image_path = destination_each_case_folder_path + '\\' + random_array[set_folder_counter][random_number_counter] + '_' + str((slide_number_variable+1)) + '_' + file_name_type
                plt.imsave(image_path, slice_data, cmap='gray')
                # Copying the other Metadata
                case_metadata_path = destination_each_case_folder_path + '\\' + metadata_name
                shutil.copyfile(metadata_path, case_metadata_path)

