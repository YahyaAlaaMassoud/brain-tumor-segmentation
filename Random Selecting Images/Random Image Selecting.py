import os
import pandas as pd
import random
import nibabel as nib
import matplotlib.pyplot as plt
import shutil
import numpy as np

#* _Reading Cases from File_
# Define the folder path
folder_path = r'G:\Academic\Master\Term Fall 2023-Carleton University\Medical Image Processing\Final Project\Brat Dataset\BRaTS 2021 Task 1 Dataset\BraTS2021_Training_Data'
# List the contents of the folder
folder_contents = os.listdir(folder_path)

#* _Selecting Cases Randomly in two Sets_
cases_number = len(folder_contents)
# Defining the number of random cases for each set
random_number = [12, 15]
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
    # Creat case folders "x" from 1 to random_number
    for random_number_counter in range(random_number[set_folder_counter]):
        #Each case source path
        source_case_folder_path = folder_path + '\\' + random_array[set_folder_counter][random_number_counter]
        source_case_folder_contents = os.listdir(source_case_folder_path)

        #Finding the segmentation metadata name in the source file
        segmentation_metadata_name = ''
        for folder_content in source_case_folder_contents:
            if 'seg' in folder_content:
                segmentation_metadata_name = folder_content

        #Creating destination each case folder
        destination_each_case_folder_path = Set_folder_path + '\\' + str(random_number_counter+1)
        os.makedirs(destination_each_case_folder_path)

        #! Writing Data 
        #* 1) Finding a random slide each its segmentation is not empty
        random_slide_number = 0
        Flag_segmentation_is_not_empty = False
        while Flag_segmentation_is_not_empty is False:
            # A random image between 50~100 slides
            random_slide_number = random.randint(0, 154)
            # Segmentation metadata path 
            segmentation_metadata_path = source_case_folder_path + '\\' + segmentation_metadata_name
            # Getting segmentation metadata data
            segmentation_metadata = nib.load(segmentation_metadata_path).get_fdata()
            # The data of slide of random number only
            segmentation_slice_data = segmentation_metadata[:, :, random_slide_number]
            # Checking the sumation of the image to see if the segmentation is empty or not
            threshold = 0.5
#           # Convert to binary image
            binary_slice_data = (segmentation_slice_data > threshold).astype(np.uint8)

            sumation_variable = 0
            for i in range(240):
                for j in range(240):
                    sumation_variable =  sumation_variable + binary_slice_data[i][j]
            #! Defining the tumor region
            if sumation_variable > 500:
                print(sumation_variable, (random_number_counter+1))
                Flag_segmentation_is_not_empty = True
                # Defining the file to save a ".png"
                # Saving the Black and White Image of the 'seg'
                segmentation_image_path = destination_each_case_folder_path + '\\' + random_array[set_folder_counter][random_number_counter] + '_' + str((random_slide_number+1)) + '_' + 'seg.png'
                plt.imsave(segmentation_image_path, binary_slice_data, cmap='gray')
                # Defining the file to save a ".png"
                segmentation_image_path = destination_each_case_folder_path + '\\' + random_array[set_folder_counter][random_number_counter] + '_' + str((random_slide_number+1)) + '_' + 'seg_original.png'
                plt.imsave(segmentation_image_path, segmentation_slice_data, cmap='gray')
                # Copying the 'seg' Metadata
                case_metadata_path = destination_each_case_folder_path + '\\' + segmentation_metadata_name
                shutil.copyfile(segmentation_metadata_path, case_metadata_path)



    
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
                slice_data = metadata[:, :, random_slide_number]
                # Copying the file to the pass
                image_path = destination_each_case_folder_path + '\\' + random_array[set_folder_counter][random_number_counter] + '_' + str((random_slide_number+1)) + '_' + file_name_type
                plt.imsave(image_path, slice_data, cmap='gray')
                # Copying the other Metadata
                case_metadata_path = destination_each_case_folder_path + '\\' + metadata_name
                shutil.copyfile(metadata_path, case_metadata_path)

