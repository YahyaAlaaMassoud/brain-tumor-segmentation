import os
import pandas as pd
import random
import nibabel as nib
import matplotlib.pyplot as plt
import shutil

## _Reading Cases from File_
# Define the folder path
folder_path = r'G:\Academic\Master\Term Fall 2023-Carleton University\Medical Image Processing\Final Project\Brat Dataset\BRaTS 2021 Task 1 Dataset\BraTS2021_Training_Data'

# List the contents of the folder
folder_contents = os.listdir(folder_path)

## _Selecting Cases Randomly in two Sets_
cases_number = len(folder_contents)

# Defining the number of random cases for each set
first_set_random_number = 12
second_set_random_number = 15

# Defining the output array that contain the random cases for each set
set1_random_array = [x for x in range(first_set_random_number)]
set2_random_array = [x for x in range(second_set_random_number)]

# Choosing Random Cases and Putting them in the first set array
for counter in range(first_set_random_number):
    
    set1_random_array[counter] = random.choice(folder_contents) 

# Choosing Random Cases and Putting them in the second set array
for counter in range(second_set_random_number):
    
    set2_random_array[counter] = random.choice(folder_contents) 



## _Creat a Folder Called "Random_Cases" for Setes of Random Cases
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

## _Creating "Cases_x" Folder_
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

## _Creating "Set1" and "Set2" folder_
Set1_folder_path = Cases_folder_path + '\\' + 'Set1'
os.makedirs(Set1_folder_path)

Set2_folder_path = Cases_folder_path + '\\' + 'Set2'
os.makedirs(Set2_folder_path)



## _Copying images to "Set1" and "Set2" folders
for set1_case_counter in range(first_set_random_number):
    # Creating Folders of Set1 Cases
    set1_case_path = Set1_folder_path + '\\' + str(set1_case_counter+1)
    os.makedirs(set1_case_path)
    # Addressing the randomlly selected cases for set1 in the dataset
    case_in_dataset_path = folder_path + '\\' + set1_random_array[set1_case_counter]

    ## Copying the files in Dataset to Set1
    # Get a list of all files in the source folder
    files = os.listdir(case_in_dataset_path)

    # Copy each file to the destination folder
    for file in files:
        source_file_path = os.path.join(case_in_dataset_path, file)
        destination_file_path = os.path.join(set1_case_path, file)
        shutil.copy(source_file_path, destination_file_path)
    


    #* Extracting the images from .nib file into the Set1 and in each 1, 2, 3, ..., 12or15 folder and in "Images" folder
    # Creat a "Images" folder in each case of set1/2
    set1_case_images_path = set1_case_path + '\\' + 'Images'
    os.makedirs(set1_case_images_path)
    
    # Reading the metadatas in the Set1\x (case) folder
    files = os.listdir(set1_case_path)

    # A random image between 50~100 slides
    Metadata_random_slide = random.randint(50, 100)
    # Opening each "Metadata" file and extract a random image between 50~100 slides and save in ".png"
    for file in files:
        if '.nii' in file:
            # Reading Data in Metadata
            Metadata_file_path = set1_case_path + '\\' + file
            Metadata_file_data = nib.load(Metadata_file_path).get_fdata()

            for slice_index in range(Metadata_file_data.shape[-1]):
                # Check so see if the Slice index is equal to the random slide
                if slice_index == Metadata_random_slide:
                    # Extracting the "random image" to silce_data
                    slice_data = Metadata_file_data[:, :, slice_index]
                    
                    # Check the type of the file (flair, seg, t1, t1ce, t2)
                    if 'flair' in file:
                        set1_case_image_path = set1_case_images_path + '\\' + 'flair'
                    elif 'seg' in file:
                        set1_case_image_path = set1_case_images_path + '\\' + 'seg'
                    elif 't1ce' in file:
                        set1_case_image_path = set1_case_images_path + '\\' + 't1ce'
                    elif 't1' in file:
                        set1_case_image_path = set1_case_images_path + '\\' + 't1'
                    elif 't2' in file:
                        set1_case_image_path = set1_case_images_path + '\\' + 't2'

                    # Defining the file to save a ".png"
                    set1_case_image_path = set1_case_image_path + '.png'
                    plt.imsave(set1_case_image_path, slice_data, cmap='gray')


    
for set2_case_counter in range(second_set_random_number):
    # Creating Folders of Set2 Cases
    set2_case_path = Set2_folder_path + '\\' + str(set2_case_counter+1)
    os.makedirs(set2_case_path)
    # Addressing the randomlly selected cases for set2 in the dataset
    case_in_dataset_path = folder_path + '\\' + set2_random_array[set2_case_counter]

    ## Copying the files in Dataset to Set2
    # Get a list of all files in the source folder
    files = os.listdir(case_in_dataset_path)

    # Copy each file to the destination folder
    for file in files:
        source_file_path = os.path.join(case_in_dataset_path, file)
        destination_file_path = os.path.join(set2_case_path, file)
        shutil.copy(source_file_path, destination_file_path)



    #* Extracting the images from .nib file into the Set1 and in each 1, 2, 3, ..., 12or15 folder and in "Images" folder
    # Creat a "Images" folder in each case of set1/2
    set2_case_images_path = set2_case_path + '\\' + 'Images'
    os.makedirs(set2_case_images_path)
    
    # Reading the metadatas in the Set2\x (case) folder
    files = os.listdir(set2_case_path)

    # A random image between 50~100 slides
    Metadata_random_slide = random.randint(50, 100)
    # Opening each "Metadata" file and extract a random image between 50~100 slides and save in ".png"
    for file in files:
        if '.nii' in file:
            # Reading Data in Metadata
            Metadata_file_path = set2_case_path + '\\' + file
            Metadata_file_data = nib.load(Metadata_file_path).get_fdata()


            for slice_index in range(Metadata_file_data.shape[-1]):
                # Check so see if the Slice index is equal to the random slide
                if slice_index == Metadata_random_slide:
                    # Extracting the "random image" to silce_data
                    slice_data = Metadata_file_data[:, :, slice_index]
                    
                    # Check the type of the file (flair, seg, t1, t1ce, t2)
                    if 'flair' in file:
                        set2_case_image_path = set2_case_images_path + '\\' + 'flair'
                    elif 'seg' in file:
                        set2_case_image_path = set2_case_images_path + '\\' + 'seg'
                        print(slice_data)
                    elif 't1ce' in file:
                        set2_case_image_path = set2_case_images_path + '\\' + 't1ce'
                    elif 't1' in file:
                        set2_case_image_path = set2_case_images_path + '\\' + 't1'
                    elif 't2' in file:
                        set2_case_image_path = set2_case_images_path + '\\' + 't2'

                    # Defining the file to save a ".png"
                    set2_case_image_path = set2_case_image_path + '.png'
                    plt.imsave(set2_case_image_path, slice_data, cmap='gray')
