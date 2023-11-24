import os
import json
import numpy as np
import matplotlib.pyplot as plt
import requests
import cv2
from scipy.ndimage import convolve, median_filter, gaussian_filter
from scipy.fft import fft2, fftshift
from skimage.util import random_noise
from sklearn.metrics import mean_squared_error
from PIL import Image
from io import BytesIO
from pprint import pprint
from eval_metrics.seg_eval_metrics import IoU, dice_similarity


def region_growing_v1(image, seed=None, threshold=10):
    """
    Perform region growing algorithm on a grayscale image.

    Parameters:
    image (numpy.ndarray): Grayscale image.
    seed (tuple): Starting point (x, y) for region growing. If None, the seed will be chosen automatically.
    threshold (int): Threshold for determining pixel similarity.

    Returns:
    numpy.ndarray: Segmented image.
    """
    # Image dimensions
    rows, cols = image.shape

    # If seed is not specified, select it based on intensity
    if seed is None:
        seed = np.unravel_index(np.argmax(image, axis=None), image.shape)

    # Initialize segmented output image
    segmented = np.zeros_like(image, dtype=int)

    # List of pixels that need to be examined
    pixel_list = [seed]

    # Region growing algorithm
    while len(pixel_list) > 0:
        x, y = pixel_list.pop(0)
        if not segmented[x, y]:
            segmented[x, y] = 255
            # Check the 8-neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        if abs(int(image[nx, ny]) - int(image[x, y])) < threshold:
                            pixel_list.append((nx, ny))

    return segmented, seed


def region_growing_v2(image, seed=None, threshold_factor=0.1):
    """
    Perform region growing algorithm on a grayscale image using histogram and CDF for dynamic thresholding.

    Parameters:
    image (numpy.ndarray): Grayscale image.
    seed (tuple): Starting point (x, y) for region growing. If None, the seed will be chosen automatically.
    threshold_factor (float): Factor to determine dynamic threshold based on intensity range.

    Returns:
    numpy.ndarray: Segmented image.
    """
    # Image dimensions
    rows, cols = image.shape

    # If seed is not specified, select it based on intensity
    if seed is None:
        seed = np.unravel_index(np.argmax(image, axis=None), image.shape)

    # Get the intensity of the seed point
    seed_intensity = image[seed]

    # Determine the dynamic threshold based on the intensity at the seed point
    intensity_range = np.max(image) - np.min(image)
    dynamic_threshold = intensity_range * threshold_factor

    # Initialize segmented output image
    segmented = np.zeros_like(image, dtype=int)

    # List of pixels that need to be examined, starting with the seed point
    pixel_list = [seed]

    # Region growing algorithm
    while pixel_list:
        x, y = pixel_list.pop(0)
        if not segmented[x, y]:
            segmented[x, y] = 255
            # Check the 8-neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    # Skip the current pixel
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        if abs(int(image[nx, ny]) - seed_intensity) < dynamic_threshold:
                            pixel_list.append((nx, ny))

    return segmented, seed


def region_growing_v3(image, seed=None, initial_threshold_factor=0.3, adjustment_factor=0.05):
    """
    Perform region growing algorithm on a grayscale image with dynamic threshold adjustment.

    Parameters:
    image (numpy.ndarray): Grayscale image.
    seed (tuple): Starting point (x, y) for region growing. If None, the seed will be chosen automatically.
    initial_threshold_factor (float): Initial factor to determine dynamic threshold based on intensity range.
    adjustment_factor (float): Factor to adjust the threshold during the growing process.

    Returns:
    numpy.ndarray: Segmented image.
    """
    rows, cols = image.shape
    if seed is None:
        seed = np.unravel_index(np.argmax(image, axis=None), image.shape)

    seed_intensity = image[seed]
    intensity_range = np.max(image) - np.min(image)
    dynamic_threshold = intensity_range * initial_threshold_factor

    segmented = np.zeros_like(image, dtype=bool)
    pixel_list = [seed]

    while pixel_list:
        x, y = pixel_list.pop(0)
        if not segmented[x, y]:
            segmented[x, y] = True
            region_mean = np.mean(image[segmented])
            region_std = np.std(image[segmented])

            # Adjust the dynamic threshold based on the mean and standard deviation of the segmented region
            dynamic_threshold = max(region_std * adjustment_factor, dynamic_threshold)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        if abs(int(image[nx, ny]) - region_mean) < dynamic_threshold:
                            pixel_list.append((nx, ny))

    segmented = (segmented * 255).astype(np.uint8)

    return segmented, seed


def draw_intersection_of_binary_images_v3(image1, image2):
    """
    Draw the intersection of two binary images, where intersection is red, non-intersection is white,
    and background is black.

    Parameters:
    image1, image2 (numpy.ndarray): Two binary images of the same size.

    Returns:
    numpy.ndarray: Image highlighting the intersection in red, non-intersection in white, and background in black.
    """
    if image1.shape != image2.shape:
        raise ValueError("The input images must have the same size")

    # Create an image to display the result
    result_image = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)

    # Intersection (red)
    intersection = (image1 == 255) & (image2 == 255)
    result_image[intersection] = [255, 0, 0]

    # Non-intersecting parts of the images (white)
    non_intersection = ((image1 == 255) | (image2 == 255)) & ~intersection
    result_image[non_intersection] = [255, 255, 255]

    # Background remains black (as initialized)

    return result_image



DATA_DIR = 'Images/Raw Images/'

for dataset_name in os.listdir(DATA_DIR):
    
    RESULTS_DIR = 'Images/results/region_growing/' + dataset_name + '/'
    
    dataset = {}
    
    dataset_path = DATA_DIR + dataset_name + '/'
    
    print(dataset_path)

    for data_folder in os.listdir(dataset_path):
        print(data_folder)
        
        flair_image = None
        for file in os.listdir(dataset_path + data_folder):
            if file.endswith('.png') and 'flair' in file:
                flair_image = cv2.imread(dataset_path + data_folder + '/' + file, cv2.IMREAD_GRAYSCALE)
                break
            
        ground_truth = None
        for file in os.listdir(dataset_path + data_folder):
            if file.endswith('.png') and 'seg' in file and 'original' not in file:
                ground_truth = cv2.imread(dataset_path + data_folder + '/' + file, cv2.IMREAD_GRAYSCALE)
                break
        
        dataset[data_folder] = {
            'flair': flair_image,
            'ground_truth': ground_truth
        }
        
    seg_results = {}
    seg_metrics = {}

    for key, data in dataset.items():
        print(key)
        print(data['flair'].shape)
        print(data['ground_truth'].shape)
        print()
        
        seg_metrics[key] = {}
        seg_seeds = {}
        
        seg_v1, seed_v1 = region_growing_v1(data['flair'], None, threshold=6)
        seg_v2, seed_v2 = region_growing_v2(data['flair'], None, threshold_factor=0.3)
        seg_v3, seed_v3 = region_growing_v3(data['flair'], None, initial_threshold_factor=0.15, adjustment_factor=0.05)
        
        seg_results[key] = {
            'v1': seg_v1,
            'v2': seg_v2,
            'v3': seg_v3,
        }
        seg_seeds = {
            'v1': seed_v1,
            'v2': seed_v2,
            'v3': seed_v3,
        }
        
        # save the results in the results directory
        if not os.path.exists(RESULTS_DIR + key):
            os.makedirs(RESULTS_DIR + key)
            
        for key2, value in seg_results[key].items():
            # save the ground truth
            cv2.imwrite(RESULTS_DIR + key + '/' + 'flair.png', data['flair'])
            
            cv2.imwrite(RESULTS_DIR + key + '/' + key2 + '.png', value)
            # save the intersection of the ground truth and the segmentation results
            intersection = draw_intersection_of_binary_images_v3(data['ground_truth'], value)
            cv2.imwrite(RESULTS_DIR + key + '/' + key2 + '_intersection.png', intersection)
            
            iou = IoU(data['ground_truth'], value)
            dice = dice_similarity(data['ground_truth'], value)
            
            seg_metrics[key][key2] = {
                'IoU': iou,
                'dice': dice
            }
            
            # create a plot that shows the segmentation results (value)
            # and a circle that shows the seed point value
            # do it without showing the axes and no white space around the image
            # then save the plot in the results directory
            
            fig, ax = plt.subplots()
            ax.imshow(value, cmap='gray')
            ax.axis('off')
            ax.set_aspect('equal')
            ax.set_xlim(0, value.shape[1])
            ax.set_ylim(value.shape[0], 0)
            ax.scatter(seg_seeds[key2][1], seg_seeds[key2][0], s=50, c='red', marker='o')
            fig.savefig(RESULTS_DIR + key + '/' + key2 + '_seed.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
        # save all results of all version including the IoU and dice similarity in the results directory as a dict in a json file
        # make it look pretty
        with open(RESULTS_DIR + key + '/' + 'metrics.json', 'w') as f:
            json.dump(seg_metrics[key], f, indent=4)
            
        # save the ground truth in the results directory
        cv2.imwrite(RESULTS_DIR + key + '/ground_truth.png', data['ground_truth'])
        
    # create a final json file with all the metrics for all the versions
    # and save it in the results directory
    # and give metrics for the whole dataset
    with open(RESULTS_DIR + 'metrics.json', 'w') as f:
        json.dump(seg_metrics, f, indent=4)

    # calculate the mean IoU and dice similarity for the whole dataset images
    # you should produce a value (mean IoU) and dice similarity) for each version
    # example {'v1': {'IoU': 0.5, 'dice': 0.6}, 'v2': {'IoU': 0.7, 'dice': 0.8}, 'v3': {'IoU': 0.9, 'dice': 0.1}}
    mean_metrics = {}

    for key, value in seg_metrics.items():
        for key2, value2 in value.items():
            if key2 not in mean_metrics:
                mean_metrics[key2] = {'IoU': 0, 'dice': 0}
            mean_metrics[key2]['IoU'] += value2['IoU']
            mean_metrics[key2]['dice'] += value2['dice']
            
    for key, value in mean_metrics.items():
        value['IoU'] /= len(seg_metrics)
        value['dice'] /= len(seg_metrics)
        
    with open(RESULTS_DIR + 'mean_metrics.json', 'w') as f:
        json.dump(mean_metrics, f, indent=4)
        
    pprint(mean_metrics)

    # create a plot that shows the mean IoU and dice similarity for all the versions
    # save the plot in the results directory
    # add a vertical line for each of the values to show the exact metric value
    # make it look pretty
    
    fig, ax = plt.subplots()
    ax.bar(mean_metrics.keys(), [value['IoU'] for value in mean_metrics.values()])
    ax.set_ylim(0, 1)
    ax.set_title('Mean IoU')
    fig.savefig(RESULTS_DIR + 'mean_iou.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    

    fig, ax = plt.subplots()
    ax.bar(mean_metrics.keys(), [value['dice'] for value in mean_metrics.values()])
    ax.set_ylim(0, 1)
    ax.set_title('Mean Dice Similarity')
    fig.savefig(RESULTS_DIR + 'mean_dice.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)