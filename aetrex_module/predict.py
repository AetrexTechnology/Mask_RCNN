"""
predict.py
This file does the prediction on the given directory of images if provided with the weights path.
Currently, Aetrex configuration is loaded so only gives the weights file trained on aetrex dataset.

Written by Vaneesh

------------------------------------------------------------

Usage: import the module or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 predict.py train --aetrex_weight_path=/path/to/aetrex_module/dataset --images_dir=/path/to/dir_of_images_for prediction
"""
import os
import sys
import matplotlib.pyplot as plt
import skimage.io
import argparse
import tensorflow as tf
import random

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append('./aetrex_module')

from aetrex_module import aetrex
from aetrex_module.mask_generation import get_filename_list
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.model import log

config = aetrex.AetrexConfig()


def setup_config_and_paths(weight_path, images_path):
    '''This Fn setup the path and configuration for prediction'''
    global MODEL_DIR, AETREX_WEIGHTS_PATH, AETREX_DIR, config
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    AETREX_WEIGHTS_PATH = weight_path
    AETREX_DIR = images_path
    config = InferenceConfig()
    config.display()


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--aetrex_weight_path", type=str, default="weights", required=True)
    parser.add_argument("--images_dir", type=str, default="images", required=True)
    args = parser.parse_args()
    return args


# Override the training configurations with a few
# changes for inferencing.

class InferenceConfig(config.__class__):
    '''This configuration use for prediction it inherits from the AetrexConfig'''
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def load_image_from_path(image_path):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(image_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def load_dataset_and_model():
    '''Load the dataset  and weight to create a model for prediction'''
    global dataset, model
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    # Load validation dataset
    dataset = aetrex.AetrexDataset()
    dataset.load_aetrex(AETREX_DIR, "val")
    # Must call before using the dataset
    dataset.prepare()
    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    # Set path to aetrex_module weights file
    weights_path = AETREX_WEIGHTS_PATH
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)


def show_predicted_result_for_loaded_dataset():
    '''This Fn shows the output image after prediction, after showing first result process waits for you to press Enter
    on each press it loads another image do the prediction and then shows the result'''

    for id in range(len(dataset.image_ids)):
        print('---------------------------------------- Processing Image ------------------------------------------\n')
        image_id = random.choice(dataset.image_ids)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))

        # uncomment below line if you wants to check single file, [..., :3] is given to ignore alpha channel if present
        # image = skimage.io.imread('/Users/vaneesh_k/Desktop/pics/cvBufferInput1.png')[..., :3]  jgj

        results = model.detect([image], verbose=1)
        # Display results
        ax = get_ax(1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions")
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
        input("------------------------------------------ press enter ---------------------------------------------\n")


def show_predicted_result_for_image_dir():
    AETREX_DIR_PATH = AETREX_DIR + '/val'
    all_images_name = get_filename_list(AETREX_DIR_PATH, 'jpg')

    for image_name in all_images_name:
        print('---------------------------------------- Processing Image ------------------------------------------\n')
        image_path = os.path.join(AETREX_DIR_PATH, image_name)
        print(f'-----------------------------------------Image Path : {image_path}-----------------------------------\n')
        image = load_image_from_path(image_path)
        results = model.detect([image], verbose=1)
        # Display results°
        ax = get_ax(1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions")
        input("------------------------------------------ press enter ---------------------------------------------\n")


def main():
    args = get_args()
    setup_config_and_paths(args.aetrex_weight_path, args.images_dir)
    load_dataset_and_model()
    # show_predicted_result_for_loaded_dataset()
    show_predicted_result_for_image_dir()


if __name__ == '__main__':
    main()
