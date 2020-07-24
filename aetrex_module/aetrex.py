"""
Mask R-CNN
Written by Vaneesh

------------------------------------------------------------

Usage: import the module or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 aetrex_module.py train --dataset=/path/to/aetrex_module/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 aetrex_module.py train --dataset=/path/to/aetrex_module/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 aetrex_module.py train --dataset=/path/to/aetrex_module/dataset --weights=imagenet

"""
import glob
import os
import shutil
import sys
import json
import datetime
import argparse
import numpy as np
import skimage.draw
from pathlib import Path
import random

import os

import mlflow.keras
import mlflow.tensorflow
import imgaug

from aetrex_module.mask_generation import get_filename_list

mlflow.keras.autolog()
mlflow.tensorflow.autolog()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class AetrexConfig(Config):
    """Configuration for training on the aetrex_module  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "aetrex_module"

    # We use a GPU with 12 GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1 + 1  # Background + foot + foot + coin

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class AetrexDataset(utils.Dataset):

    def load_aetrex(self, dataset_dir, subset):
        """Load a subset of the aetrex_module dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only 3 classes to add.
        self.add_class("aetrex_module", 1, "left-foot")
        self.add_class("aetrex_module", 2, "right-foot")
        self.add_class("aetrex_module", 3, "coin")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        json_files = glob.glob(dataset_dir + '/*.json')  # load all json files

        # iterate via each json file one by one
        for json_file in json_files:
            annotations = json.load(open(json_file))

            # Add images
            image_path = os.path.join(dataset_dir, annotations['imagePath'])

            # check if image exists for the json if not then skip it
            my_file = Path(image_path)
            if not my_file.is_file():
                continue

            # adding polygons cordinates
            polygons = []
            # add all x and y points seperately
            for a in annotations['shapes']:
                polygon = {"label": a['label'],
                           "all_points_x": [x[0] for x in a['points']],
                           "all_points_y": [x[1] for x in a['points']]
                           }
                polygons.append(polygon)

            self.add_image(
                "aetrex_module",
                image_id=annotations['imagePath'],  # use file name as a unique image id
                path=image_path,
                width=annotations['imageWidth'], height=annotations['imageHeight'],
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a aetrex_module dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "aetrex_module":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = []
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

            if p["label"] == "left foot":
                class_ids.append(1)
            elif p["label"] == "right foot":
                class_ids.append(2)
            else:
                class_ids.append(3)  # appending class for coin

        # Return mask, and array of class IDs of each instance.
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "aetrex_module":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = AetrexDataset()
    dataset_train.load_aetrex(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = AetrexDataset()
    dataset_val.load_aetrex(args.dataset, "val")
    dataset_val.prepare()

    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network as whole")

    # creating augmentations for tranining data using imgaug library
    augmentation = imgaug.augmenters.Sometimes(0.5, [
        imgaug.augmenters.Fliplr(0.5),
        imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0)),
        imgaug.augmenters.Affine(rotate=(-45, 45))
    ])

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=4000,
                layers='all',
                augmentation=augmentation)


def get_shuffled_image_and_json(dataset_path):
    '''This Fn create and randomly shuffle all the data in train(80%) and val(20%) directory '''
    all_json_files = [os.path.join(dataset_path, item) for item in get_filename_list(dataset_path, 'json')]

    all_image_files = [file.replace('json', 'jpg') for file in all_json_files if
                       os.path.isfile(file.replace('json', 'jpg'))]

    # removing files for which we didn't have images (e.g only json is present but image file is not)
    for idx, value in enumerate(all_json_files):
        if not (value.replace('json', 'jpg') in all_image_files):
            del all_json_files[idx]

    files = list(zip(all_json_files, all_image_files))
    # shuffling all the files
    return random.shuffle(files)


def split_dataset_in_train_val(dataset_path, files):
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    # creating directories for train and val
    train_files = files[:round(len(files) * .8)]
    val_files = files[-round(len(files) * .2):]
    # moving the shuffled files to train and val dir
    for file in train_files:
        shutil.move(file[0], os.path.join(train_dir, file[0].split('/')[-1]))
        shutil.move(file[1], os.path.join(train_dir, file[1].split('/')[-1]))
    for file in val_files:
        shutil.move(file[0], os.path.join(val_dir, file[0].split('/')[-1]))
        shutil.move(file[1], os.path.join(val_dir, file[1].split('/')[-1]))


############################################################
#  Training
############################################################

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect aetrex_module dataset feet.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/aetrex_module/dataset/",
                        help='Directory of the aetrex_module dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = AetrexConfig()
    else:
        class InferenceConfig(AetrexConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # randomly split data files for train and validation
        files = get_shuffled_image_and_json(args.dataset)
        split_dataset_in_train_val(args.dataset, files)
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
