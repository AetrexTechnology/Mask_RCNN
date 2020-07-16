"""
Mask R-CNN
Written by Vaneesh

------------------------------------------------------------

Usage: import the module or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 aetrex.py train --dataset=/path/to/aetrex/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 aetrex.py train --dataset=/path/to/aetrex/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 aetrex.py train --dataset=/path/to/aetrex/dataset --weights=imagenet

"""
import glob
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

import os

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
    """Configuration for training on the aetrex  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "aetrex"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1  # Background + foot + foot + coin

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class AetrexDataset(utils.Dataset):

    def load_aetrex(self, dataset_dir, subset):
        """Load a subset of the aetrex dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only 3 classes to add.
        self.add_class("aetrex", 1, "left-foot")
        self.add_class("aetrex", 2, "right-foot")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        json_files = glob.glob(dataset_dir + '/*.json')  # load all json files

        # iterate via each json file one by one
        for json_file in json_files:
            annotations = json.load(open(json_file))

            # Add images
            image_path = os.path.join(dataset_dir, annotations['imagePath'])

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
                "aetrex",
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
        # If not a aetrex dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "aetrex":
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
            # else:
            #     class_ids.append(3)  # appending class for coin

        # Return mask, and array of class IDs of each instance.
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "aetrex":
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
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect aetrex dataset feet.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/aetrex/dataset/",
                        help='Directory of the aetrex dataset')
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
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
