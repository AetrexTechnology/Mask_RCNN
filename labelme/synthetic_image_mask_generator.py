import sys, os

sys.path.append('/Users/vaneesh_k/PycharmProjects/Mask_RCNN/labelme')

import argparse
import glob
import os
import random
import json
import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw

from augment import augmenting
from imantics import Polygons, Mask

LEFT_FOOT_LABEL = 1
RIGHT_FOOT_LABEL = 2
COIN_LABEL = 3


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--annotation_dir_name", type=str, default="annotation", required=False)
    parser.add_argument("--image_dir_name", type=str, default="images", required=False)
    parser.add_argument("--mask_dir_name", type=str, default="mask", required=False)
    parser.add_argument("--coin_dir_name", type=str, default="coin_images", required=False)
    parser.add_argument("--output_dir_name", type=str, default="final_training_data", required=False)
    parser.add_argument("--annotation_template_name", type=str, default="final_training_data", required=False)
    args = parser.parse_args()
    return args


def get_filename_list(dir_name, ext="*", recursive=False):
    """
    generate all filenames under dir_name (exclude dir_name)
    Args:
        dir_name: target directory name
        ext: file extension
        recursive: recursive seraching
    Returns:
        filename_list: list including only filenames
    """
    filename_list = []
    if recursive:
        path = os.path.join(dir_name, "**", "*." + ext)
    else:
        path = os.path.join(dir_name, "*." + ext)
    for file_path in glob.glob(path):
        filename = file_path.split("/")[-1]
        filename_list.append(filename)

    return filename_list


def read_image_using_metadata(image_path):
    """
    read image and rotate image using metadata
    Args:
        image_path
    Returns:
        img: cv (numpy) image
    """
    img = Image.open(image_path)
    # rotate image using metadata
    # print(img.size)
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == "Orientation":
            break
    if img._getexif() is not None:
        exif = dict(img._getexif().items())
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)

    cv_img = np.array(img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img


def get_coin_position(mask, coin_size=100):
    h, w = mask.shape
    kernel = np.ones((coin_size * 2, coin_size * 2), np.uint8)
    dilation = np.where(mask > 0, 255, 0).astype(np.uint8)
    dilation = cv2.dilate(dilation, kernel, iterations=1)
    while True:
        if h == w:
            x = np.random.randint(coin_size, w - coin_size)
            y = np.random.randint(coin_size, h - coin_size)
        else:
            x = np.random.randint(coin_size, w - coin_size)
            y = np.random.randint(h // 4, h - h // 4)
        if dilation[y, x] == 0:
            return (x, y)


def get_coin_data(coin_dir_name, coin_size=100, display=False):
    coin_filename_list = get_filename_list(coin_dir_name, "*", False)
    coin_path = os.path.join(coin_dir_name, random.choice(coin_filename_list))
    coin_img = Image.open(coin_path)
    coin_img = coin_img.resize((coin_size, coin_size))
    coin_mask = Image.new("L", coin_img.size, 0)
    ImageDraw.Draw(coin_mask).ellipse((0, 0, coin_mask.size), fill=COIN_LABEL)

    cv_coin_img = np.array(coin_img)
    cv_coin_img = cv2.cvtColor(cv_coin_img, cv2.COLOR_RGB2BGR)
    cv_coin_mask = np.array(coin_mask)

    # augmentation for coin
    aug_coin, aug_size = augmenting(cv_coin_img)
    aug_mask = cv2.resize(cv_coin_mask, aug_size)

    if display:
        cv_coin_overlay = np.array(aug_coin)
        cv_coin_overlay[:, :, 1][aug_mask == COIN_LABEL] = 0
        cv_coin_overlay[:, :, 2][aug_mask == COIN_LABEL] = 255
        cv2.imshow("cv_coin_img", aug_coin)
        cv2.imshow("cv_coin_mask", aug_mask)
        cv2.imshow("cv_coin_overlay", cv_coin_overlay)
        k = cv2.waitKey(0)
        if k & 0xFF == ord("q"):
            quit()
    return aug_coin, aug_mask


def generate_syntetic_data(xy_position, image, mask, coin_dir_name, coin_size):
    x, y = xy_position
    final_image = image.copy()
    coin_img, coin_mask = get_coin_data(coin_dir_name=coin_dir_name, coin_size=coin_size, display=False)
    coin_in_image_size = np.zeros(image.shape, np.uint8)
    coin_mask_in_image_size = np.zeros(mask.shape, np.uint8)
    coin_in_image_size[y: y + coin_img.shape[0], x: x + coin_img.shape[1], :] = coin_img
    coin_mask_in_image_size[y: y + coin_mask.shape[0], x: x + coin_mask.shape[1]] = coin_mask
    final_image[coin_mask_in_image_size == COIN_LABEL] = coin_in_image_size[coin_mask_in_image_size == COIN_LABEL]
    final_mask = mask + coin_mask_in_image_size

    return final_image, final_mask


def display(image, mask):
    if mask.shape[0] > 4000:
        ratio = 0.2
        small = cv2.resize(image, None, fx=ratio, fy=ratio)
        m_small = cv2.resize(mask, None, fx=ratio, fy=ratio)
    elif mask.shape[0] > 2000:
        ratio = 0.4
        small = cv2.resize(image, None, fx=ratio, fy=ratio)
        m_small = cv2.resize(mask, None, fx=ratio, fy=ratio)
    else:
        ratio = 0.7
        small = cv2.resize(image, None, fx=ratio, fy=ratio)
        m_small = cv2.resize(mask, None, fx=ratio, fy=ratio)
    cv2.imshow("image", small)
    cv2.imshow("mask", m_small)
    k = cv2.waitKey(0)
    if k == ord("q"):
        quit()


def save_images(output_dir, file_id, image, mask):
    image_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    image_path = os.path.join(image_dir, str(file_id) + ".jpg")
    mask_path = os.path.join(mask_dir, str(file_id) + ".jpg")
    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, mask)


def create_polygon_coordinates_from_mask(mask, image):
    mask[mask > 0] = 255
    ret, thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 6)

    # sort contours by area so that coin contour comes at top
    # only taknig last 3 contours some times small false + contours are found
    sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-3:]

    for idx, contour in enumerate(sorted_contours):
        if idx == 0:
            # skiiping coordinates by 15 else it will add many points which are redundant
            all_x_coin_coordinates = contour[:, 0, 0][0::15]
            all_y_coin_coordinates = contour[:, 0, 1][0::15]

        else:
            # if true that means its left foot cordinates
            if contour[0][0][0] < sorted_contours[idx + 1][0][0][0]:
                # skiiping coordinates by 60 else it will add many points(1500 aprox and we need only 20-30) which are redundant
                all_x_left_coordinates = contour[:, 0, 0][0::60]
                all_y_left_coordinates = contour[:, 0, 1][0::60]

                all_x_right_coordinates = sorted_contours[idx + 1][:, 0, 0][0::60]
                all_y_right_coordinates = sorted_contours[idx + 1][:, 0, 1][0::60]

            # then this is right foot coordinates
            else:
                all_x_right_coordinates = contour[:, 0, 0][0::60]
                all_y_right_coordinates = contour[:, 0, 1][0::60]

                all_x_left_coordinates = sorted_contours[idx + 1][:, 0, 0][0::60]
                all_y_left_coordinates = sorted_contours[idx + 1][:, 0, 1][0::60]
            break

    return (all_x_coin_coordinates, all_y_coin_coordinates, all_x_left_coordinates, all_y_left_coordinates,
            all_x_right_coordinates, all_y_right_coordinates)


def create_annotated_json(json_template, image_shape, output_dir, file_id, coordinates):
    annotations = json.load(open(json_template))  # loading template file

    for annotation in annotations['shapes']:
        if annotation['label'] == 'coin':
            annotation['points'] = [[x, y]
                                    for x, y in zip(coordinates[0].tolist(), coordinates[1].tolist())]

        elif annotation['label'] == 'left foot':
            annotation['points'] = [[x, y]
                                    for x, y in zip(coordinates[2].tolist(), coordinates[3].tolist())]
            # for right foot
        else:
            annotation['points'] = [[x, y]
                                    for x, y in zip(coordinates[4].tolist(), coordinates[5].tolist())]

    # save the annotation json
    annotations['imagePath'] = str(file_id) + ".jpg"
    annotations['imageHeight'] = image_shape[0]
    annotations['imageWidth'] = image_shape[1]
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    json_path = os.path.join(annotations_dir, str(file_id) + ".json")

    with open(json_path, 'w') as json_file:
        json.dump(annotations, json_file)


def main():
    args = get_args()
    mask_filename_list = get_filename_list(args.mask_dir_name, "jpg", False)
    file_id = 1
    for mask_filename in mask_filename_list:
        image_path = os.path.join(args.image_dir_name, mask_filename)
        print(image_path)
        image = read_image_using_metadata(image_path)

        h, w, _ = image.shape
        coin_size = int(100 * (h / 4032))
        mask_path = os.path.join(args.mask_dir_name, mask_filename)
        # print(mask_path)
        mask = cv2.imread(mask_path, 0)
        margin = w // 10
        half_image = image[h - w + margin: h - margin, margin: w - margin]
        half_mask = mask[h - w + margin: h - margin, margin: w - margin]
        print(mask.shape, half_mask.shape)
        coin_position = get_coin_position(mask, coin_size=coin_size)

        # generating full mask and image
        final_image, final_mask = generate_syntetic_data(coin_position, image, mask, args.coin_dir_name, coin_size)
        # display(final_image, final_mask * 85)

        # creating polygon cordinates
        (all_x_coin_coordinates, all_y_coin_coordinates, all_x_left_coordinates, all_y_left_coordinates,
         all_x_right_coordinates, all_y_right_coordinates) = create_polygon_coordinates_from_mask(final_mask,
                                                                                                  final_image)

        # creating annotations and saving them in json file
        create_annotated_json(args.annotation_template_name, final_image.shape, args.output_dir_name, file_id, (
            all_x_coin_coordinates, all_y_coin_coordinates, all_x_left_coordinates, all_y_left_coordinates,
            all_x_right_coordinates, all_y_right_coordinates))

        save_images(args.output_dir_name, file_id, final_image, final_mask)
        file_id += 1

        # generating half image and mask
        coin_position = get_coin_position(half_mask, coin_size=coin_size)
        final_image, final_mask = generate_syntetic_data(coin_position, half_image, half_mask, args.coin_dir_name,
                                                         coin_size)
        # creating annotations file
        (all_x_coin_coordinates, all_y_coin_coordinates, all_x_left_coordinates, all_y_left_coordinates,
         all_x_right_coordinates, all_y_right_coordinates) = create_polygon_coordinates_from_mask(final_mask,
                                                                                                  final_image)

        # creating annotations and saving them in json file
        create_annotated_json(args.annotation_template_name, final_image.shape, args.output_dir_name, file_id, (
            all_x_coin_coordinates, all_y_coin_coordinates, all_x_left_coordinates, all_y_left_coordinates,
            all_x_right_coordinates, all_y_right_coordinates))

        # display(final_image, final_mask * 85)
        save_images(args.output_dir_name, file_id, final_image, final_mask)
        file_id += 1


if __name__ == "__main__":
    main()
