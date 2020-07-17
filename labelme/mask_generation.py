import argparse
import glob
import json
import os

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw

LEFT_FOOT_LABEL = 1
RIGHT_FOOT_LABEL = 2
COIN_LABEL = 3


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--annotation_dir_name", type=str, default="annotation", required=False)
    parser.add_argument("--image_dir_name", type=str, default="images", required=False)
    parser.add_argument("--mask_dir_name", type=str, default="mask", required=False)
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


def read_labelme_annotation(dir_name, filename):
    """
    read JSON file generated by LabelMe and return annotation in dict ({label: polygon points})
    Args:
        dir_name: annotation directory name
        filename: JSON filename
    Returns:
        annotated_data: {label: polygon points}
            polygon points --> [(x1, y1), (x2, y2), ...]
    """
    with open(os.path.join(dir_name, filename)) as f:
        data = json.load(f)

    annotated_data = {}
    for key, value in data.items():
        if key == "shapes":
            for i in range(len(value)):
                polygon = [tuple(i) for i in value[i]["points"]]
                annotated_data[value[i]["label"]] = polygon
    return annotated_data


def generate_mask(annotated_data, output_dir, image_dir, json_filename, image_ext="jpg", mask_ext="jpg"):
    """
    generate mask images (left foot: 125, right foot: 255)
    Args:
        annotated_data: output of read_labelme_annotation()
        output_dir:
        image_dir:
        json_filename:
        image_ext: image file extension
        mask_ext: mask file extension
    Returns:
    """
    image_path = os.path.join(image_dir, json_filename.replace("json", image_ext))
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
        # print(img.size)

    width, height = img.size
    mask = Image.new("L", (width, height), 0)
    for label, polygon in annotated_data.items():
        if label == "left foot":
            ImageDraw.Draw(mask).polygon(polygon, outline=LEFT_FOOT_LABEL, fill=LEFT_FOOT_LABEL)
        else:
            ImageDraw.Draw(mask).polygon(polygon, outline=RIGHT_FOOT_LABEL, fill=RIGHT_FOOT_LABEL)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, json_filename.replace("json", mask_ext))
    mask.save(output_path)

    cv_img = np.array(img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    cv_img_original = cv_img.copy()
    cv_mask = np.array(mask)
    print(cv_img.shape, cv_mask.shape)
    cv_img[:, :, 1][cv_mask == LEFT_FOOT_LABEL] = 0
    cv_img[:, :, 2][cv_mask == LEFT_FOOT_LABEL] = 255
    cv_img[:, :, 1][cv_mask == RIGHT_FOOT_LABEL] = 255
    cv_img[:, :, 2][cv_mask == RIGHT_FOOT_LABEL] = 0
    if cv_mask.shape[0] > 4000:
        small = cv2.resize(cv_img, None, fx=0.3, fy=0.3)
        o_small = cv2.resize(cv_img_original, None, fx=0.3, fy=0.3)
    elif cv_mask.shape[0] > 2000:
        small = cv2.resize(cv_img, None, fx=0.5, fy=0.5)
        o_small = cv2.resize(cv_img_original, None, fx=0.5, fy=0.5)
    else:
        small = cv2.resize(cv_img, None, fx=0.7, fy=0.7)
        o_small = cv2.resize(cv_img_original, None, fx=0.7, fy=0.7)
    cv2.imshow("overlay_img", small)
    cv2.imshow("image", o_small)
    k = cv2.waitKey(100)
    if k == ord("q"):
        quit()


def main():
    args = get_args()
    json_filename_list = get_filename_list(args.annotation_dir_name, "json", False)
    for json_filename in json_filename_list:
        print(json_filename)
        annotated_data = read_labelme_annotation(args.annotation_dir_name, json_filename)
        generate_mask(annotated_data, args.mask_dir_name, args.image_dir_name, json_filename)


if __name__ == "__main__":
    main()
