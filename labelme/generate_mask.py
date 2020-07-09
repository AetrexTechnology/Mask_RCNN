import argparse
import glob
import json
import os

import cv2
import numpy
from PIL import Image, ImageDraw


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--annotation_dir_name", type=str, default="annotation", required=False)
    parser.add_argument("--image_dir_name", type=str, default="images", required=False)
    parser.add_argument("--mask_dir_name", type=str, default="mask", required=False)
    args = parser.parse_args()
    return args


def get_filename_list(dir_name, ext, recursive):
    filename_list = []
    if recursive:
        path = os.path.join(dir_name, "**", "*." + ext)
    else:
        path = os.path.join(dir_name, "*." + ext)
    for file_path in glob.glob(path):
        filename = file_path.split("/")[-1]
        print(filename)
        filename_list.append(filename)

    return filename_list


def read_labelme_annotation(dir_name, filename):
    with open(os.path.join(dir_name, filename)) as f:
        data = json.load(f)

    annotated_data = {}
    for key, value in data.items():
        if key == "shapes":
            for i in range(len(value)):
                polygon = [tuple(i) for i in value[i]["points"]]
                annotated_data[value[i]["label"]] = polygon
    return annotated_data


def generate_mask(annotated_data, output_dir, image_dir, json_filename, image_ext="jpg", mask_ext="png"):
    image_path = os.path.join(image_dir, json_filename.replace("json", image_ext))
    img = Image.open(image_path)
    width, height = img.size
    # print(width, height)

    mask = Image.new("L", (width, height), 0)
    for label, polygon in annotated_data.items():
        ImageDraw.Draw(mask).polygon(polygon, outline=255, fill=255)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, json_filename.replace("json", mask_ext))
    mask.save(output_path)

    cv_img = numpy.array(img)
    cv_mask = numpy.array(mask)
    print(cv_img.shape, cv_mask.shape)


def main():
    args = get_args()
    json_filename_list = get_filename_list(args.annotation_dir_name, "*", False)
    for json_filename in json_filename_list:
        annotated_data = read_labelme_annotation(args.annotation_dir_name, json_filename)
        generate_mask(annotated_data, args.mask_dir_name, args.image_dir_name, json_filename)


if __name__ == "__main__":
    main()
