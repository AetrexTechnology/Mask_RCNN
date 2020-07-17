import json
import os


def create_annotated_json():
    annotations = json.load(open("/Users/vaneesh_k/PycharmProjects/Mask_RCNN/labelme/annotation_template.json")) # loading template file
    print('Done')


if __name__ == "__main__":
    create_annotated_json()