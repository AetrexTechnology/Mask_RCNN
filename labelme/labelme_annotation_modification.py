import argparse
import glob
import json
import os


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--original_annotation_dir_name", type=str, default="vaneesh_annotation", required=False)
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


def save_json(json_path, data):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def modify_labelme_annotation(
    dir_name, filename, output_dir="annotation", key_name="imagePath", prefix_new_value="../images"
):
    """
    modify JSON file based on key_name and save it in output_dir
    Args:
        dir_name: annotation directory name
        filename: JSON filename
        output_dir: output directory name
        key_name: key name to change it's value
        prefix_new_value:
    Returns:
    """
    with open(os.path.join(dir_name, filename)) as f:
        data = json.load(f)

    old_value = data[key_name]
    new_value = os.path.join(prefix_new_value, old_value)
    data[key_name] = new_value

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    save_json(output_path, data)


def main():
    args = get_args()
    json_filename_list = get_filename_list(args.original_annotation_dir_name, "json", False)
    for json_filename in json_filename_list:
        print(json_filename)
        modify_labelme_annotation(args.original_annotation_dir_name, json_filename)


if __name__ == "__main__":
    main()
