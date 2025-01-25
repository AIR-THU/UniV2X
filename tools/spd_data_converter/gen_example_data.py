#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
#----------------------------------------------------------------#

import os
import json
import argparse
from tqdm import tqdm

def read_json(path_json):
    """Reads JSON file from given path.

    Args:
        path_json (str): Path to the JSON file.

    Returns:
        dict: The JSON content as a dictionary.
    """
    with open(path_json, "r") as load_f:
        return json.load(load_f)

def write_json(data, path_json):
    """Writes dictionary to a JSON file.

    Args:
        data (dict): Data to write to the JSON file.
        path_json (str): Path to the JSON file.
    """
    with open(path_json, "w") as dump_f:
        json.dump(data, dump_f)

def create_directory_structure(output_dataset_path):
    """Creates required directory structure for the dataset.

    Args:
        output_dataset_path (str): Base output path for the dataset.
    """
    os.system(f'rm -rf {output_dataset_path}')
    directory_structure = [
        'vehicle-side/label/lidar',
        'vehicle-side/label/camera',
        'infrastructure-side/label/virtuallidar',
        'infrastructure-side/label/camera',
        'cooperative/label',
        'cooperative/infrastructure-side',
        'cooperative/vehicle-side'
    ]

    for directory in directory_structure:
        os.makedirs(os.path.join(output_dataset_path, directory), exist_ok=True)

def update_label_file(input_label_file_path, output_label_file_path, token, operation=None):
    """Updates label file based on the given token and operation.

    Args:
        input_label_file_path (str): Path to the input label file.
        output_label_file_path (str): Path to the output label file.
        token (str): Token to identify the label to update.
        operation (tuple, optional): Operation to perform on the label.
    """
    if os.path.exists(input_label_file_path):
        label_info = read_json(input_label_file_path)
        if operation:
            for item in label_info:
                if item["token"] == token:
                    item["type"] = operation[0]
                    item["track_id"] = operation[1]
                    break
        else:
            for i in label_info:
                if i["token"] == token:
                    label_info.remove(i)
                    break
        write_json(label_info, output_label_file_path)

def update_label_from_json(input_dataset_path, output_dataset_path, update_label_info_path="./tools/spd_data_converter/update_label_info.json"):
    """Updates labels based on the provided JSON file.

    Args:
        input_dataset_path (str): Path to the input dataset.
        output_dataset_path (str): Path to the output dataset.
        update_label_info_path (str, optional): Path to the update label info JSON file.
    """
    update_label_info = read_json(update_label_info_path)

    for item in update_label_info["vehicle-side"]:
        update_label_file(
            os.path.join(input_dataset_path, 'vehicle-side/label/lidar', f'{item["frame"]}.json'),
            os.path.join(output_dataset_path, 'vehicle-side/label/lidar', f'{item["frame"]}.json'),
            item["token"],
            item["operation"]
        )

    for item in update_label_info["infrastructure-side"]:
        update_label_file(
            os.path.join(input_dataset_path, 'infrastructure-side/label/virtuallidar', f'{item["frame"]}.json'),
            os.path.join(output_dataset_path, 'infrastructure-side/label/virtuallidar', f'{item["frame"]}.json'),
            item["token"],
            item["operation"]
        )
    
    for item in update_label_info["cooperative"]:
        update_label_file(
            os.path.join(input_dataset_path, 'cooperative/label', f'{item["frame"]}.json'),
            os.path.join(output_dataset_path, 'cooperative/label', f'{item["frame"]}.json'),
            item["token"],
            item["operation"]
        )

def gen_sequence_data_info(input_dataset_path, output_dataset_path, list_sequences, freq):
    """Generates sequence data information.

    Args:
        input_dataset_path (str): Path to the input dataset.
        output_dataset_path (str): Path to the output dataset.
        list_sequences (list): List of sequences to process.
        freq (int): Sampling frequency.
    """
    if not os.path.exists(output_dataset_path):
        create_directory_structure(output_dataset_path)

    input_paths = {
        'vehicle': os.path.join(input_dataset_path, 'vehicle-side/data_info.json'),
        'infrastructure': os.path.join(input_dataset_path, 'infrastructure-side/data_info.json'),
        'cooperative': os.path.join(input_dataset_path, 'cooperative/data_info.json')
    }
    output_paths = {
        'vehicle': os.path.join(output_dataset_path, 'vehicle-side/data_info.json'),
        'infrastructure': os.path.join(output_dataset_path, 'infrastructure-side/data_info.json'),
        'cooperative': os.path.join(output_dataset_path, 'cooperative/data_info.json')
    }

    data_info = {key: read_json(path) for key, path in input_paths.items()}
    dict_frame2info = {
        'vehicle': {info["frame_id"]: info for info in data_info['vehicle']},
        'infrastructure': {info["frame_id"]: info for info in data_info['infrastructure']},
        'cooperative': {info["vehicle_frame"]: info for info in data_info['cooperative']}
    }

    dict_coop_sequence2frames = {}
    for info in data_info['cooperative']:
        seq = info["vehicle_sequence"]
        dict_coop_sequence2frames.setdefault(seq, []).append(info["vehicle_frame"])

    interval = 10 // freq
    output_data_info = {'vehicle': [], 'infrastructure': [], 'cooperative': []}

    for sequence in list_sequences:
        print("sequence:", sequence)
        seq_frames = list(sorted(dict_coop_sequence2frames[sequence]))
        list_interval = []
        for i in range(int(seq_frames[0]), int(seq_frames[-1]) + 1, interval):
            list_interval.append(i)
        list_interval.append(list_interval[-1] + interval)
        # print("list_interval:", list_interval)
        # print("source frames:", seq_frames)

        c = 0
        seq_frames_new = []
        for veh_frame in seq_frames:
            for j in range(c, len(list_interval) - 2):
                if int(veh_frame) >= list_interval[j + 1]:
                    print(veh_frame)
                    c += 1
            if list_interval[c] <= int(veh_frame) < list_interval[c + 1]:
                seq_frames_new.append(veh_frame)
                output_data_info['cooperative'].append(dict_frame2info['cooperative'][veh_frame])
                c += 1

    for coop_info in tqdm(output_data_info['cooperative']):
        output_data_info['vehicle'].append(dict_frame2info['vehicle'][coop_info["vehicle_frame"]])
        output_data_info['infrastructure'].append(dict_frame2info['infrastructure'][coop_info["infrastructure_frame"]])

    for key in output_paths:
        write_json(output_data_info[key], output_paths[key])

def filt_label(input_label_file, output_label_file):
    """Filters duplicate track IDs from label file.

    Args:
        input_label_file (str): Path to the input label file.
        output_label_file (str): Path to the output label file.
    """
    label_info = read_json(input_label_file)
    new_label_info = []
    track_ids = set()

    for item in label_info:
        if item["track_id"] not in track_ids:
            new_label_info.append(item)
            track_ids.add(item["track_id"])
        else:
            print(f"Duplicate track ID {item['track_id']} in {input_label_file}")

    write_json(new_label_info, output_label_file)

def filt_label_coop(input_label_file, output_label_file):
    """Filters cooperative label file.

    Args:
        input_label_file (str): Path to the input label file.
        output_label_file (str): Path to the output label file.
    """
    label_info = read_json(input_label_file)
    new_label_info = []
    track_ids = set()
    filt_count = 0

    for item in label_info:
        if item["from_side"] != "inf":
            filt_count += 1
            continue
        if item["track_id"] not in track_ids:
            new_label_info.append(item)
            track_ids.add(item["track_id"])
        else:
            print(f"Duplicate track ID {item['track_id']} in {input_label_file}")

    print(f"Filtered {filt_count} objects")
    write_json(new_label_info, output_label_file)

def copy_dataset(input_dataset_path, output_dataset_path, update_label):
    """Copies and optionally updates the dataset.

    Args:
        input_dataset_path (str): Path to the input dataset.
        output_dataset_path (str): Path to the output dataset.
        update_label (bool): Whether to update labels.
    """
    veh_data_info = read_json(os.path.join(output_dataset_path, 'vehicle-side/data_info.json'))
    inf_data_info = read_json(os.path.join(output_dataset_path, 'infrastructure-side/data_info.json'))
    coop_data_info = read_json(os.path.join(output_dataset_path, 'cooperative/data_info.json'))

    for i in tqdm(veh_data_info):
        os.system(f"cp -f {input_dataset_path}/vehicle-side/{i['label_lidar_std_path']} {output_dataset_path}/vehicle-side/label/lidar/")
        os.system(f"cp -f {input_dataset_path}/vehicle-side/{i['label_camera_std_path']} {output_dataset_path}/vehicle-side/label/camera/")
    # os.system(f"cp -r {input_dataset_path}/vehicle-side/label {output_dataset_path}/vehicle-side")
    os.system(f"ln -s {input_dataset_path}/vehicle-side/calib {output_dataset_path}/vehicle-side/calib")
    os.system(f"ln -s {input_dataset_path}/vehicle-side/image {output_dataset_path}/vehicle-side/image")
    os.system(f"ln -s {input_dataset_path}/vehicle-side/velodyne {output_dataset_path}/vehicle-side/velodyne")

    for j in tqdm(inf_data_info):
        os.system(f"cp {input_dataset_path}/infrastructure-side/{j['label_lidar_std_path']} {output_dataset_path}/infrastructure-side/label/virtuallidar/")
        os.system(f"cp {input_dataset_path}/infrastructure-side/{j['label_camera_std_path']} {output_dataset_path}/infrastructure-side/label/camera/")
    # os.system(f"cp -r {input_dataset_path}/infrastructure-side/label {output_dataset_path}/infrastructure-side")
    os.system(f"ln -s {input_dataset_path}/infrastructure-side/calib {output_dataset_path}/infrastructure-side/calib")
    os.system(f"ln -s {input_dataset_path}/infrastructure-side/image {output_dataset_path}/infrastructure-side/image")
    os.system(f"ln -s {input_dataset_path}/infrastructure-side/velodyne {output_dataset_path}/infrastructure-side/velodyne")

    for k in tqdm(coop_data_info):
        os.system(f"cp {input_dataset_path}/cooperative/label/{k['vehicle_frame']}.json {output_dataset_path}/cooperative/label/")
    # os.system(f"cp -r {input_dataset_path}/cooperative/label {output_dataset_path}/cooperative")
    os.system(f"ln -s {input_dataset_path}/vehicle-side/calib {output_dataset_path}/cooperative/calib")
    os.system(f"ln -s {input_dataset_path}/vehicle-side/image {output_dataset_path}/cooperative/image")
    os.system(f"ln -s {input_dataset_path}/vehicle-side/velodyne {output_dataset_path}/cooperative/velodyne")
    os.system(f"ln -s {input_dataset_path}/maps {output_dataset_path}/maps")

    os.system(f"ln -s {input_dataset_path}/vehicle-side/calib {output_dataset_path}/cooperative/vehicle-side/calib")
    os.system(f"ln -s {input_dataset_path}/vehicle-side/image {output_dataset_path}/cooperative/vehicle-side/image")
    os.system(f"ln -s {input_dataset_path}/vehicle-side/velodyne {output_dataset_path}/cooperative/vehicle-side/velodyne")
    os.system(f"ln -s {input_dataset_path}/infrastructure-side/calib {output_dataset_path}/cooperative/infrastructure-side/calib")
    os.system(f"ln -s {input_dataset_path}/infrastructure-side/image {output_dataset_path}/cooperative/infrastructure-side/image")
    os.system(f"ln -s {input_dataset_path}/infrastructure-side/velodyne {output_dataset_path}/cooperative/infrastructure-side/velodyne")

    if update_label:
        update_label_from_json(output_dataset_path, output_dataset_path)
    
    for info in tqdm(veh_data_info):
        filt_label(
            os.path.join(output_dataset_path, 'vehicle-side', info['label_lidar_std_path']),
            os.path.join(output_dataset_path, 'vehicle-side', info['label_lidar_std_path'])
        )
    
    for info in tqdm(inf_data_info):
        filt_label(
            os.path.join(output_dataset_path, 'infrastructure-side', info['label_lidar_std_path']),
            os.path.join(output_dataset_path, 'infrastructure-side', info['label_lidar_std_path'])
        )

if __name__ == "__main__":
    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_folder_path, '../..'))
    cur_directory = os.getcwd()
    print(cur_directory)

    parser = argparse.ArgumentParser(description='Generate and copy dataset files.')
    parser.add_argument('--input', type=str, required=True, help='Input dataset path')
    parser.add_argument('--output', type=str, required=True, help='Output dataset path')
    parser.add_argument('--sequences', nargs='+', required=True, help='List of sequences to process')
    parser.add_argument('--update-label', action='store_true', default=False, help='Whether to update labels.')
    parser.add_argument('--freq', type=int, default=2, help='Sample frequency.')
    args = parser.parse_args()

    if 'all' in args.sequences:
        # means that all sequences are used
        list_sequences = []
        for i in range(95):
            list_sequences.append(f"{i:04d}")

        test = ["0006", "0009", "0011", "0012", "0013", 
                "0019", "0024", "0026", "0027", "0028", 
                "0031", "0038", "0039", "0043", "0044", 
                "0045", "0046", "0051", "0053", "0064", 
                "0065", "0067", "0069", "0074", "0076", 
                "0083", "0090", "0091"]
        list_sequences = [i for i in list_sequences if i not in test]
        args.sequences = list_sequences

    create_directory_structure(args.output)
    gen_sequence_data_info(args.input, args.output, args.sequences, args.freq)
    copy_dataset(args.input, args.output, args.update_label)
