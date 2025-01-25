#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
#----------------------------------------------------------------#

import argparse
import shutil
import os
import os.path as osp
import uuid
import pyquaternion

from spd_to_uniad import create_spd_infos, _get_instance_token_mappings,create_spd_infos_coop
from spd_to_uniad import load_json, write_json, visibility_mappings


def generate_category_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating category json ---------------------")
    if not os.path.exists(osp.join(data_root, version)):
        os.mkdir(osp.join(data_root, version))

    sr_file = osp.join(local_root, 'category.json')
    target_file_path = osp.join(data_root, version, 'category.json')
    shutil.copy(sr_file, target_file_path)


def generate_attribute_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating attribute json ---------------------")
    if not os.path.exists(osp.join(data_root, version)):
        os.mkdir(osp.join(data_root, version))

    sr_file = osp.join(local_root, 'attribute.json')
    target_file_path = osp.join(data_root, version, 'attribute.json')
    shutil.copy(sr_file, target_file_path)


def generate_visibility_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating visibility json ---------------------")
    if not os.path.exists(osp.join(data_root, version)):
        os.mkdir(osp.join(data_root, version))

    sr_file = osp.join(local_root, 'visibility.json')
    target_file_path = osp.join(data_root, version, 'visibility.json')
    shutil.copy(sr_file, target_file_path)


def generate_instance_json(total_annotations, sample_info_mappings, data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating instance json ---------------------")
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    category_data = load_json(osp.join(local_root, 'category.json'))
    category_token_mappings = {}
    for category_type in category_data:
        category_token_mappings[category_type['name']] = category_type['token']

    instance_json_datas = []
    for instance_token in instance_token_mappings.keys():
        cur_instance_samples = instance_token_mappings[instance_token]
        nbr_annotations = len(cur_instance_samples)
        category_name = cur_instance_samples[0]['annotation']['type']

        json_data = {
            'token': instance_token,
            'category_token': category_token_mappings[category_name],
            'nbr_annotations': nbr_annotations,
            'first_annotation_token': cur_instance_samples[0]['annotation']['token'],
            'last_annotation_token': cur_instance_samples[nbr_annotations - 1]['annotation']['token']
        }

        instance_json_datas.append(json_data)

    target_file_path = osp.join(data_root, version, 'instance.json')
    write_json(instance_json_datas, target_file_path)


def generate_sensor_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating sensor json ---------------------")
    if not os.path.exists(osp.join(data_root, version)):
        os.mkdir(osp.join(data_root, version))

    sr_file = osp.join(local_root, 'sensor.json')
    target_file_path = osp.join(data_root, version, 'sensor.json')
    shutil.copy(sr_file, target_file_path)


## UniV2X TODO: reduce hard code
def generate_calibrated_sensor_json(spd_infos, data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating calibrated_sensors json ---------------------")

    def gen_token(sensor_name, str_x, str_y, str_z):
        """
            Args:
                sensor_name: "VEHICLE_CAM_FRONT"
                str_x: str label["3d_location"]["x"]
                str_y: str label["3d_location"]["y"]
                str_z: str label["3d_location"]["z"]
            Returns:
                str(token)
        """
        token_name = sensor_name + str_x + str_y + str_z
        token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
        return str(token)

    sensor_data = load_json(osp.join(local_root, 'sensor.json'))
    calibrated_sensors = ['VEHICLE_CAM_FRONT', 'LIDAR_TOP']

    calibrated_sensor_infos = []
    for sensor_info in sensor_data:
        print(sensor_info['channel'])
        if sensor_info['channel'] in calibrated_sensors:
            if sensor_info['modality'] == 'camera':
                rotation = spd_infos[0]['cams'][sensor_info['channel']]['sensor2ego_rotation']
                translation = spd_infos[0]['cams'][sensor_info['channel']]['sensor2ego_translation']
                camera_intrinsic = spd_infos[0]['cams'][sensor_info['channel']]['cam_intrinsic']
                token = gen_token(sensor_info['channel'], str(translation[0]), str(translation[1]), str(translation[2]))
                info = {
                    'token': token,
                    'sensor_token': sensor_info['token'],
                    'translation': translation.tolist(),
                    'rotation': rotation.tolist(),
                    'camera_intrinsic': camera_intrinsic.tolist()
                }
            else:
                rotation = spd_infos[0]['lidar2ego_rotation']
                translation = spd_infos[0]['lidar2ego_translation']
                token = gen_token(sensor_info['channel'], str(translation[0]), str(translation[1]), str(translation[2]))
                info = {
                    'token': token,
                    'sensor_token': sensor_info['token'],
                    'translation': translation.tolist(),
                    'rotation': rotation.tolist(),
                    'camera_intrinsic': []
                }
            calibrated_sensor_infos.append(info)

    target_file_path = osp.join(data_root, version, 'calibrated_sensor.json')
    write_json(calibrated_sensor_infos, target_file_path)


def generate_ego_pose_json(spd_infos, data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating ego_pose json ---------------------")

    ego_pose_infos = []
    for spd_info in spd_infos:
        info = {
            'token': spd_info['token'],
            'timestamp': spd_info['timestamp'],
            'rotation': spd_info['ego2global_rotation'].tolist(),
            'translation': spd_info['ego2global_translation'].tolist()
        }

        ego_pose_infos.append(info)

    target_file_path = osp.join(data_root, version, 'ego_pose.json')
    write_json(ego_pose_infos, target_file_path)


def generate_log_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating log json ---------------------")

    if not os.path.exists(osp.join(data_root, version)):
        os.mkdir(osp.join(data_root, version))

    sr_file = osp.join(local_root, 'log.json')
    target_file_path = osp.join(data_root, version, 'log.json')
    shutil.copy(sr_file, target_file_path)


def generate_scene_json(sample_info_mappings, data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating scene json ---------------------")

    scene_mappings = {}
    for sample_token in sample_info_mappings.keys():
        scene_token = sample_info_mappings[sample_token]['scene_token']
        if scene_token not in scene_mappings.keys():
            scene_mappings[scene_token] = []

        scene_mappings[scene_token].append(sample_info_mappings[sample_token])

    log_data = load_json(osp.join(local_root, 'log.json'))
    log_mappings = {}
    for log in log_data:
        log_mappings[log['location']] = log['token']

    scene_infos = []
    for scene_token in scene_mappings.keys():
        nbr_samples = len(scene_mappings[scene_token])
        location = scene_mappings[scene_token][0]['location']
        info = {
            'token': scene_token,
            'log_token': log_mappings[location],
            'nbr_samples': nbr_samples,
            'first_sample_token': scene_mappings[scene_token][0]['token'],
            'last_sample_token': scene_mappings[scene_token][nbr_samples - 1]['token'],
            'name': scene_token,
            'description': ''
        }

        scene_infos.append(info)

    target_file_path = osp.join(data_root, version, 'scene.json')
    write_json(scene_infos, target_file_path)


## UniV2X TODO: remove the hard code about sensor_name
def generate_sample_json(sample_info_mappings, data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating sample json ---------------------")

    def gen_token(sensor_name, sample_token):
        """
            Args:
                sensor_name: "VEHICLE_CAM_FRONT"
                str_x: str label["3d_location"]["x"]
                str_y: str label["3d_location"]["y"]
                str_z: str label["3d_location"]["z"]
            Returns:
                str(token)
        """
        token_name = sensor_name + sample_token
        token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
        return str(token)

    sensor_name = 'VEHICLE_CAM_FRONT'
    for sample_token in sample_info_mappings.keys():
        sample_info_mappings[sample_token]['image_token'] = gen_token(sensor_name, sample_token)

    image_sample_infos = []
    for sample_token in sample_info_mappings.keys():
        sample_info = sample_info_mappings[sample_token]
        prev_token = '' if sample_info['prev'] == '' else sample_info_mappings[sample_info['prev']]['image_token']
        next_token = '' if sample_info['next'] == '' else sample_info_mappings[sample_info['next']]['image_token']
        info = {
            'token': sample_token,
            'timestamp': sample_info['image_timestamp'],
            'prev': sample_info['prev'],
            'next': sample_info['next'],
            'scene_token': sample_info['scene_token']
        }

        image_sample_infos.append(info)

    target_file_path = osp.join(data_root, version, 'sample.json')
    write_json(image_sample_infos, target_file_path)


def generate_sample_data_json(spd_infos, data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating sample data json ---------------------")

    calibrated_sensor_data = load_json(osp.join(data_root, version, 'calibrated_sensor.json'))
    for calibrated_sensor in calibrated_sensor_data:
        if calibrated_sensor['camera_intrinsic'] is not []:
            calibrated_sensor_token = calibrated_sensor['token']

    sample_data_infos = []
    for sample_info in spd_infos:
        info = {
            'token': sample_info['token'],
            'sample_token': sample_info['token'],
            'ego_pose_token': sample_info['token'],
            'calibrated_sensor_token': calibrated_sensor_token,
            'timestamp': sample_info['timestamp'],
            'fileformat': 'pcd',
            'is_key_frame': bool(1),
            'height': 0,
            'width': 0,
            'filename': sample_info['lidar_path'],
            'prev': sample_info['prev'],
            'next': sample_info['next']
        }

        sample_data_infos.append(info)

    target_file_path = osp.join(data_root, version, 'sample_data.json')
    write_json(sample_data_infos, target_file_path)


def rotation_z2quaternion(rotation_z):
    # https://www.zhihu.com/question/23005815/answer/33971127
    import math

    q = [math.cos(rotation_z / 2), 0, 0, 1 * math.sin(rotation_z / 2)]

    return q


def generate_sample_annotation_json(total_annotations, data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating sample annotation json ---------------------")
    import numpy as np
    from pyquaternion import Quaternion

    sample_annotation_infos = []
    for sample_token in total_annotations.keys():
        annotations = total_annotations[sample_token]
        scene_token = sample_info_mappings[sample_token]['scene_token']
        frame_idx = sample_info_mappings[sample_token]['frame_idx']
        timestamp = sample_info_mappings[sample_token]['timestamp']

        for info in spd_infos:
            if info['token'] == sample_token:
                sample_lidar2ego_rotation = Quaternion(info['lidar2ego_rotation'])
                sample_lidar2ego_translation = np.array(info['lidar2ego_translation'])
                sample_ego2global_rotation = Quaternion(info['ego2global_rotation'])
                sample_ego2global_translation = np.array(info['ego2global_translation'])
                break

        for anno_token in annotations.keys():
            annotation = annotations[anno_token]

            # cvt global
            center = np.array([annotation['3d_location']['x'], annotation['3d_location']['y'],
                               annotation['3d_location']['z']])
            rot = Quaternion(axis=[0, 0, 1], radians=annotation['rotation'])

            # lidar2ego
            center = np.dot(sample_lidar2ego_rotation.rotation_matrix, center) + sample_lidar2ego_translation
            rot = sample_lidar2ego_rotation * rot

            # ego2global
            center = np.dot(sample_ego2global_rotation.rotation_matrix, center) + sample_ego2global_translation
            rot = sample_ego2global_rotation * rot

            info = {
                'token': annotation['token'],
                'sample_token': sample_token,
                'instance_token': annotation['instance_token'],
                'visibility_token': visibility_mappings[annotation['occluded_state']],
                'attribute_tokens': [],
                'translation': center.tolist(),
                'size': [annotation['3d_dimensions']['w'], annotation['3d_dimensions']['l'],
                         annotation['3d_dimensions']['h']],
                'rotation': rot.elements.tolist(),
                'prev': annotation['prev'],
                'next': annotation['next'],
                'num_lidar_pts': 100,
                'num_radar_pts': 0
            }

            sample_annotation_infos.append(info)

    target_file_path = osp.join(data_root, version, 'sample_annotation.json')
    write_json(sample_annotation_infos, target_file_path)


def generate_map_json(data_root, version='v1.0-mini', local_root=''):
    print("--------------------- Start generating map json ---------------------")

    log_data = load_json(osp.join(local_root, 'log.json'))
    log_mappings = {}
    for log in log_data:
        log_mappings[log['location']] = log['token']

    map_infos = []
    for location in log_mappings.keys():
        info = {
            'category': 'semantic_prior',
            'token': location,
            'filename': '',
            'log_tokens': [log_mappings[location]]
        }

        map_infos.append(info)

    target_file_path = osp.join(data_root, version, 'map.json')
    write_json(map_infos, target_file_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--data-root', type=str, default="./datasets/V2X-Seq-SPD-Example")
    parser.add_argument('--save-root', type=str, default="./datasets/V2X-Seq-SPD-Example")
    parser.add_argument('--split-file', type=str, default="./data/split_datas/cooperative-split-data-spd.json")
    parser.add_argument('--local-root', type=str, default="./tools/spd_data_converter/nuscenes_jsons")
    parser.add_argument('--v2x-side', type=str, default="vehicle-side")
    parser.add_argument('--version', type=str, default="v1.0-trainval")
    parser.add_argument('--info-prefix', type=str, default="spd")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    import os

    curDirectory = os.getcwd()
    basepath = os.path.basename(os.path.normpath(curDirectory))
    if basepath != 'UniV2X':
        os.chdir('UniV2X/')

    curDirectory = os.getcwd()
    print(curDirectory)

    v2x_side = args.v2x_side
    version = args.version
    data_root = args.data_root
    save_root = args.save_root
        
    local_root = args.local_root
    split_path = args.split_file
    can_bus_root_path = ''
    info_prefix = args.info_prefix

    if v2x_side == 'cooperative':
        print('this is cooperative!')
        total_annotations, sample_info_mappings, spd_infos = create_spd_infos_coop(data_root,
                            save_root,
                            v2x_side,
                            split_path, 
                            can_bus_root_path,
                            info_prefix,
                            version=version,
                            max_sweeps=10,
                            flag_save=False)  
    else:      
        print('this is single!')  
        total_annotations, sample_info_mappings, spd_infos = create_spd_infos(data_root,
                                                                            save_root,
                                                                            v2x_side,
                                                                            split_path,
                                                                            can_bus_root_path,
                                                                            info_prefix,
                                                                            version=version,
                                                                            max_sweeps=10,
                                                                            flag_save=False)

    save_root = osp.join(args.save_root, v2x_side)

    generate_category_json(save_root,
                           version=version,
                           local_root=local_root)

    generate_attribute_json(save_root,
                            version=version,
                            local_root=local_root)

    generate_visibility_json(save_root,
                             version=version,
                             local_root=local_root)

    generate_instance_json(total_annotations,
                           sample_info_mappings,
                           save_root,
                           version=version,
                           local_root=local_root)

    generate_sensor_json(save_root,
                         version=version,
                         local_root=local_root)

    generate_calibrated_sensor_json(spd_infos,
                                    save_root,
                                    version=version,
                                    local_root=local_root)

    generate_ego_pose_json(spd_infos,
                           save_root,
                           version=version,
                           local_root=local_root)

    generate_log_json(save_root,
                      version=version,
                      local_root=local_root)

    generate_scene_json(sample_info_mappings,
                        save_root,
                        version=version,
                        local_root=local_root)

    generate_sample_json(sample_info_mappings,
                         save_root,
                         version=version,
                         local_root=local_root)

    generate_sample_data_json(spd_infos,
                              save_root,
                              version=version,
                              local_root=local_root)

    generate_sample_annotation_json(total_annotations,
                                    save_root,
                                    version=version,
                                    local_root=local_root)

    generate_map_json(save_root,
                      version=version,
                      local_root=local_root)
