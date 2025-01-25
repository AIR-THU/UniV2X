import mmcv
import json
import os.path as osp
import numpy as np
from scipy.linalg import polar
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import argparse

def iterative_closest_point(A, num_iterations=100):
    R = A.copy()

    for _ in range(num_iterations):
        U, _ = polar(R)
        R = U

    return R

def read_json(path_json):
    with open(path_json, 'r') as load_f:
        data_json = json.load(load_f)
    return data_json

def inf2veh_convert(result, spd_data_root, pair_info_file, out_path):
    result = mmcv.load(result)['bbox_results']
    pair_infos = read_json(pair_info_file)
    
    inf2veh = {}
    for pair in pair_infos:
        inf2veh[pair['infrastructure_frame']] = pair['vehicle_frame']
        inf2veh[pair['infrastructure_frame']+'offset'] = pair['system_error_offset']

    for det in mmcv.track_iter_progress(result):
        inf_id = det['token']
        veh_id = inf2veh[inf_id]
        det['token'] = veh_id

        inf_virtuallidar2world_path = osp.join(spd_data_root, 'infrastructure-side/calib/virtuallidar_to_world', inf_id+'.json')
        inf_virtuallidar2world = read_json(inf_virtuallidar2world_path)
        inf_virtuallidar2world_rotation = inf_virtuallidar2world['rotation']
        inf_virtuallidar2world_translation = inf_virtuallidar2world['translation']

        veh_ego2world_path = osp.join(spd_data_root, 'vehicle-side/calib/novatel_to_world', veh_id+'.json')
        veh_ego2world = read_json(veh_ego2world_path)
        veh_ego2world_rotation = veh_ego2world['rotation']
        veh_ego2world_translation = veh_ego2world['translation']

        veh_lidar2ego_path = osp.join(spd_data_root, 'vehicle-side/calib/lidar_to_novatel', veh_id+'.json')
        veh_lidar2ego = read_json(veh_lidar2ego_path)
        veh_lidar2ego_rotation = veh_lidar2ego['transform']['rotation']
        veh_lidar2ego_translation = veh_lidar2ego['transform']['translation']

        err_offset = inf2veh[inf_id+'offset']


        veh_l2e_r = np.array(veh_lidar2ego_rotation)
        veh_l2e_t = np.array(veh_lidar2ego_translation).reshape(3)
        veh_e2g_r = np.array(veh_ego2world_rotation)
        veh_e2g_t = np.array(veh_ego2world_translation).reshape(3)

        inf_e2g_r = np.array(inf_virtuallidar2world_rotation)
        inf_e2g_t = np.array(inf_virtuallidar2world_translation).reshape(3)
        inf_l2e_r = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        inf_l2e_t = np.array([0, 0, 0]).reshape(3)

        err_offset = np.array([err_offset['delta_x'], err_offset['delta_y'],0])

        r = ((veh_l2e_r.T @ veh_e2g_r.T) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)).T
        t = (-err_offset @ veh_l2e_r.T @ veh_e2g_r.T + veh_l2e_t @ veh_e2g_r.T + veh_e2g_t) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)
        t -= inf_e2g_t @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T) +\
                inf_l2e_t @ (np.linalg.inv(inf_l2e_r).T)
        vehlidar2inflidar_rotation, vehlidar2inflidar_translation = r, t
        
        appro_vehlidar2inflidar = iterative_closest_point(vehlidar2inflidar_rotation)
        inflidar2vehlidar_rotation = np.linalg.inv(appro_vehlidar2inflidar)
        inflidar2vehlidar_translation = - np.dot(inflidar2vehlidar_rotation, vehlidar2inflidar_translation)

        boxes_3d = det['boxes_3d']
        boxes_3d_det = det['boxes_3d_det']

        boxes_3d.rotate(inflidar2vehlidar_rotation.T)
        boxes_3d.translate(inflidar2vehlidar_translation)
        boxes_3d_det.rotate(inflidar2vehlidar_rotation.T)
        boxes_3d_det.translate(inflidar2vehlidar_translation)

        det['boxes_3d'] = boxes_3d
        det['boxes_3d_det'] = boxes_3d_det

    outputs = {'bbox_results': result}
    mmcv.dump(outputs, out_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default='./output/stage1_inf_0_100.pkl', help='path to results.pkl')
    parser.add_argument('--data_root', default='datasets/V2X-Seq-SPD-New', help='data root')
    parser.add_argument('--output_result_path', default='./output/stage1_inf_0_100_inf2veh.pkl', help='path to converted results.pkl')
    args = parser.parse_args()

    result_path = args.result_path
    spd_data_root = args.data_root
    pair_info_file = osp.join(spd_data_root, 'cooperative/data_info.json')
    output_result_path = args.output_result_path

    inf2veh_convert(result_path, spd_data_root, pair_info_file, output_result_path)
