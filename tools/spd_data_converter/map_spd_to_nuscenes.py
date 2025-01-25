#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
#----------------------------------------------------------------#

import os
import json
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

from shapely.geometry import Polygon, LineString, Point

from spd_to_uniad import load_json, write_json, gen_token

'''
nuscenes map format: https://www.nuscenes.org/tutorials/map_expansion_tutorial.html
'''

baidu_offset = [-40251.76572214719, 326531.9706723457]
dair_offset = [-456089, -4403869]

# yizhuang9: 'CITY_DRIVING', 'BIKING'
# boston-seaport: 'CAR'
lane_type_mappings = {
    'CITY_DRIVING': 'CAR',
    'LEFT_TURN_WAITING_ZONE': 'CAR',
    'EMERGENCY_LANE': 'CAR',
    'ROUNDABOUT': 'CAR',
    'BIKING': 'BIKING'
}


def map_spd_to_nuscenes_lane(spd_map_data, nuscenes_map_data):
    """Convert the lanes from SPD format to nuScenes format.
    Args:
        maps_root: SPD maps
    Return:
        lane maps: nuScenes Format
    """
    nuscenes_node_infos = []
    nuscenes_line_infos = []
    nuscenes_polygon_infos = []
    nuscenes_lane_infos = []

    for lane_id, lane in  tqdm(spd_map_data['LANE'].items()):
        nodes_token_mappings = {}
        token_node_mappings = {}

        lane_boundary = []
        for pt in lane['left_boundary']:
            xy = []
            pt_split = pt.split(', ')
            xy.append(float(pt_split[0][1:]))
            xy.append(float(pt_split[1][:-1]))
            lane_boundary.append(xy)
        
        for pt in lane['right_boundary'][::-1]:
            xy = []
            pt_split = pt.split(', ')
            xy.append(float(pt_split[0][1:]))
            xy.append(float(pt_split[1][:-1]))
            lane_boundary.append(xy)

        polygon_nodes = np.around(np.array(lane_boundary), 2)

        for ii in range(polygon_nodes.shape[0]):
            x, y = polygon_nodes[ii, 0], polygon_nodes[ii, 1]
            node_token = gen_token('node', str(x), str(y))
            node_info = {
                'token': node_token,
                'x': x,
                'y': y
            }
            nuscenes_node_infos.append(node_info)

            nodes_token_mappings[(lane_id, x, y)] = node_token
            token_node_mappings[node_token] = (lane_id, x, y)

        polygon_token = gen_token(lane_id, 'polygon')
        polygon_info = {
            'token': polygon_token,
            'exterior_node_tokens': [key for key in token_node_mappings.keys()],
            'holes': []
        }
        nuscenes_polygon_infos.append(polygon_info)

        x1, y1 = polygon_nodes[0, 0], polygon_nodes[0, 1]
        x2, y2 = polygon_nodes[1, 0], polygon_nodes[1, 1]
        line_token = gen_token(lane_id, str(x1), str(y1), str(x2), str(y2))
        line_info = {
            'token': line_token,
            'node_tokens': [nodes_token_mappings[(lane_id, x1, y1)],
                                            nodes_token_mappings[(lane_id, x2, y2)]]
        }
        nuscenes_line_infos.append(line_info)
        from_edge_line_token = line_token

        x1, y1 = polygon_nodes[-1, 0], polygon_nodes[-1, 1]
        x2, y2 = polygon_nodes[0, 0], polygon_nodes[0, 1]
        line_token = gen_token(lane_id, str(x1), str(y1), str(x2), str(y2))
        line_info = {
            'token': line_token,
            'node_tokens': [nodes_token_mappings[(lane_id, x1, y1)],
                                            nodes_token_mappings[(lane_id, x2, y2)]]
        }
        nuscenes_line_infos.append(line_info)
        to_edge_line_token = line_token

        lane_token = gen_token(lane_id, 'lane') # (lane_id, 'lane')
        lane_info = {
            'token': lane_token,
            'polygon_token': polygon_token,
            'lane_type': lane_type_mappings[lane['lane_type']],
            'from_edge_line_token': from_edge_line_token,
            'to_edge_line_token': to_edge_line_token,
            'left_lane_divider_segments': [],
            'right_lane_divider_segments': []
        }
        nuscenes_lane_infos.append(lane_info)

    nuscenes_map_data['node'] += nuscenes_node_infos
    nuscenes_map_data['line'] += nuscenes_line_infos
    nuscenes_map_data['polygon'] += nuscenes_polygon_infos
    nuscenes_map_data['lane'] += nuscenes_lane_infos

    return nuscenes_map_data


def map_spd_to_nuscenes_road_segment(spd_map_data, nuscenes_map_data):
    nuscenes_node_infos = []
    nuscenes_line_infos = []
    nuscenes_polygon_infos = []
    nuscenes_road_segment_infos = []

    for segment_id, segment in  tqdm(spd_map_data['JUNCTION'].items()):
        nodes_token_mappings = {}
        token_node_mappings = {}

        polygon_nodes = []
        for pt in segment['polygon']:
            xy = []
            pt_split = pt.split(', ')
            xy.append(float(pt_split[0][1:]))
            xy.append(float(pt_split[1][:-1]))
            polygon_nodes.append(xy)
        polygon_nodes = np.array(polygon_nodes) # [first_point, ..., end_point], different from nodes generated from centerline_to_polygon
        polygon_nodes = np.around(polygon_nodes, 2)

        for ii in range(polygon_nodes.shape[0]): # first point is not end point
            x, y = polygon_nodes[ii, 0], polygon_nodes[ii, 1]
            node_token = gen_token('node', str(x), str(y))
            node_info = {
                'token': node_token,
                'x': x,
                'y': y
            }
            nuscenes_node_infos.append(node_info)

            nodes_token_mappings[(segment_id, x, y)] = node_token
            token_node_mappings[node_token] = (segment_id, x, y)

        polygon_token = gen_token(segment_id, 'road_segment', 'polygon')
        polygon_info = {
            'token': polygon_token,
            'exterior_node_tokens': [key for key in token_node_mappings.keys()],
            'holes': []
        }
        nuscenes_polygon_infos.append(polygon_info)

        segment_token = gen_token(segment_id, 'road_segment')
        segment_info = {
            'token': segment_token,
            'polygon_token': polygon_token,
            'is_intersection': bool(1),
            'drivable_area_token': ''
        }
        nuscenes_road_segment_infos.append(segment_info)

    nuscenes_map_data['node'] += nuscenes_node_infos
    nuscenes_map_data['line'] += nuscenes_line_infos
    nuscenes_map_data['polygon'] += nuscenes_polygon_infos
    nuscenes_map_data['road_segment'] += nuscenes_road_segment_infos

    return nuscenes_map_data


def map_spd_to_nuscenes_ped_crossing(spd_map_data, nuscenes_map_data):
    nuscenes_node_infos = []
    nuscenes_line_infos = []
    nuscenes_polygon_infos = []
    nuscenes_ped_crossing_infos = []

    for segment_id, ped_crossing in  tqdm(spd_map_data['CROSSWALK'].items()):
        nodes_token_mappings = {}
        token_node_mappings = {}

        polygon_nodes = []
        for pt in ped_crossing['polygon']:
            xy = []
            pt_split = pt.split(', ')
            xy.append(float(pt_split[0][1:]))
            xy.append(float(pt_split[1][:-1]))
            polygon_nodes.append(xy)
        polygon_nodes = np.array(polygon_nodes) # [first_point, ..., end_point], different from nodes generated from centerline_to_polygon
        polygon_nodes = np.around(polygon_nodes, 2)

        for ii in range(polygon_nodes.shape[0]): # first point is not end point
            x, y = polygon_nodes[ii, 0], polygon_nodes[ii, 1]
            node_token = gen_token('node', str(x), str(y))
            node_info = {
                'token': node_token,
                'x': x,
                'y': y
            }
            nuscenes_node_infos.append(node_info)

            nodes_token_mappings[(segment_id, x, y)] = node_token
            token_node_mappings[node_token] = (segment_id, x, y)

        polygon_token = gen_token(segment_id, 'ped_crossing', 'polygon')
        polygon_info = {
            'token': polygon_token,
            'exterior_node_tokens': [key for key in token_node_mappings.keys()],
            'holes': []
        }
        nuscenes_polygon_infos.append(polygon_info)

        segment_token = gen_token(segment_id, 'ped_crossing') # (segment_id, 'road_segment')
        crossing_info = {
            'token': segment_token,
            'polygon_token': polygon_token,
            'road_segment_token': ''
        }
        nuscenes_ped_crossing_infos.append(crossing_info)

    nuscenes_map_data['node'] += nuscenes_node_infos
    nuscenes_map_data['line'] += nuscenes_line_infos
    nuscenes_map_data['polygon'] += nuscenes_polygon_infos
    nuscenes_map_data['ped_crossing'] += nuscenes_ped_crossing_infos

    return nuscenes_map_data


## We regard the union of {'lane', 'road_segment', 'ped_crossing'} as drivable area
def map_spd_to_nuscenes_drivable_area(spd_map_data, nuscenes_map_data):
    nuscenes_drivable_area_infos = []

    drivable_area_token = gen_token('000000', 'drivable_area')
    drivable_area_info = {
        'token': drivable_area_token,
        'polygon_tokens': []
    }

    for lane in nuscenes_map_data['lane']:
        polygon_token = lane['polygon_token']
        drivable_area_info['polygon_tokens'].append(polygon_token)

    for segment in nuscenes_map_data['road_segment']:
        polygon_token = segment['polygon_token']
        drivable_area_info['polygon_tokens'].append(polygon_token)

    for crossing in nuscenes_map_data['ped_crossing']:
        polygon_token = crossing['polygon_token']
        drivable_area_info['polygon_tokens'].append(polygon_token)

    nuscenes_drivable_area_infos.append(drivable_area_info)
    nuscenes_map_data['drivable_area'] += nuscenes_drivable_area_infos

    return nuscenes_map_data

def map_spd_to_nuscenes_canvas_edge(nuscenes_map_data):
    nuscenes_canvas_edge_infos = {
        'min_x': 10000000,
        'min_y': 10000000,
        'max_x': 0,
        'max_y': 0
    }

    for node in tqdm(nuscenes_map_data['node']):
        x, y = node['x'], node['y']
        if x < nuscenes_canvas_edge_infos['min_x']:
            nuscenes_canvas_edge_infos['min_x'] = x
        if x > nuscenes_canvas_edge_infos['max_x']:
            nuscenes_canvas_edge_infos['max_x'] = x
        if y < nuscenes_canvas_edge_infos['min_y']:
            nuscenes_canvas_edge_infos['min_y'] = y
        if y > nuscenes_canvas_edge_infos['max_y']:
            nuscenes_canvas_edge_infos['max_y'] = y

    nuscenes_map_data['canvas_edge'] = nuscenes_canvas_edge_infos

    return nuscenes_map_data


def map_spd_to_nuscenes(maps_root, save_root):
    nuscenes_map_keys = ['version', 'polygon', 'line', 'node', 
                                                'drivable_area', 'road_segment', 'road_block', 
                                                'lane', 'ped_crossing', 'walkway', 'stop_line', 
                                                'carpark_area', 'road_divider', 'lane_divider', 
                                                'traffic_light', 'canvas_edge', 'connectivity', 
                                                'arcline_path_3', 'lane_connector']
    
    ## Step 0: load maps and initialize nuscenes data
    location_names = []
    map_json_paths = os.listdir(maps_root)
    for map_path in map_json_paths:
        location_name = map_path.replace('.json', '')
        location_names.append(location_name)
    
    nuscenes_map_datas = {}
    for location_name in location_names:
        nuscenes_map_datas[location_name] = {}
        for map_key in nuscenes_map_keys:
            if map_key == 'version':
                nuscenes_map_datas[location_name][map_key] = '1.3'
            else:
                nuscenes_map_datas[location_name][map_key] = []

    for location_name in location_names:
        print("Location is: ", location_name)
        spd_map_data = load_json(os.path.join(maps_root, location_name+'.json'))
        nuscenes_map_data = nuscenes_map_datas[location_name]

        ## Step 1: generate lane elements
        nuscenes_map_data = map_spd_to_nuscenes_lane(spd_map_data, nuscenes_map_data)

        ## Step 2: generate road_segment elements
        nuscenes_map_data = map_spd_to_nuscenes_road_segment(spd_map_data, nuscenes_map_data)

        ## Step 3: generate ped_crossing elements
        nuscenes_map_data = map_spd_to_nuscenes_ped_crossing(spd_map_data, nuscenes_map_data)

        ## Step 4: generate drivable area elements
        nuscenes_map_data = map_spd_to_nuscenes_drivable_area(spd_map_data, nuscenes_map_data)
        
        ## Step N -1: get canvas_edge
        nuscenes_map_data = map_spd_to_nuscenes_canvas_edge(nuscenes_map_data)

        ## Step N: save map with nuscenes format
        # Remove duplicated nodes
        nuscenes_map_data_nodes = []
        nuscenes_map_data_node_xy = []
        for node in tqdm(nuscenes_map_data['node']):
            nuscenes_map_data_node_xy.append((node['x'], node['y']))
        nuscenes_map_data_node_xy= set(nuscenes_map_data_node_xy)

        for xy in tqdm(nuscenes_map_data_node_xy):
            x, y = xy[0], xy[1]
            node_token = gen_token('node', str(x), str(y))
            node_info = {
                'token': node_token,
                'x': x,
                'y': y
            }
            nuscenes_map_data_nodes.append(node_info)
        nuscenes_map_data['node'] = nuscenes_map_data_nodes

        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_map_path = os.path.join(save_root, location_name+'.json')
        write_json(nuscenes_map_data, save_map_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--maps-root', type=str, default="./datasets/V2X-Seq-SPD-Example/maps")
    parser.add_argument('--save-root', type=str, default="./datasets/V2X-Seq-SPD-Example")
    parser.add_argument('--v2x-side', type=str, default="vehicle-side")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    maps_root = args.maps_root
    v2x_side = args.v2x_side
    save_root = os.path.join(args.save_root, v2x_side, 'maps/expansion')

    map_spd_to_nuscenes(
        maps_root=maps_root,
        save_root=save_root
    )