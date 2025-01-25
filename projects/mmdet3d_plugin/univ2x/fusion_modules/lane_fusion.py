#-------------------------------------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                                      #
# Copyright (c) DAIR-V2X. All rights reserved.                                                    #
#-------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class LaneQueryFusion(nn.Module):
    def __init__(self, pc_range, embed_dims=256):
        super(LaneQueryFusion, self).__init__()

        self.pc_range = pc_range
        self.embed_dims = embed_dims

        # reference_points ---> pos_embed
        self.get_pos_embedding = nn.Linear(3, self.embed_dims)
        # cross-agent feature alignment
        self.cross_agent_align = nn.Linear(self.embed_dims+9, self.embed_dims)
        self.cross_agent_align_pos = nn.Linear(self.embed_dims+9, self.embed_dims)
        self.cross_agent_fusion = nn.Linear(self.embed_dims, self.embed_dims)

        # parameter initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _lidar2norm(self, locs, pc_range, norm_mode='sigmoid'):
        """
        absolute (x,y,z) in global coordinate system ---> normalized (x,y,z)
        """
        from mmdet.models.utils.transformer import inverse_sigmoid

        if norm_mode not in ['sigmoid', 'inverse_sigmoid']:
            raise Exception('mode is not correct with {}'.format(norm_mode))

        locs[..., 0:1] = (locs[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        if norm_mode == 'inverse_sigmoid':
            locs = inverse_sigmoid(locs)

        return locs
    
    def _norm2lidar(self, ref_pts, pc_range, norm_mode='sigmoid'):
        """
        normalized (x,y) ---> absolute (x,y) in inf lidar coordinate system
        """
        if norm_mode not in ['sigmoid', 'inverse_sigmoid']:
            raise Exception('mode is not correct with {}'.format(norm_mode))
        if norm_mode == 'inverse_sigmoid':
            locs = ref_pts.sigmoid().clone()
        else:
            locs = ref_pts.clone()

        locs[:, 0:1] = (locs[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])

        return locs


    def filter_other_lanes(self, cls_score, threshold=0.05, num_things_classes=3):
        '''
        refer to _get_bboxes_single in panseg_head.py
        cls_score = other_outputs_classes[-1, bs]
        '''
        cls_score = cls_score.sigmoid()
        indexes = list(torch.where(cls_score.view(-1) > threshold))[0]
        det_labels = indexes % num_things_classes
        bbox_index = indexes // num_things_classes
        
        return bbox_index

    def forward(self, other_outputs_classes, other_outputs_coords, other_query, other_query_pos, other_reference,
                                    veh_outputs_classes, veh_outputs_coords, veh_query, veh_query_pos, veh_reference,
                                    ego2other_rt, other_agent_pc_range, threshold=0.05):
        '''
        reference: (x, y, w, h), reference = inverse_sigmoid(reference)
        outputs_coords: (x, y, w, h), outputs_coords = outputs_coords.sigmoid()
        '''
        calib_other2ego = np.linalg.inv(ego2other_rt[0].cpu().numpy().T)
        calib_other2ego = torch.tensor(calib_other2ego).to(other_query)

        # UniV2X TODO: hardcode for filtering inf queries with scores
        # UniV2X TODO: supposed that img num = 1
        other_cls_scores = other_outputs_classes[-1]
        other_bbox_index = self.filter_other_lanes(other_cls_scores, threshold=threshold)
        other_outputs_classes = other_outputs_classes[:, :, other_bbox_index, :]
        other_outputs_coords = other_outputs_coords[:, :, other_bbox_index, :]
        other_query = other_query[:, other_bbox_index, :]
        other_query_pos = other_query_pos[:, other_bbox_index, :]
        other_reference = other_reference[:, other_bbox_index, :]

        # other_reference: other2ego
        other_ref_pts = torch.zeros(other_reference.shape[0],
                                                            other_reference.shape[1],
                                                            3).to(other_query)
        other_ref_pts[..., :2] = other_reference[..., :2]
        for ii in range(other_ref_pts.shape[0]):
            other_ref_pts[ii] = self._norm2lidar(other_ref_pts[ii], other_agent_pc_range, norm_mode='inverse_sigmoid')
            other_ref_tmp = torch.cat((other_ref_pts[ii], torch.ones_like(other_ref_pts[ii][..., :1])), -1).unsqueeze(-1)
            other_ref_pts[ii] = torch.matmul(calib_other2ego, other_ref_tmp).squeeze(-1)[..., :3]

            other_ref_pts[ii] = self._lidar2norm(other_ref_pts[ii], self.pc_range, norm_mode='inverse_sigmoid')
        other_reference[..., :2] = other_ref_pts[..., :2]

        # other_bboxes: other2ego
        other_bboxes = torch.zeros(other_outputs_coords.shape[0],
                                                        other_outputs_coords.shape[1],
                                                        other_outputs_coords.shape[2],
                                                        3).to(other_query)
        other_bboxes[..., :2] = other_outputs_coords[..., :2]
        for ii in range(other_bboxes.shape[0]):
            for jj in range(other_bboxes.shape[1]):
                other_bboxes[ii, jj] = self._norm2lidar(other_bboxes[ii, jj], other_agent_pc_range, norm_mode='sigmoid')
                other_ref_tmp = torch.cat((other_bboxes[ii, jj], torch.ones_like(other_bboxes[ii, jj][..., :1])), -1).unsqueeze(-1)
                other_bboxes[ii, jj] = torch.matmul(calib_other2ego, other_ref_tmp).squeeze(-1)[..., :3]

                other_bboxes[ii, jj] = self._lidar2norm(other_bboxes[ii, jj], self.pc_range, norm_mode='sigmoid')
        other_outputs_coords[..., :2] = other_bboxes[..., :2]

        # cross-agent feature alignment
        for ii in range(other_query.shape[0]):
            inf2veh_r = calib_other2ego[:3,:3].reshape(1,9).repeat(other_query[ii].shape[0], 1)
            other_query[ii] = self.cross_agent_align(torch.cat([other_query[ii], inf2veh_r], -1))
            other_query_pos[ii] = self.cross_agent_align_pos(torch.cat([other_query_pos[ii], inf2veh_r], -1))

        # UniV2X TODO: directly concat other-agent queries and veh queries
        # UniV2X TODO: supposed that img num = 1
        other_outputs_classes = torch.cat((veh_outputs_classes, other_outputs_classes), dim=2)
        other_outputs_coords = torch.cat((veh_outputs_coords, other_outputs_coords), dim=2)
        other_query = torch.cat((veh_query, other_query), dim=1)
        other_query_pos = torch.cat((veh_query_pos, other_query_pos), dim=1)
        other_reference = torch.cat((veh_reference, other_reference), dim=1)

        return other_outputs_classes, other_outputs_coords, other_query, other_query_pos, other_reference