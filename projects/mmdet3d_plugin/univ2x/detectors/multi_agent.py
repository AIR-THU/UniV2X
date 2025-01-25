#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
# Contact: yuhaibao94@gmail.com                                  #
#----------------------------------------------------------------#

import torch
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

class MultiAgent(MVXTwoStageDetector):
    def __init__(self, model_ego_agent, model_other_agents={}):
        super(MultiAgent, self).__init__()
        self.model_ego_agent = model_ego_agent

        self.other_agent_names = []
        self.pc_range_dict = {}
        for name_other_agent, model_other_agent in model_other_agents.items():
            setattr(self, name_other_agent, model_other_agent)
            self.other_agent_names.append(name_other_agent)
            if hasattr(model_other_agent, 'pc_range'):
                self.pc_range_dict[name_other_agent] = model_other_agent.pc_range


    def forward(self, ego_agent_data=None, other_agent_data_dict={}, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(ego_agent_data=ego_agent_data,
                                                                other_agent_data_dict=other_agent_data_dict,
                                                                return_loss=return_loss,
                                                                **kwargs)
        else:
            return self.forward_test(ego_agent_data=ego_agent_data,
                                                                other_agent_data_dict=other_agent_data_dict,
                                                                return_loss=return_loss,
                                                                **kwargs)

    def forward_train(self, ego_agent_data=None, other_agent_data_dict={}, return_loss=True, **kwargs):
        # UniV2X TODO: hardcode with 'img_metas'
        kwargs.pop('img_metas')

        # UniV2X TODO: unfrozen and update other-agent models
        with torch.no_grad():
            other_agent_results = {}
            for name_other_agent in self.other_agent_names:
                loss, univ2x_outs = getattr(self, name_other_agent)(
                        return_loss=return_loss,
                        **(other_agent_data_dict[name_other_agent]),
                        **kwargs
                )
                univ2x_outs['ego2other_rt'] = other_agent_data_dict[name_other_agent]['veh2inf_rt']
                univ2x_outs['pc_range'] = self.pc_range_dict[name_other_agent]
                other_agent_results[name_other_agent] = univ2x_outs

        loss, univ2x_outs = self.model_ego_agent(return_loss=return_loss, other_agent_results=other_agent_results, **ego_agent_data, **kwargs)

        return loss

    def forward_test(self, ego_agent_data=None, other_agent_data_dict={}, return_loss=False, **kwargs):
        other_agent_results = {}
        for name_other_agent in self.other_agent_names:
            other_agent_result = getattr(self, name_other_agent)(
                    return_loss=return_loss,
                    **(other_agent_data_dict[name_other_agent]),
                    **kwargs
            )
            other_agent_result[0]['ego2other_rt'] = other_agent_data_dict[name_other_agent]['veh2inf_rt']
            other_agent_result[0]['pc_range'] = self.pc_range_dict[name_other_agent]
            other_agent_results[name_other_agent] = other_agent_result

        result = self.model_ego_agent(return_loss=return_loss, other_agent_results=other_agent_results, **ego_agent_data, **kwargs)

        return result