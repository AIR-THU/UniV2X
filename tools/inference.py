import argparse
import os
import warnings
from mmcv import Config, DictAction
import mmcv
import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.apis.test import custom_multi_gpu_test_wo_label, custom_multi_gpu_test
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent
from mmdet.apis import set_random_seed

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='3D Detection Inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl', required=True, help='output result file path')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='pytorch')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--deterministic', action='store_true')
    return parser.parse_args()

def build_models(cfg):
    # Build other agents
    other_agents = {}
    for key in cfg.keys():
        if 'model_other_agent' in key:
            agent_cfg = cfg.get(key)
            agent_cfg.train_cfg = None
            model = build_model(agent_cfg, test_cfg=cfg.get('test_cfg'))
            if agent_cfg.load_from:
                load_checkpoint(model, agent_cfg.load_from, map_location='cpu', 
                              revise_keys=[(r'^model_ego_agent\.', '')])
            other_agents[key] = model
    
    # Build ego agent
    cfg.model_ego_agent.train_cfg = None
    ego_agent = build_model(cfg.model_ego_agent, test_cfg=cfg.get('test_cfg'))
    if cfg.model_ego_agent.load_from:
        load_checkpoint(ego_agent, cfg.model_ego_agent.load_from, 
                      map_location='cpu', revise_keys=[(r'^model_ego_agent\.', '')])
    
    return MultiAgent(ego_agent, other_agents)

def main():
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    
    # Handle plugins and custom imports
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # Init distributed
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
    
    # Set random seed
    set_random_seed(args.seed, deterministic=args.deterministic)
    
    # Build dataset
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    
    # Build model
    model = build_models(cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # FP16 support
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg:
        wrap_fp16_model(model)
    
    # Class info handling
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.model_ego_agent.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.model_ego_agent.CLASSES = dataset.CLASSES
    
    # Distributed inference
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False
        )
        if dataset.inference_wo_label:
            outputs = custom_multi_gpu_test_wo_label(model, data_loader, tmpdir=None, gpu_collect=False)
        else:
            outputs = custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False)
    else:
        raise NotImplementedError("Non-distributed inference not supported")
    
    # Save results
    rank, _ = get_dist_info()
    if rank == 0:
        print(f'Saving results to {args.out}')
        mmcv.dump(outputs, args.out)

if __name__ == '__main__':
    main()
