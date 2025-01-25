#!/bin/bash
export PYTHONPATH=$PYTHONPATH:./
python ./tools/analysis_tools/visualize/univ2x_run.py \
    --predroot ./output/vehicle_results.pkl \
    --out_folder ./output_visualize \
    --demo_video test_demo.avi \
    --project_to_cam 0 \
    --dataroot datasets/V2X-Seq-SPD-New/cooperative