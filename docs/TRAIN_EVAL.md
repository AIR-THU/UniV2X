# Train/Eval Models

## Training
UniV2X training consists of four main stages. Pretrained and trained checkpoints for each stage—except for the final self-supervised learning stage—along with the corresponding model results, are listed below.

### Stage-one: Infrastructure and ego-vehicle sub-systems pretraining
In the first stage, we train the infrastructure sub-system and ego-vehicle sub-system separately to get pretrained model as the initialization for the stable UniV2X training. The infrastructure sub-system and ego-vehicle sub-system training are also following two-stage training pipeline as UniAD.

  - Download the pretrained bevformer weights
    ```
    mkdir ckpts && cd ckpts
    # Pretrained weights of bevformer
    wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
    ```
  
  - Infrastructure sub-system training
    ```
    # Stage 1
    CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_train.sh ./projects/configs_e2e_univ2x/univ2x_sub_inf_e2e_track.py ${GPU_NUM}

    # Stage 2
    CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_train.sh ./projects/configs_e2e_univ2x/univ2x_sub_inf_e2e.py ${GPU_NUM}
    ```

  - Vehicle sub-system training
    ```
    # Stage 1
    CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_train.sh ./projects/configs_e2e_univ2x/univ2x_sub_vehicle_e2e_track.py ${GPU_NUM}

    # Stage 2. You can also skip this step and directly use univ2x_sub_veh_stg1.pth as the pretrained of cooperative perception training.
    CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_train.sh ./projects/configs_e2e_univ2x/univ2x_sub_vehicle_e2e.py ${GPU_NUM}
    ```

### Stage-two: Cooperative perception training

In the second stage, we train the cooperative perception modules, including tracking and online mapping.
  ```
  CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_train.sh ./projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py ${GPU_NUM}
  # We suggest that you save the infrastructure agent queries of training part as the inference way, and train the agent fusion with the saved infrastructure agent queries.
  ```

### Stage-three: Cooperative planning training

In the third stage, we optimize all fusion modules and tasks together, including agent fusion, lane fusion and occupancy fusion, track, map, motion, occupancy occupancy and planning.
  ```
  CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_train.sh ./projects/configs_e2e_univ2x/univ2x_coop_e2e.py ${GPU_NUM}
  # We suggest that you save the infrastructure transmission data of training part as the inference way, and train the fusion moduels with the saved infrastructure transmission data.
  ```

### Stage-four: Self-supervised learning

To simplify the training process, here we directly skip the flow prediction module training.

### Trained Checkpoints 

We provide all trained checkpoints here.

- Infastrcture-only

| Model | Sub-inf-stage-1 | Sub-inf-stage-2 |
| :---: | :---: | :---: |
| Config | [univ2x_sub_inf_e2e_track.py](../projects/configs_e2e_univ2x/univ2x_sub_inf_e2e_track.py) | [univ2x_sub_inf_e2e.py](../projects/configs_e2e_univ2x/univ2x_sub_inf_e2e.py) |
| Checkpoints | [univ2x_sub_inf_stg1.pth](https://drive.google.com/file/d/1XJvMDmdasO-eHnLLQQgctU1x7FmPGOJR/view?usp=sharing)  | [univ2x_sub_inf_stg2.pth](https://drive.google.com/file/d/1ubZySia8smrlPbgTxVhAe3PyFIpoliYK/view?usp=sharing) |
|Checkpoints-md5 hash| f14ef0d540156cc9318399661fc08d5e | 7337567c6012f8b9fc326b66235d7c9b |

- Vehicle-only

| Model | Sub-vehicle-stage-1 | Sub-vehicle-stage-2 |
| :---: | :---: | :---: |
| Config | [univ2x_sub_vehicle_e2e_track.py](../projects/configs_e2e_univ2x/univ2x_sub_vehicle_e2e_track.py) | [univ2x_sub_vehicle_e2e.py](../projects/configs_e2e_univ2x/univ2x_sub_vehicle_e2e.py) |
| Checkpoints | [univ2x_sub_veh_stg1.pth](https://drive.google.com/file/d/1tEpnqKwTFgnz40oAr4lvPQvfSdU3b2s2/view?usp=sharing) | [univ2x_sub_veh_stg2.pth](https://drive.google.com/file/d/1kaU0_Vf_DpiLNh0r4h2ciKmQaAkkytWe/view?usp=sharing) |
|Checkpoints-md5 hash| 7ee07fc34dfac28070e640b16aebf26c | 2843db6bfabf4572ef621a486f5097e1 |

- Cooperation

| Model | Coop Perception | Coop Planning |
| :---: | :---: | :---: |
| Config | [univ2x_coop_e2e_track.py](../projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py) | [univ2x_coop_e2e.py](../projects/configs_e2e_univ2x/univ2x_coop_e2e.py) |
| Checkpoints | [univ2x_coop_e2e_stg1.pth](https://drive.google.com/file/d/1Ugm4fHZW8Tz0M-Gfcf1q4GWOGaLacb1a/view?usp=sharing) | [univ2x_coop_e2e_stg2.pth](https://drive.google.com/file/d/1V2vLqpjJencg2dZoGtwPb9UQQwsK74hN/view?usp=sharing) |
|Checkpoints-md5 hash | 66a8e1eace582bdaadf1fd0293dd9a5c | 8a08c5826059af32264025054b38f16e |


## Evaluation
Cooperative Evaluation
  ```
  CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_eval.sh ./projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py ./ckpts/univ2x_coop_e2e_stg1.pth ${GPU_NUM}
  ```
- Evaluation Results
  | Model | Tracking AMOTA | Mapping IoU-lane | Config | Download |
  | :---: | :---: | :---: | :---: | :---: |
  | Cooperative Perception | 0.300  | 16.2% | [univ2x_coop_e2e_track.py](../projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py) | [univ2x_coop_e2e_stg1.pth](https://drive.google.com/file/d/1Ugm4fHZW8Tz0M-Gfcf1q4GWOGaLacb1a/view?usp=sharing) |

Cooperative Planning
  ```
  CUDA_VISIBLE_DEVICES=${GPU_IDs} ./tools/univ2x_dist_eval.sh ./projects/configs_e2e_univ2x/univ2x_e2e.py ./ckpts/univ2x_coop_e2e_stg2.pth ${GPU_NUM}
  ```
- Evaluation Results

  | Model | Tracking AMOTA | Mapping IoU-lane | Occupancy IoU-n | Planning Col. | Config | Download | log |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | Cooperative Planning | 0.239 | 17.8%  | 22.6% | 0.54% |[univ2x_coop_e2e.py](../projects/configs_e2e_univ2x/univ2x_coop_e2e.py) | [univ2x_coop_e2e_stg2.pth](https://drive.google.com/file/d/1V2vLqpjJencg2dZoGtwPb9UQQwsK74hN/view?usp=sharing) | [eval_log](https://drive.google.com/file/d/1GsuXhDZKLWVhPueah38mD5Pqe5qfbn1C/view?usp=sharing) |

  Please note that the occupancy module was recently adjusted, resulting in differences compared to the results reported in the original paper. You can get the same results with following config and checkpoints.

  | Model | Tracking AMOTA | Mapping IoU-lane | Occupancy IoU-n | Planning Col. | Config | Checkpoint | log |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | Cooperative Planning | 0.239 | 17.8%  | 25.2% | 0.34% |[univ2x_coop_e2e_old_mode_inference_only.py](../projects/configs_e2e_univ2x/univ2x_coop_e2e_old_mode_inference_only.py) | [univ2x_coop_e2e_stg2_old_mode_inference_only.pth](https://drive.google.com/file/d/1Zu5pYkEms9q9n2ucMU6CYTWx3FfMVCpr/view?usp=sharing) | [eval_log](https://drive.google.com/file/d/1BJXvJwujNQQ8udRj5uyoqhsYO5pzmVr-/view?usp=sharing) |

### Visualization
After evaluation, you will get `./output/results.pkl` which is used for visualization.
  ```
  ./tools/univ2x_vis_results.sh
  ```