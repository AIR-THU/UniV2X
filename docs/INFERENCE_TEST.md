# Inference Pipeline without label 

## Step 1: Prepare V2X-Seq-SPD Testset

The Testset has been released at 18:00 (GMT+8), May 17, 2025.
Download Testset data and prepare it for inference.
The preprocess pipeline is similar to preprocessing trainval dataset.

You also needs these two pkls (spd_infos_temporal_test.pkl and spd_command_test.pkl) to run the inference, you can directly download them [here](https://drive.google.com/drive/folders/1qjpq0g9WB0sRQnKCRFQADYv7lybcnbwL) and modify the data path in config (univ2x_coop_e2e_old_mode_inference_wo_label.py).

## Step 2: Run Inference
Modify corresponding configs and enviroment configuration in tools/univ2x_inference.sh
  ```
  cd UniV2X
  bash tools/univ2x_inference.sh
  ```
You will get a pkl format result file, including detection, tracking and planning results.

Result file is organized as follows:

  ```
    {
        results: [
            {
                token: str
                # Tracking results
                boxes_3d: LiDARInstance3DBoxes
                scores_3d: Tensor, shape=(N,)
                labels_3d: Tensor, shape=(N,)
                track_scores: Tensor, shape=(N,)
                track_ids: Tensor, shape=(N,)
                # Detection results
                boxes_3d_det: LiDARInstance3DBoxes
                scores_3d_det: Tensor, shape=(M,)
                labels_3d_det: Tensor, shape=(M,)
                # Planning Results (5 seconds 2Hz traj points)
                planning_traj: Tensor, shape=(1, 10, 2)
            },
            {……},
            ……
        ]
    }
  ```

  The definition of LiDARInstance3DBoxes is in [mmdet3d](https://github.com/open-mmlab/mmdetection3d/blob/1.0/mmdet3d/core/bbox/structures/lidar_box3d.py).
