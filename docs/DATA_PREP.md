
# V2X-Seq-SPD Data Preparation

We use [V2X-Seq-SPD](https://drive.google.com/drive/folders/1gnrw5llXAIxuB9sEKKCm6xTaJ5HQAw2e?usp=sharing) dataset, which is the first real-world sequential V2X perception dataset. You can use the following scripts to perform data processing.

## Step 1: Download V2X-Seq-SPD
Download V2X-Seq-SPD and organize these files as follows:
```
V2X-Seq-SPD
├── cooperative/
│   ├── label/
│   ├── data_info.json
├── infrastructure-side/
│   ├── velodyne/
│   ├── image/
│   ├── calib/
│   ├── label/
│   ├── data_info.json
├── maps/
│   ├── yizhuang02.json
│   ├── yizhuangxx.json (and other map files)
├── vehicle-side/
│   ├── velodyne/
│   ├── image/
│   ├── calib/
│   ├── label/
│   ├── data_info.json
```

## Step 2: Create new V2X-Seq-SPD
Create new V2X-Seq-SPD for following usage.

```
python tools/spd_data_converter/gen_example_data.py
    --input YOUR_V2X-Seq-SPD_ROOT \
    --output ./datasets/V2X-Seq-SPD-New \
    --sequences 'all' \ # You can use '--sequences 0010 0016 0018 0022 0023 0025 0029 0030 0032 0033 0034 0035 0014 0015 0017 0020 0021' to create part of new V2X-Seq-SPD for the fast test.
    --update-label \
    --freq 2
```

## Step 3: Convert the new V2X-Seq-SPD dataset into UniV2X format

```
bash ./tools/spd_data_converter/spd_dataset_converter.sh V2X-Seq-SPD-New vehicle-side
bash ./tools/spd_data_converter/spd_dataset_converter.sh V2X-Seq-SPD-New infrastructure-side
bash ./tools/spd_data_converter/spd_dataset_converter.sh V2X-Seq-SPD-New cooperative
```


## Step 4: Prepare pretrained and trained weights
We use [BEVFormer](https://github.com/fundamentalvision/BEVFormer) as pretrained weight. 
```
mkdir ckpts && cd ckpts
# Pretrained weights of bevformer
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
```
And we provide our trained models for quick evaluation. Please dowload the weights from [Train/Eval](docs/TRAIN_EVAL.md).


## Overall Structure
After the above steps, you will see the overall structure.
```
UniV2X
├── projects/
├── tools/
├── ckpts/
│   ├── bevformer_r101_dcn_24ep.pth
│   ├── xxx.pth # You want to download
├── datasets/
│   ├── V2X-Seq-SPD-New
│   │   ├── vehicle-side
│   │   ├── infrastructure-side
│   │   ├── cooperative
├── data/
│   ├── infos/
│   │   ├── V2X-Seq-SPD-New
│   │   │   ├── vehicle-side
│   │   │   │   ├── spd_infos_temporal_train.pkl
│   │   │   │   ├── spd_infos_temporal_val.pkl
│   │   │   ├── infrastructure-side
│   │   │   ├── cooperative
│   ├── split_datas/
│   │   ├── cooperative-split-data-spd.json
```