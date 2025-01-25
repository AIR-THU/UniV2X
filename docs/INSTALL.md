# Installation
Environment installation is similar to [UniAD](https://github.com/OpenDriveLab/UniAD/blob/main/docs/INSTALL.md).

**a. Env: Create a conda virtual environment and activate it.**
```shell
conda create -n univ2x python=3.8 -y
conda activate univ2x
```

**b. Torch: Install PyTorch and torchvision.**
```shell
conda install cudatoolkit=11.1.1 -c conda-forge
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
```

**c. CUDA: Before installing MMCV family, you need to set up the CUDA_HOME (for compiling some operators on the gpu).**
```shell
export CUDA_HOME=YOUR_CUDA_PATH/
# Eg: export CUDA_HOME=/mnt/cuda-11.1/
```

**d. Install mmcv-full, mmdet and mmseg.**
```shell
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
cd YOUR_MMDET3D_DIR
# cd ..
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install scipy==1.7.3
pip install scikit-image==0.20.0
pip install -v -e .
```

**f. Install argoverse.**
```shell
cd YOUR_ARGOVERSE_DIR
# cd ..
git clone https://github.com/argoverse/argoverse-api.git
cd argoverse-api
pip install -e .
```

**g. Install other requirements.**
```shell
cd YOUR_UNIV2X_DIR
git clone https://github.com/AIR-THU/UniV2X.git
cd UniV2X
pip install -r requirements.txt
```