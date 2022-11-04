# ALIFE
This is the implementation of the paper "ALIFE: Adaptive Logit Regularizer and Feature Replay for Incremental Semantic Segmentation".

For detailed information, please check out our paper [[arXiv](https://arxiv.org/abs/2210.06816)].



## Requirements
* Python >= 3.6
* PyTorch >= 1.3.0
* yacs (https://github.com/rbgirshick/yacs)



## Getting started
```bash
git clone https://github.com/cvlab-yonsei/ALIFE.git
cd ALIFE

mkdir checkpoints logs
mkdir -p datasets/voc/19-1-ov datasets/voc/15-5-ov datasets/voc/15-5s-ov
mkdir -p datasets/ade/100-50-ov datasets/ade/50-ov datasets/ade/100-10-ov
```

### Datasets
The structure should be organized as follows:
```bash
├─ ALIFE
└─ data
    ├─ ADEChallengeData2016
    └─ VOCdevkit
```

### Training
#### Example commands
```Shell
bash scripts/voc/alife.sh   # RUN ALIFE on PASCAL VOC (overlapped setting)
bash scripts/ade/alife-m.sh # RUN ALIFE-M on ADE20K (overlapped setting)
```
NOTE: we also provide individual scripts for each step of ALIFE (e.g., base.sh, step1.sh, step2.sh, and step3.sh). You may need to modify those scripts for running a specific scenario (Please see [./scripts/](https://github.com/cvlab-yonsei/ALIFE/tree/main/scripts)). You can change config options by modifying the corresponding config files ([./configs/](https://github.com/cvlab-yonsei/ALIFE/tree/main/configs)) or typing 'KEY VALUE' pairs after '--opts' on the command line within the scripts.



## Acknowledgements
Our codes are partly based on the following repositories.
- [MiB](https://github.com/fcdl94/MiB)
- [PLOP](https://github.com/arthurdouillard/CVPR2021_PLOP)
- [Detectron2](https://github.com/facebookresearch/detectron2)
