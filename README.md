<div align="center">
<h1>Prototypical Calibrating Ambiguous Samples for Micro-Action Recognition</h1>

[**Kun Li**](https://scholar.google.com/citations?user=UQ_bInoAAAAJ), [**Dan Guo**](https://scholar.google.com/citations?user=DsEONuMAAAAJ), [**Guoliang Chen**](https://scholar.google.com/citations?user=TQ_oIrsAAAAJ), [**Chunxiao Fan**](https://scholar.google.com/citations?user=J-5cNxsAAAAJ), [**Jingyuan Xu**](https://scholar.google.com/citations?user=AT0tjn8AAAAJ), [**Zhiliang Wu**](), [**Hehe Fan**](https://scholar.google.com/citations?user=hVuflMQAAAAJ), [**Meng Wang**](https://scholar.google.com/citations?user=rHagaaIAAAAJ)


Hefei University of Technology, Zhejiang University

![arXiv](https://img.shields.io/badge/arXiv-2412.14719-b31b1b.svg?style=flat)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=kunli-cs.PCAN&left_color=green&right_color=red)
![GitHub issues](https://img.shields.io/github/issues-raw/kunli-cs/PCAN?color=%23FF9600)
![GitHub stars](https://img.shields.io/github/stars/kunli-cs/PCAN?style=flat&color=yellow)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prototypical-calibrating-ambiguous-samples/micro-action-recognition-on-ma-52)](https://paperswithcode.com/sota/micro-action-recognition-on-ma-52?p=prototypical-calibrating-ambiguous-samples)

</div>

## 🛠️ Installation

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/kunli-cs/PCAN.git
cd ./PCAN/mmaction2
pip install -v -e .
```

## Data Preparation

Download the MA-52 RGB data and Pose data.

[![RGB Data](https://img.shields.io/badge/HuggingFace-MA--52--RGB-blue?logo=huggingface&style=flat)](https://huggingface.co/datasets/kunli-cs/MA-52)
[![Skeleton Data](https://img.shields.io/badge/HuggingFace-MA--52--Skeleton--28kp-blue?logo=huggingface&style=flat)](https://huggingface.co/datasets/kunli-cs/MA-52_openpose_28kp)



```bash
pip install -U huggingface_hub
## use hf-mirror to accelerate
export HF_ENDPOINT=https://hf-mirror.com

## download RGB data
huggingface-cli download --repo-type dataset --resume-download kunli-cs/MA-52 --local-dir ./data/ma52
mkdir -p ./data/ma52/raw_videos && unzip ./data/ma52/train.zip -d ./data/ma52/raw_videos && rm ./data/ma52/train.zip
mkdir -p ./data/ma52/raw_videos && unzip ./data/ma52/val.zip -d ./data/ma52/raw_videos && rm ./data/ma52/val.zip
mkdir -p ./data/ma52/raw_videos && unzip ./data/ma52/test.zip -d ./data/ma52/raw_videos && rm ./data/ma52/test.zip

## download Pose data
huggingface-cli download --repo-type dataset --resume-download kunli-cs/MA-52_openpose_28kp --local-dir ./data/ma52/MA-52_openpose_28kp 
```

Download the pre-trained weights and checkpoint. 

[![HF Model](https://img.shields.io/badge/HuggingFace-PCAN--weights-blue?logo=huggingface&style=flat)](https://huggingface.co/kunli-cs/PCAN_weights/tree/main)

```bash
huggingface-cli download --repo-type dataset --resume-download kunli-cs/PCAN_weights --local-dir ./checkpoints 
```

## Training

Step 1. Pretraining 

We following the [RGBPoseConv3D](https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/posec3d/rgbpose_conv3d#step-1-pretraining) to pretraining PCAN.

You first need to train the RGB-only and Pose-only model on the MA-52 dataset, the pretrained checkpoints will be used to initialize the RGBPoseConv3D model. 

You can use the provided [IPython notebook](/mmaction2/configs/skeleton/posec3d/rgbpose_conv3d/merge_pretrain.ipynb) to merge two pretrained models into a single `rgbpose_conv3d_init.pth`.

You can do it your own or directly download and use the provided [rgbpose_conv3d_init.pth](https://huggingface.co/kunli-cs/PCAN_weights/blob/main/rgbpose_conv3d_init.pth).


Step 2. Training

```bash
python tools/train.py configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py 
```

## Evaluation

```bash
## export the result.pkl on the test set.
python tools/test.py ./configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py \
    pretrained/PCAN_checkpoint_7c4fba7c.pth --dump eval_ma52/result.pkl

## build the set set results `prediction.csv` in csv format.
python eval_ma52/eval_test.py
```
Please submit the test predictions `./eval_ma52/submission.zip` to the [Codabench evaluation server](https://www.codabench.org/competitions/9066/). 


## 📞 Contact Authors

If you have any questions or suggestions, please do not hesitate to contact [Kun Li](mailto:kunli.hfut@gmail.com).

## 🖊️ Citation

If you found this code useful, please consider cite:
```
@inproceedings{li2025prototypical,
  title={Prototypical calibrating ambiguous samples for micro-action recognition},
  author={Li, Kun and Guo, Dan and Chen, Guoliang and Fan, Chunxiao and Xu, Jingyuan and Wu, Zhiliang and Fan, Hehe and Wang, Meng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={4815--4823},
  year={2025}
}

@article{guo2024benchmarking,
  title={Benchmarking Micro-action Recognition: Dataset, Methods, and Applications},
  author={Guo, Dan and Li, Kun and Hu, Bin and Zhang, Yan and Wang, Meng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  volume={34},
  number={7},
  pages={6238-6252},
}

@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}

@misc{duan2021revisiting,
      title={Revisiting Skeleton-based Action Recognition},
      author={Haodong Duan and Yue Zhao and Kai Chen and Dian Shao and Dahua Lin and Bo Dai},
      year={2021},
      eprint={2104.13586},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🤝 Acknowledgement
This code began with [mmaction2](https://github.com/open-mmlab/mmaction2). We thank the developers for doing most of the heavy-lifting.
