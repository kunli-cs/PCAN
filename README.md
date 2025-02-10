# PCAN
Official PyTorch implementation for the paper:

> **Prototypical Calibrating Ambiguous Samples for Micro-Action Recognition**, ***AAAI 2025***.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prototypical-calibrating-ambiguous-samples/micro-action-recognition-on-ma-52)](https://paperswithcode.com/sota/micro-action-recognition-on-ma-52?p=prototypical-calibrating-ambiguous-samples)


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
cd ./mmaction2
pip install -v -e .
```

## Training

TBD


## 🖊️ Citation

If you found this code useful, please consider cite:
```
@article{li2024prototypical,
  title={Prototypical Calibrating Ambiguous Samples for Micro-Action Recognition},
  author={Li, Kun and Guo, Dan and Chen, Guoliang and Fan, Chunxiao and Xu, Jingyuan and Wu, Zhiliang and Fan, Hehe and Wang, Meng},
  journal={arXiv preprint arXiv:2412.14719},
  year={2024}
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
```

## 🤝 Acknowledgement
This code began with [mmaction2](https://github.com/open-mmlab/mmaction2). We thank the developers for doing most of the heavy-lifting.
