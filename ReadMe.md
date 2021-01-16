# Rectified Text Recognition

本项目旨在提供一套易于修改的文字识别模型，基于PyTorch框架。

The purpose of this project is to provide a set of easily modified character recognition model based on PyTorch framework.

目前还在调整中，预训练模型要等一段时间才能发布。

At present, the code is still under adjustment, and it will take some time for the pre-training model to be released.

## Dataset

you can use the datasets from [aster.pytorch](https://github.com/ayumiymk/aster.pytorch)
 ([link](https://pan.baidu.com/s/1BMYb93u4gW_3GJdjBWSCSw) password: wi05).

or you can use the datasets from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
 ([link](https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ) password: rryk)(much more dataset is provided on goole drive on this repo).

or you can create your own datasets only if follow the same format.

## Train

```
python train.py
```

## Test

```
python test.py
```

## Ref

[aster.pytorch](https://github.com/ayumiymk/aster.pytorch)

## Citation

```bibtex
@inproceedings{wang2020scene,
  title={Scene Text Recognition With Linear Constrained
Rectification},
  author={Gang, Wang and Huaping, Zhang and jianyun, Shang},
  booktitle={2020 International Conference on Computational Science and Computational Intelligence (CSCI)},
  year={2020},
  organization={IEEE}
}
```
