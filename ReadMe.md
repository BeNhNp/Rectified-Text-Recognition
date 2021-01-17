# Rectified Text Recognition

The purpose of this project is to provide a set of easily modified character recognition model based on PyTorch framework.

At present, the code is still under adjustment, and it will take some time for the pre-training model to be released.

Basicly, I want the code to be clean and easy to read. Some utilities you can add by yourself.

This implementation ennable you to train large amout of imgae-text data on gpu.
The imgae transformation and the decoder is adapted to gpu too.
So the rate of gpu utilization is improved greatly.

Hope you can enjoy your training.

## Dataset

you can use the datasets from [aster.pytorch](https://github.com/ayumiymk/aster.pytorch)
 ([link](https://pan.baidu.com/s/1BMYb93u4gW_3GJdjBWSCSw) password: wi05).

or you can use the datasets from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
 ([link](https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ) password: rryk)(much more dataset is provided on google drive on this repo).

or you can create your own datasets as long as follow the same format.

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
  title={Scene Text Recognition With Linear Constrained Rectification},
  author={Gang, Wang and Huaping, Zhang and jianyun, Shang},
  booktitle={2020 International Conference on Computational Science and Computational Intelligence (CSCI)},
  year={2020},
  organization={IEEE}
}
```
