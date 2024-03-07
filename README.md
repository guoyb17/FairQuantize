# FairQuantize

Repo for MICCAI24 paper **FairQuantize: Achieving Fairness Through Weight Quantization for Dermatological Disease Diagnosis**.

## Dependency

The repo has been built and run on Python 3.9.18.

First, install PyTorch and related packages. See [PyTorch website](https://pytorch.org/) for installation instructions that work for your machine.

Then, install the following packages (the command is for conda; modify according to your environment):

```
conda install pandas bokeh tqdm scikit-learn scikit-image
```

```
pip install torch-pruning backpack-for-pytorch
```

Also, install `inq` from [INQ-pytorch](https://github.com/Mxbonn/INQ-pytorch), which offers its own installation methods.

## Dataset

The datasets are too large, so we do not upload it to GitHub, but we offer links for you to look for them.

### Fitzpatrick 17k

[offical website](https://github.com/mattgroh/fitzpatrick17k)

### ISIC 2019

[offical website](https://challenge.isic-archive.com/landing/2019/)

### CelebA

**Note**: so far, our paper has not used CelebA for any experiment yet.

[offical website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Usage

For most modules (e.g., `pre_train.py`), use `--help` or `-h` for usage information.

```
python pre_trained.py --help
```

`pre_trained.py` is to train a vanilla model on the selected dataset.

`quantize.py` applies quantization to a given model.

`test.py` and `test_group.py` test given models. `test.py` tests one model per time, while `test_group.py` tests all models in a given directory. If you have a bunch of models to test (e.g., output models from quantization), `test_group.py` would be faster than calling `test.py` for each model.

## Contact

If you have any question, feel free to submit issues. Thank you so much for your support!
