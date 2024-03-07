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

The datasets are too large, so we do not upload it to GitHub, but we offer links for you to look for or simly download them.

### Fitzpatrick 17k

[offical website](https://github.com/mattgroh/fitzpatrick17k)

[packaged dataset](https://notredame.box.com/s/pjf9kw5y1rtljnh81kuri4poecuiqngf)

[direct link](https://notredame.box.com/shared/static/pjf9kw5y1rtljnh81kuri4poecuiqngf.tar)

The following commands will make the data ready for default paths in the code:

```
wget https://notredame.box.com/shared/static/pjf9kw5y1rtljnh81kuri4poecuiqngf.tar -O fitzpatrick17k.tar
tar –xvf fitzpatrick17k.tar
```

### ISIC 2019

[offical website](https://challenge.isic-archive.com/landing/2019/)

[packaged dataset](https://notredame.box.com/s/uw8g5urs7m4n4ztxfo100kkga6arzi9k)

[direct link](https://notredame.box.com/shared/static/uw8g5urs7m4n4ztxfo100kkga6arzi9k.tar)

The following commands will make the data ready for default paths in the code:

```
wget https://notredame.box.com/shared/static/uw8g5urs7m4n4ztxfo100kkga6arzi9k.tar -O ISIC_2019_train.tar
tar –xvf ISIC_2019_train.tar
```

### CelebA

**Note**: so far, our paper have not used CelebA for any experiment yet.

[offical website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

[packaged dataset](https://notredame.box.com/s/s2ais65ldhzpm6wx4sej11ltecajbtqt)

[direct link](https://notredame.box.com/shared/static/s2ais65ldhzpm6wx4sej11ltecajbtqt.tar)

The following commands will make the data ready for default paths in the code:

```
wget https://notredame.box.com/shared/static/s2ais65ldhzpm6wx4sej11ltecajbtqt.tar -O img_align_celeba.tar
tar –xvf img_align_celeba.tar
```

## Usage

For most modules (e.g., `pre_train.py`), use `--help` or `-h` for usage information.

```
python pre_trained.py --help
```

`pre_trained.py` is to train a vanilla model on the selected dataset.

`quantize.py` applies quantization to a given model.

`test.py` and `test_group.py` test given models. `test.py` tests one model per time, while `test_group.py` tests all models in a given directory. If you have a bunch of models to test (e.g., output models from quantization), `test_group.py` would be faster than calling `test.py` for each model.

## Contact

If you have any question, feel free to submit issues and/or email me (Yuanbo Guo): yguo6 AT nd DOT edu. Thank you so much for your support!
