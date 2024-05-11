# DiSyn

## Disentangled Synthesis Transfer Network for Drug Response Prediction

-----------------------------------------------------------------

This is a python implementation of Disentangled Synthesis Transfer Network for Drug Response Prediction which enhances generalizability by extracting drug response related features to synthesize new training samples.[paper]

![img](https://github.com/LiHongCSBLab/DiSyn/blob/5f97683b78b9eaf9ac21f47d87e02e3a3273a7e8/imgs/Overview.png)

## 1. Installation

**DiSyn** depends on PyTorch(>1.13), Numpy, scikit-learn, pandas.

### Conda environment

Use the provided environment.yml in ./codes to create a conda required environment.
```commandline
cd codes/
conda env create -f environment.yaml
```
Running the command above will create environment `disyn`. To activate the disyn environment, use:
```commandline
conda activate disyn
```
