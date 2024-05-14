# DiSyn

## Disentangled Synthesis Transfer Network for Drug Response Prediction

This is a python implementation of Disentangled Synthesis Transfer Network for Drug Response Prediction which enhances generalizability by extracting drug response related features to synthesize new training samples.[paper]

![img](https://github.com/LiHongCSBLab/DiSyn/blob/5f97683b78b9eaf9ac21f47d87e02e3a3273a7e8/imgs/Overview.png)

## 1. Quick Start

### 1.1 Installation

**DiSyn** depends on PyTorch(>1.13), Numpy, scikit-learn, pandas.

#### Conda environment

Use the provided configuration file `environment.yaml` in `./codes` to create a conda required environment.
```commandline
cd codes/
conda env create -f environment.yaml
```
Running the command above will create environment `disyn`. To activate the disyn environment, use:
```commandline
conda activate disyn
```
### 1.2 Re-evaluation and inferrence
For re-evaluation or inferrence with your own data, please refer to the `inference.py` in `./codes` :
```commandline
cd codes/
python inference.py --drug=$drug
```
We have provided the model parameters with the highest metric(AUROC) obtained on GDSC and TCGA datasets.

| Drug           | SHA256                                                                                                                                                                                 |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 5-Fluorouracil | [23f73c2b9e15af1fdc03ee081084b47b928433df40ed8dc86ff1b231b684f13d](https://github.com/LiHongCSBLab/DiSyn/blob/8453b3dbd27ffb84e3e4aacd4261f13b3436e621/best_models/5-Fluorouracil.pt)  |
| Bicalutamide   | [534968be4b0fa70dedb82ea78cfd50272b83aecf2f8fc7c480a34fcd17d2bc4c](https://github.com/LiHongCSBLab/DiSyn/blob/93771c61e08a5d85530e2c33d949449bc0899a9f/best_models/Bicalutamide.pt)    |
| Bleomycin      | [5948efc4c3508943847c11d75d17ea3e09230433d89445660fa745ba377e2deb](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Bleomycin.pt)       |
| Cisplatin      | [02ad2b62dfd965fce5eaf80ed792ea1acebe61997cff17aa721d774f9d3dc2b8](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Cisplatin.pt)       |
| Docetaxel      | [b4c815b847f70fd97364551c0380f007fe45cfcb8cb2e661121118148375207d](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Docetaxel.pt)       |
| Doxorubicin    | [3b21e1a08f1fe3e61c83c0f1213e320a1d4d8e4f4b4bb8ada92f1634726a6888](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Doxorubicin.pt)     |
| Etoposide      | [9114ee71b5ef014a18d6e9f8bbd7bddd3f5360bbff13afb221e76f739a95716f](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Etoposide.pt)       |
| Gemcitabine    | [e6b658c88b0925ff27b236a5aaa4e028fa01a505a8d904979acbd74bb75584c1](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Gemcitabine.pt)     |
| Methotrexate   | [d058ddfc92dfeaf62cefe799bead5b45dae87492dc4d97026af2fdbecfdc16f7](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Methotrexate.pt)    |
| Paclitaxel     | [b0dabc57efc7fd349770e9eb627f16b7cb1079461093147870b18d1a3557e998](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Paclitaxel.pt)      |
| Pemetrexed     | [2bd588f30c75d24e55afa300307443950ebea958efe62f00fc0511179baff549](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Pemetrexed.pt)      |
| Sorafenib      | [b8d7dc953472538d87fb6ed8a3bd571fe49e148bb69e61157fca8ad000ec4651](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Sorafenib.pt)       |
| Tamoxifen      | [ac65f639b980f3bfb363176bc50745be515a18828b3e0d25715077a97f716f4e](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Tamoxifen.pt)       |
| Temozolomide   | [b65378082288d2ee9901a7aef54eff845e0b9be158ecc0cbce58deb132653480](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Temozolomide.pt)    |
| Vinblastine    | [a6a1ed7f9b32da3424e4a354ef2036e1d4b4370f7a2258b06a88c683eba70f13](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Vinblastine.pt)     |
| Vinorelbine    | [0f8dea963a249397d86d353025d58c63d1d3af19833c8648bbdea5e40c941750](https://github.com/LiHongCSBLab/DiSyn/blob/56faa78a02ae3b4f272e40c39c72b36ec4f836ce/best_models/Vinorelbine.pt)     |

## 2. Model Retrain
If you want to retrain the disyn models, refer to the following instructions.

### 2.1 Data
The pre-processed data we used for model training is accessible at `https://zenodo.org` </br>
For retraining, please unzip it and put it into ./data.

### 2.2 Model Retraining

##### 2.2.1 Model Pre-train
```commandline
python main_pretrain.py --nums_recon=$nums_recon --nums_critic=$nums_critic --drop_out=$drop_out
```
For parameter ranges we used in the paper work, please refer to the supplementary materials.

##### 2.2.2 Task-specific train
```commandline
python main_task_specific_train.py --drug_name=$drug --nums_recon=$nums_recon --nums_critic=$nums_critic --drop_out=$drop_out
```
where `$nums_recon`, `$nums_critic`, `$drop_out` are hyer-parameters of the pretrained models.

### iterative steps
Depending on the number of iteration steps, you might loop through these two steps.

##### 2.2.3 Disentanglement
```commandline
# The script we used to reconstruct the novel disentangled synthesis network.
python main_task_specific_train_step2_recon.py --drug_name=$drug --step=$step  --recon_epochs=$recon_epochs --clsadv_alpha=$clsadv_alpha --drop_out=$drop_out
```
`step` refers to the number of iterations, starts from `2`.

##### 2.2.4 Task-specific train in iterration
```commandline
python main_task_specific_train_step2_ft.py --drug_name=$drug --recon_epochs=$recon_epochs --clsadv_alpha=$clsadv_alpha --drop_out=$drop_out
```

## 3. Figures reproducing
For Figure reproducing, please refer to `./codes/fig2_reproduce.ipynb`

## 4. Others
Parts of implementations of this project come from
```commandline
https://github.com/XieResearchGroup/CODE-AE/blob/6dc17a5f3b7ce2e89736d1d575fb75951bd2c9ea/code/fine_tuning.py.
https://github.com/fungtion/DANN_py3/blob/master/functions.py
```
Thanks to the open source community contributors.

