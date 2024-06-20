# [Transformers4NewsRec](https://github.com/Ruixinhua/transformers4newsrec)

## Introduction

This repository contains the code for the implementations of neural news recommendation models based on the [transformers](https://huggingface.co/transformers/) library. The implementation includes the following models:
- Neural Recommendation with Multi-Head Self-Attention (NRMS)
- Neural Recommendation with Personalized Attention (NPA)
- Neural Recommendation with Attentive Multi-View Learning (NAML)
- Neural Recommendation with Long- and Short-term User Representations (LSTUR)

## Get Started
Clone the repository and install the dependencies.
```bash
git clone https://github.com/Ruixinhua/transformers4newsrec.git
cd transformers4newsrec
# activate your virtual environment
source activate your_env
pip install -r requirements.txt

# install torch from the website: https://pytorch.org/ (check your nvidia-driver version, we suggest to use cuda version of 11.8 or 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
Ensure that you have login to [wandb](https://wandb.ai/site) to log the training process.
```bash
wandb login
# paste your API key when prompted
```
To run the code for training the baselines and evaluating its performance, follow these commands:
```bash
export PYTHONPATH=PYTHONPATH:./:./newsrec  # set current directory and the module directory as PYTHONPATH
python newsrec/trainer/nrs_trainer.py model_name=NRMSRSModel running_mode=train_only  output_dir=./output_dir
# model_name: NRMSRSModel, NPARSModel, NAMLRSModel, LSTURRSModel
```
Check description of arguments by [default_config](https://github.com/Ruixinhua/transformers4newsrec/blob/master/newsrec/config/default_config.py) under `newsrec/config` directory.