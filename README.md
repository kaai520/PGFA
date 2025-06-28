# Zero-shot Skeleton-based Action Recognition with Prototype-guided Feature Alignment [TIP 2025]
> [Kai Zhou](https://scholar.google.com/citations?user=58UyQ9cAAAAJ&hl=zh-CN&oi=ao), [Shuhai Zhang](https://scholar.google.com/citations?user=oNhLYoEAAAAJ&hl=zh-CN), [Zeng You](https://scholar.google.com/citations?user=7xCkJ-QAAAAJ&hl=zh-CN), [Jinwu Hu](https://scholar.google.com/citations?user=XmqjPi0AAAAJ&hl=en), [Mingkui Tan](https://tanmingkui.github.io/), and [Fei Liu](https://scholar.google.com/citations?user=gC-YMYgAAAAJ&hl=en)\
South China University of Technology

This is an official PyTorch implementation of **"Zero-shot Skeleton-based Action Recognition with Prototype-guided Feature Alignment" in IEEE TIP 2025**.
<!-- [[Paper]](https://arxiv.org/abs/2308.03950) -->

# Abstract
Zero-shot skeleton-based action recognition aims to classify unseen skeleton-based human actions without prior exposure to such categories during training. This task is extremely challenging due to the difficulty in generalizing from known to unknown actions. Previous studies typically use two-stage training: pre-training skeleton encoders on seen action categories using cross-entropy loss and then aligning pre-extracted skeleton and text features, enabling knowledge transfer to unseen classes through skeleton-text alignment and language models' generalization.
However, their efficacy is hindered by 1) insufficient discrimination for skeleton features, as the fixed skeleton encoder fails to capture necessary alignment information for effective skeleton-text alignment; 2) the neglect of alignment bias between skeleton and unseen text features during testing. 
To this end, we propose a prototype-guided feature alignment paradigm for zero-shot skeleton-based action recognition, termed PGFA.
Specifically, we develop an end-to-end cross-modal contrastive training framework to improve skeleton-text alignment, ensuring sufficient discrimination for skeleton features. Additionally, we introduce a prototype-guided text feature alignment strategy to mitigate the adverse impact of the distribution discrepancy during testing.
We provide a theoretical analysis to support our prototype-guided text feature alignment strategy and empirically evaluate our overall PGFA on three well-known datasets.
Compared with the top competitor SMIE method, our PGFA achieves absolute accuracy improvements of 22.96\%, 12.53\%, and 18.54\% on the NTU-60, NTU-120, and PKU-MMD datasets, respectively.

# Framework
## Training Framework
![traing](./assets/training.png)
## Testing Framework
![testing](./assets/testing.png)
<!-- ![prototype](./assets/prototype.png){:style="width:50%"} -->
<img src="./assets/prototype.png" alt="prototype" width="60%" />

## Requirements
![python = 3.11](https://img.shields.io/badge/python-3.7.11-green)
```
sacred
tqdm
einops
torch==1.13.1
logging
sentence-transformers
pprint
scikit-learn
```

## Installation
```bash
# Install the python libraries
$ cd SMIE
$ pip install -r requirements.txt

# Install the ShiftGCN
$ cd ./module/Temporal_shift
$ bash run.sh
```

Please consult the official installation tutorial (e.g., [ShiftGCN](https://github.com/kchengiva/Shift-GCN) and [PyTorch](https://pytorch.org/get-started/previous-versions/)) if you experience any difficulties.

## Data Preparation
We apply the same dataset processing as [SMIE](https://github.com/YujieOuO/SMIE). You can download in BaiduYun link [data.zip](https://pan.baidu.com/s/1G8q_0fhLGIlNrCt4Oy0pkg). Please download and extract it to the current folder (PGFA). 

The code: pgfa

The subfolder "zero-shot" of "data" contains the processed skeleton data for each dataset, already split into seen and unseen categories. The subfolder "language" contains the pre-extracted text features obtained using Sentence-Bert.

* [dataset]_embeddings.npy: based on label names using Sentence-Bert.
* [dataset]_des_embeddings.npy: based on complete descriptions using Sentence-Bert.

If you don't want to download data.zip using BaiduYun link, please contact kayjoe0723@gmail.com. If you want to process the data by yourself, please refer to the Data Preparation section in [SMIE](https://github.com/YujieOuO/SMIE).

### Action Label Descriptions
The total label descriptions can be found in ./descriptions.

## Different Experiment Settings
Our PGFA employs two experiment setting.
* Setting 1 : three datasets are used (NTU-60, NTU-120, PKU-MMD), and each dataset have three random splits. The skeleton feature extractor is classical ST-GCN.
* Setting 2: two datasets are used, split_5 and split_12 on NTU60, and split_10 and split_24 on NTU120. The skelelton feature extractor is Shift-GCN. 

### Setting 1

#### Training & Testing
Example for training and testing on NTU-60 split_1.  
```bash
# Setting 1
$ python main.py with 'train_mode="main"'
```
You can change some settings of config.py.  

### Setting 2
#### Training & Testing
Example for training and testing on NTU-60 split_5 data.
```bash
# Setting 2
$ python main.py with 'train_mode="sota"'
```
You can also choose different split id of config.py (sota compare part).  

## Todo List
1. Upload checkpoints.
2. Upload skeleton-focused descriptions and text features.
3. One-shot experiments.

## Acknowledgement
* The codebase is from [MS2L](https://github.com/LanglandsLin/MS2L).
* The skeleton backbone is based on [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md) and [ShiftGCN](https://github.com/kchengiva/Shift-GCN).
* The text feature is based on [Sentence-Bert](https://github.com/UKPLab/sentence-transformers).
* The baseline methods are from [SMIE](https://github.com/YujieOuO/SMIE).

## Licence
This project is licensed under the terms of the MIT license.

## Contact
For any questions, feel free to contact: kayjoe0723@gmail.com