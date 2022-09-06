# Basic Info
This is the unofficial implementation of the paper "Multi-Agent Trajectory Prediction with Heterogeneous Edge-Enhanced Graph Attention Network", Arxiv ID: 2106.07161.

The code is based on repo of the author. Since the Interaction Dataset has updated to v1.2 which is incompatible with the author's code, we re-code the dataset related part, and also made some simplification on the code.


# Requirements
To install the required python libraries, use pip to install: 
`pip install -r requirements.txt`


# Run Code

## Preprocess
This code is based on INTERACTION dataset v1.2. The dataset folder arrangement is as follows:<br>
.<br>
├── maps<br>
|&emsp;&emsp;├── {Scene_name_n}.osm_xy<br>
|&emsp;&emsp;├── {Scene_name_n}.osm<br>
|&emsp;&emsp;├── ...<br>
|&emsp;&emsp;└── ...<br>
├── train<br>
|&emsp;&emsp;├── {Scene_name_n}_train.csv<br>
|&emsp;&emsp;├── ...<br>
|&emsp;&emsp;└── ...<br>
├── val<br>
|&emsp;&emsp;├── {Scene_name_n}_val.csv<br>
|&emsp;&emsp;├── ...<br>
|&emsp;&emsp;└── ...<br>
└── test<br>
 &emsp;&emsp;├── {Scene_name_n}_obs.csv<br>
 &emsp;&emsp;├── ...<br>
 &emsp;&emsp;└── ...<br>

To preprocess dataset, run: `python data_preprocess.py --root /path/to/dataset --split {train/val/test}`, the processed data would be store as '.pyg' and '.pt', folder structure are as follows:<br>
.<br>
├── maps<br>
|&emsp;&emsp;├── {Scene_name_n}.osm_xy<br>
|&emsp;&emsp;├── {Scene_name_n}.osm<br>
|&emsp;&emsp;├── ...<br>
|&emsp;&emsp;└── ...<br>
├── maps_png<br>
|&emsp;&emsp;├── {Scene_name_n}_map.png<br>
|&emsp;&emsp;├── ...<br>
|&emsp;&emsp;└── ...<br>
├── train<br>
|&emsp;&emsp;├── {Scene_name_n}_train.csv<br>
|&emsp;&emsp;├── ...<br>
|&emsp;&emsp;└── ...<br>
├── val<br>
|&emsp;&emsp;├── {Scene_name_n}_val.csv<br>
|&emsp;&emsp;├── ...<br>
|&emsp;&emsp;└── ...<br>
├── test<br>
|&emsp;&emsp;├── {Scene_name_n}_obs.csv<br>
|&emsp;&emsp;├── ...<br>
|&emsp;&emsp;└── ...<br>
└── processed<br>
&emsp;&emsp;├── train<br>
&emsp;&emsp;|&emsp;&emsp;├── {Scene_name_n}_{case_id}.pyg<br>
&emsp;&emsp;|&emsp;&emsp;├── {Scene_name_n}_map.pt<br>
&emsp;&emsp;|&emsp;&emsp;├── ...<br>
&emsp;&emsp;|&emsp;&emsp;└── ...<br>
&emsp;&emsp;├── val<br>
&emsp;&emsp;|&emsp;&emsp;├── {Scene_name_n}_{case_id}.pyg<br>
&emsp;&emsp;|&emsp;&emsp;├── {Scene_name_n}_map.pt<br>
&emsp;&emsp;|&emsp;&emsp;├── ...<br>
&emsp;&emsp;|&emsp;&emsp;└── ...<br>
&emsp;&emsp;└── test<br>
&emsp;&emsp; &emsp;&emsp;├── {Scene_name_n}_{case_id}.pyg<br>
&emsp;&emsp; &emsp;&emsp;├── {Scene_name_n}_map.pt<br>
&emsp;&emsp; &emsp;&emsp;├── ...<br>
&emsp;&emsp; &emsp;&emsp;└── ...<br>

Please notice that the '.png' map are provided by the original repo.

## Run
Run `python trainval.py --batch_size {bs}` to train the model. The trained model will be saved to './models' as '.tar' file.
Run `python trainval.py --eval --metrics ALL --model {path/to/model}` to evaluate the trained model with specific or all metrics. 

## Results
Since the original repo didn't give the result, we don't have a reference of the model. And now we don't get result as the paper show. 
Results after 60 epoches training:
| ADE    | FDE    | ApFDE  | LogCosh | AFDE   | challenge_ADE |
|:-------|:-------|:-------|:--------|:-------|:--------------|
| 0.2291 | 0.7646 | 0.6114 |  0.0842 | 0.4969 |        0.2333 |  

Hope someone would find the mistake of our code and issue us, and let's get the model work together!

## Citation
If you have found this work to be useful, please consider citing the original paper:
```
@article{mo2022multi,
  title={Multi-agent trajectory prediction with heterogeneous edge-enhanced graph attention network},
  author={Mo, Xiaoyu and Huang, Zhiyu and Xing, Yang and Lv, Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2022},
  publisher={IEEE}
}
```
Thanks for code of the authors! https://github.com/Xiaoyu006/MATP-with-HEAT.git
