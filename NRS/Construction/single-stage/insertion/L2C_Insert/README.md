

## [NeurIPS 2025] Learning to Insert for Constructive Neural Vehicle Routing Solver

This repository contains the code implementation of paper [Learning to Insert for Constructive Neural Vehicle Routing Solver](https://openreview.net/forum?id=SXr3Dynctm). 
In this paper, we propose Learning to Construct with Insertion-based Paradigm (L2C-Insert), a novel insertion-based learning framework for constructive NCO.
### Dependencies
```bash
Python=3.8.6
matplotlib==3.5.2
numpy==1.23.3
pandas==1.5.1
pytz==2022.1
torch==1.12.1
torchaudio==0.12.1
torchvision==0.13.1
tqdm==4.64.1
```
Also see `environment.yml`.

### Datasets and pre-trained models
The training and test datasets can be downloaded from Google Drive:
```bash
https://drive.google.com/drive/folders/1pJr3W8lbtAcbP9qfs82VJCjXbqOGQW02?usp=sharing
```
or  Baidu Cloud:
```bash
https://pan.baidu.com/s/1H7bJJmS32-fgnXUgGFJYHw?pwd=7cgg
```

### Implementation

#### Testing
```bash
TSP:
    cd L2C_Insert/TSP/Test
    python test_synthetic.py (on the synthetic dataset)
    python test_lib.py (on the tsplib dataset)
CVRP:
    cd L2C_Insert/CVRP/Test
    python test_synthetic.py (on the synthetic dataset)
    python test_lib.py (on the cvrplib dataset)
```

#### Training
```bash
TSP:
    cd /L2C_Insert/TSP/Train
    python train.py
CVRP:
    cd /L2C_Insert/CVRP/Train
    python train.py
```

## Citation

**If this repository is helpful for your research, please cite our paper:<br />**
*"Fu Luo, Xi Lin, Mengyuan Zhong, Fei Liu, Zhenkun Wang, Jianyong Sun, and Qingfu Zhang, Learning to Insert for Constructive Neural Vehicle Routing Solver, The Thirty-ninth Annual Conference on Neural Information Processing Systems, NeurIPS 2025" <br />*

**OR**

```
@inproceedings{
luo2025learning,
title={Learning to Insert for Constructive Neural Vehicle Routing Solver},
author={Fu Luo and Xi Lin and Mengyuan Zhong and Fei Liu and Zhenkun Wang and Jianyong Sun and Qingfu Zhang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=SXr3Dynctm}
}
```
****


## Acknowledgements
- https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD
- https://github.com/yd-kwon/POMO
- https://github.com/henry-yeh/GLOP