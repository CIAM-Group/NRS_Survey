## DGL
This is the GitHub repository for 'DGL: Dynamic Global-Local Information Aggregation for Scalable VRP Generalization with Self-Improvement Learning'.

![alt text](image.png)

## DataSet
We use the same dataset as [INViT](https://github.com/Kasumigaoka-Utaha/INViT), Please prepare your dataset under the data directory in the project root directory first.

## How to run
**train**
~~~shell
cd CVRP
python train.py

cd ..
cd TSP
python train.py
~~~

**test**
~~~shell
cd CVRP
python test.py

cd ..
cd TSP
python test.py
~~~

## Acknowledgments
We would like to thank the following repository, which is the baseline of our code:

https://github.com/Kasumigaoka-Utaha/INViT

https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD


### Citation

If you find our paper and code useful, please cite our paper:

```tex
@InProceedings{DGL2025, 
              author={Xiao, Yubin and Wu, Yuesong and Cao, Rui and Wang, Di and Cao, Zhiguang and Wu, Xuan and Zhao, Peng and Li, Yuanshu and Zhou, You and Jiang, Yuan}, 
              title = {DGL: Dynamic Global-Local Information Aggregation for Scalable VRP Generalization with Self-Improvement Learning}, 
              booktitle={Proceedings of International Joint Conference on Artificial Intelligence}, 
              year={2025}}
```
