### Student-teacher Learning for Efficient TrafficCounting 

This repository provides the implementation of our paper [Automated training of location-specific edge models for traffic counting](https://www.sciencedirect.com/science/article/pii/S0045790622000672?dgcid=coauthor). The goal of this paper is to count multiple traffic modalities (car, cyclist, pedestrians, and others) with a model that is as small as possible while maintaining a high accuracy. We experimentally show that we achieve similar results as the SToA counting methods with 5x fewer parameters.
![](frames/counting.gif)


#### Installation and preparation
1. Clone this repo and prepare the environment
  ```bash
  git clone https://github.com/lyn1874/efficient_traffic_count_on_edge_devices.git
  cd efficient_traffic_count_on_edge_devices
  ./requirement.sh
  ```

#### Traffic counting on a toy dataset
  ```bash
  python3 inference.py --compound_coef 0 --skip 4
  ```


#### Credits:
- https://www.youtube.com/watch?v=MNn9qKG2UFI&t=6s
- https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch



#### TODO
- [x] traffic counting tutorial 
- [x] update clean code
- [x] convert model to onnx
- [x] convert model to coreml
- [ ] debug the coreml model
- [ ] deploy the algorithm on Jetson, report the inference speed
- [ ] simulate the online streaming input


#### Citation
If you use our code, please cite
```
@article{LEROUX2022107763,
title = {Automated training of location-specific edge models for traffic counting},
journal = {Computers & Electrical Engineering},
volume = {99},
pages = {107763},
year = {2022},
issn = {0045-7906},
doi = {https://doi.org/10.1016/j.compeleceng.2022.107763},
url = {https://www.sciencedirect.com/science/article/pii/S0045790622000672},
author = {Sam Leroux and Bo Li and Pieter Simoens},
}
```
