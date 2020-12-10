### Student-teacher Learning for Efficient TrafficCounting 

This repository provides the implementation of our paper [Student-teacher Learning for Efficient TrafficCounting on Edge Devices (Bo Li, Sam Leroux, Pieter Simoens)](). The goal of this paper is to count multiple traffic modalities (car, cyclist, pedestrians, and others) with a model that is as small as possible while maintaining a high accuracy. We experimentally show that we achieve similar results as the SToA counting methods with 5x fewer parameters.
![](frames/counting.gif)


#### Installation and preparation
1. Clone this repo and prepare the environment
  ```bash
  git clone https://gitlab.ilabt.imec.be/bobli/STTcount.git
  cd STTcount
  ./requirement.sh
  ```
2. The pre-defined parameters for the Antwerp dataset can be downloaded from: 
  ```bash
  wget 
  ```
3. Record frames from Antwerp Smart Zone
  ```bash
  python3 utils/record_antwerp.py --camera CAM11_1 --numday 1 --numhour 1 --outputdir path_to_save_data
  ```

#### Traffic counting on a toy dataset
  ```bash
  python3 inference.py --compound_coef 0 --skip 4
  ```

#### Student-teacher pipeline for the Antwerp dataset
Our framework follows: generate labels for object detection with the teacher model --> train a smaller student model with these generated labels --> evaluate the student models --> select the best one to do counting
  ```bash
  ./run_teacher_student.sh Antwerp CAM11_1 Nov_27_2020 generate_label 750  7
  ./run_teacher_student.sh Antwerp CAM11_1 Nov_27_2020 train_student 750 2 ped_car False False Nov_27_2020 0
  ./run_teacher_student.sh Antwerp CAM11_1 Nov_28_2020 evaluate_student 750 2 ped_car False False Nov_27_2020 0
  ```

#### Traffic counting on the Antwerp dataset
As for the traffic counting, our method follows detecting, tracking, and counting:
  ```bash
  ./run_antwerp_count.sh CAM11_1 Dec_09_2020 detect
  ./run_antwerp_count.sh CAM11_1 Dec_09_2020 track_count
  ./run_antwerp_count.sh CAM11_1 Dec_09_2020 save_csv
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