### Efficient multiple traffic modalities counting

The goal of this repository is to count multiple traffic modalities (car, truck, pedestrians, and cyclists) with a model that is as small as possible while maintaining high accuracy. 

Our counting framework follows detecting, tracking, and counting as shown in the figure below:

![algorithm](imgs/algorithm.png)



##### Detail

- The surveillance video that I used in the tutorial is taken from YouTube (https://www.youtube.com/watch?v=MNn9qKG2UFI&t=6s). The counting result is at: https://drive.google.com/file/d/1IuLCoXZe5OMdSxe3RNbYOAPsPouGrjSA/view?usp=sharing

- To get the counting result, run:
`python3 inference.py --compound_coef 0 --skip 4`

##### TODO
- [x] traffic counting tutorial 
- [x] update clean code
- [x] convert model to onnx
- [x] convert model to coreml
- [ ] debug the coreml model
- [ ] deploy the algorithm on Jetson, report the inference speed
- [ ] simulate the online streaming input