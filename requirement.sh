#!/bin/bash
pip install pycocotools
pip install tensorboardX
pip install numpy==1.17.1
pip install torch==1.7.0
pip install --upgrade torchvision
pip install filterpy
pip install openpyxl

# ------------------------------------------------------------------------------------#
# The packages below are only needed if you want to convert the model to onnx or coreml
# ------------------------------------------------------------------------------------#

# pip install --upgrade coremltools
# pip install onnxruntime
# pip install onnx


