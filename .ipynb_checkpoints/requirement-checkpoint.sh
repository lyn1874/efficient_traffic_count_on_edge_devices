#!/bin/bash
# pip install pycocotools
# pip install tensorboardX
# pip install numpy==1.17.1
# pip install motmetrics
pip install filterpy
pip install openpyxl
pip install --upgrade coremltools
pip install onnxruntime
pip install onnx
pip install --upgrade torchvision


# ckptdir=checkpoints/
# cd $ckptdir
# echo "current directory $PWD"
# for i in 1 2 3 4 5 6 7
# do
#     filename=efficientdet-d$i.pth
#     if [ -f "$filename" ]; then
#         echo "Yeah, ckpt $filename exists"
#     else
#         echo "Download the ckpt file $filename"
# #         wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/$filename
# fi
# done
# cd ..