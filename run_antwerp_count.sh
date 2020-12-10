#!/bin/bash
trap "exit" INT
camera=${1?Error: which camera? CAM11_1, CAM14_4}
date=${2?Error: which date will i do the detection? Nov_28_2020}
opt=${3?Error: which stage do I want to evaluate? detect, track_count, detect_track_count}

# ***************************************************************************#
#                       User Defined Variables                               #
# ***************************************************************************#
# ckpt=checkpoints/antwerp/CAM11_1_tr_Oct_02_2020_subtractBG_False_head_False_lr_0.0010_efficientdet-d2_24_22425.pth
ckpt=/home/jovyan/bo/exp_data/student_model_D7/D2/CAM11_1_Nov_27_28_2020_lr_0.0050_decay_cosine_optim_adamw_subtractBG_False_head_False_version_102/efficientdet-d2_44_18270.pth  # can either be a pre-trained Efficientdet model or the best student model
framepath=/home/jovyan/bo/dataset  
compound_coef=2 # used component coefficient
skip=2 # read every %d frame 
ooi=ped_car # object of interest, ped_car/person/car/

# ***************************************************************************#
#                Run Detection, Tracking, Counting                           #
# ***************************************************************************#

for seq in {0..12}
do
    python3 inference_antwerpen.py --frame_path $framepath --camera $camera --date $date --sequence $seq --compound_coef $compound_coef --skip $skip --ooi $ooi --program $opt
done



