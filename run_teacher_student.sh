#!/bin/bash
trap "exit" INT
project=${1?Error: which project? Antwerp, Toy}
camera=${2?Error: which camera? CAM11_1, CAM14_4}
date=${3?Error: which date will i do the detection? Nov_28_2020}
opt=${4?Error: which operation am I doing?}
skip=${5?Error: the interval between frames, default 750}
compound=${6?Error: which compounent will I use? int, 0 to 7}
ooi=${7:-ped_car}
subtract_bg=${8:-false}
head_only=${9:-false}
traindate=${10:-none}
version=${11:-0}
weightpath=${12:-none}

datadir=/home/jovyan/bo/dataset/
logpath=/home/jovyan/bo/exp_data/
if [ $opt = train_student ]; then
    weightpath=checkpoints/efficientdet-d$compound.pth
elif [ $opt = evaluate_student ]; then
    if [ $compound -le 3 ]; then
        lr_use=0.0050
    else
        lr_use=0.0010
    fi
    weightpath=${logpath}student_model_D7/D$compound/${camera}_${traindate}_lr_${lr_use}_decay_cosine_optim_adamw_subtractBG_False_head_${head_only}_version_$version/
fi

if [ $opt = generate_label ]; then
    python evaluate_ts.py --project $project --camera_name $camera --date $date --subtract_bg False --compound_coef 7 --ooi $ooi --skip $skip --datadir $datadir
elif [ $opt = train_student ]; then
    echo $weightpath
    python train_student.py --project $project --compound $compound --datadir $datadir --version $version --subtract_bg $subtract_bg --head_only $head_only --saved_path $logpath/student_model_D7/ --load_weights $weightpath --lr_decay cosine
elif [ $opt = evaluate_student ]; then
    echo $weightpath
    python evaluate_ts.py --project $project --camera_name $camera --date $date --subtract_bg False --compound_coef $compound --ooi $ooi --skip $skip --datadir $datadir --weights_path $weightpath 
fi

