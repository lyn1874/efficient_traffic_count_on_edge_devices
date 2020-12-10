#!/bin/bash
trap "exit" INT
project=${1?Error: which project am I working on? Antwerp}
camera=${2?Error: which camera will I use? CAM11_1, CAM14_4}
date=${3?Error: which date will i use for generating labels/training models/evaluating student models? Nov_28_2020}
opt=${4?Error: which operation am I doing? generate_label/train_student/evaluate_student}
skip=${5?Error: default 750. There is no need to evaluate every frame because there are 25 frames per second}
compound=${6?Error: which compounent will I use? int, 0 to 7}
ooi=${7:-ped_car}  # the class of interest, car/ped_car/person/
subtract_bg=${8:-false} # subtract (true)/ Not subtract (false) background for training the student models, bool variable
head_only=${9:-false} # finetune the head (true)/ the whole model (false) for updating the student models
traindate=${10:-none} # the date that is used to train the student model
version=${11:-0} # int

datadir=/home/jovyan/bo/dataset/  # Need to manually define
logpath=/home/jovyan/bo/exp_data/ # Need to manually define

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

