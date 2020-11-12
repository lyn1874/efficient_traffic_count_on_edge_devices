import torch
import cv2
import numpy as np
import time
from tqdm import tqdm
import pickle
import argparse
import os
import utils.aicity_utils as aic_utils
import prediction as model_arch
import efficientdet.track_count as tc
import efficientdet.utils as eff_utils
import vis_utils as vu
import count_class_batch as cc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')    


def get_args():
    parser = argparse.ArgumentParser('traffic counting')
    parser.add_argument('--project', type=str, default="aicity")    
    parser.add_argument('--frame_path', type=str, default="/project_scratch/bo/normal_data/camera/")
    parser.add_argument('-cc', '--compound_coef', type=int, default=0)
    parser.add_argument('--subtract_bg', default=False, type=str2bool,
                        help="whether preprocess the input image by subtracting the background")
    parser.add_argument('--skip', type=int, default=1, help="every %skip frame is processed")    
    parser.add_argument('--ckpt', help="detector ckpt", default=None)    
    parser.add_argument('--ooi', default="car", type=str, help="objects of interest")
    parser.add_argument('--filter_small_box', default=True, type=str2bool)
    parser.add_argument('--x_y_threshold', default=[1400, 0], 
                       help="throw out predictions that are outside of region of interest")
    parser.add_argument('--bike_speed', default=[], 
                        help="if the object of interest include cyclist, then bike_speed determines how to separate between pedestrians and cyclists. Already implemented methods use either speed information or location specific information to separate them")
    parser.add_argument('--roi_interest',default=[],
                        help="defines the region of interest, needs to be an array which has the same size as original input image")
    parser.add_argument('--save_video', default=True, type=str2bool,
                        help="whether save the statistics for creating a video")
    parser.add_argument('--active_kalman_filter', default=True, type=str2bool,
                        help="whether the kalman filter is activated during tracking stage")
    
    parser.add_argument('--real_time', default=False, type=str2bool,
                        help="whether the input is coming from a streaming video (true) or an offline video (false)")
    args = parser.parse_args()
    return args


print("---------------------------------------------------")
print("----------Please manually define the line----------")
print("---------------------------------------------------")
line0 = [0, 500, 750, 500]
line1 = [900, 450, 1200, 550]
predefine_line=[line0, line1]


def run(opt):
    #-----------------------------------------------------------#
    #--------------------Prepare Videos-------------------------#
    #-----------------------------------------------------------#
    if not os.path.exists(opt.frame_path) or len(os.listdir(opt.frame_path)) == 0:
        video_path = "frames/camera.mp4"
        save_path = opt.frame_path
        print("extract frames from the video......")
        if not os.path.isfile(video_path):
            print("The video does not exist")
            print("please download video from")
            print("https://www.youtube.com/watch?v=MNn9qKG2UFI&t=6s",
                  "and save it as camera.mp4 in the folder frames/")
        aic_utils.save_frame_from_video(video_path, opt.frame_path)
        print("Finish writing frames in the folder", opt.frame_path)
    else:
        print("There are %d frames " % len(os.listdir(opt.frame_path)))
    frames = sorted([opt.frame_path + v for v in os.listdir(opt.frame_path) if '.jpg' in v])
    
    detection_path = "frames/detections.obj"
    #-----------------------------------------------------------#
    #------------------Load Models------------------------------#
    #-----------------------------------------------------------#
    count_class = opt.ooi
    if count_class == "car":
        class_index = [1]
        class_label = ["car"]
    elif count_class == "ped_car":
        class_index = [0, 1]
        class_label = ["person", "car"]
    elif count_class == "person":
        class_index = [0]
        class_label = ["person"]
        
    threshold, nms_threshold, iou_threshold, student, params, \
        model = model_arch.get_parameters(opt.compound_coef, opt.ooi, 
                                          opt.ckpt, "aicity")
    mean, std =  params['mean'], params['std']
    if opt.subtract_bg == True:
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    frames = frames[::opt.skip]
    
    #-----------------------------------------------------------#
    #------------------Save detections--------------------------#
    #-----------------------------------------------------------#
    num_frame = len(frames)
    if not os.path.isfile(detection_path) and opt.real_time == False:
        regressboxes = eff_utils.BBoxTransform()
        clipboxes = eff_utils.ClipBoxes()
        opt_batch_size = [i for i in range(50)[2:] if num_frame % i == 0]
        if len(opt_batch_size) > 0 and max(opt_batch_size) > 20:
            opt_batch_size = opt_batch_size[-1]
        else:
            opt_batch_size = 50
            frames = frames[:opt_batch_size * (num_frame // opt_batch_size)]
        print("1. saving the detections with batch size %d.............." % opt_batch_size)
        preds = {}
        num_frame = len(frames)
        fps_detection = 0.0
        for i in tqdm(range(num_frame // opt_batch_size)):
            _imsubset = frames[i * opt_batch_size:(i+1)*opt_batch_size]
            time_init = time.time()
            _pred_oois, \
                framed_metas = model_arch.get_prediction_batch(_imsubset, input_sizes[opt.compound_coef],
                                                                  mean, std, None, model, threshold,
                                                                  nms_threshold, regressboxes, clipboxes,
                                                                  count_class, student=student,
                                                                  filter_small_box=opt.filter_small_box,
                                                                  x_y_threshold=opt.x_y_threshold,
                                                                  roi_interest=opt.roi_interest,
                                                                  only_detection=True)
            fps_detection += (time.time() - time_init)
            for j, _s_file in enumerate(_imsubset):
                preds["%s" % _s_file.split("/")[-1]] = [_pred_oois[j][0], _pred_oois[j][1], 
                                                        _pred_oois[j][2]]
        preds["framed_metas"] = framed_metas
        print("------------------------------------------")
        print("Shape of frames: %d" % num_frame, "Shape of predictions", len(preds.keys()))
        pickle.dump(preds, open(detection_path, 'wb'))
        print("Detection FPS: %.2f" % (num_frame / fps_detection))
        print("-------------------------------------------")

    else:
        print("Detection already exists, go to next step")
        
    
    #-----------------------------------------------------------#
    #---------------Tracking and Counting-----------------------#
    #-----------------------------------------------------------#
    if not os.path.isfile("frames/track_count_video/%d" % (len(frames) - 1)):
        max_objects = 30
        show=False    
        track_counter = cc.TrackCount(model, opt.compound_coef, threshold, nms_threshold, 
                                      count_class, max_objects, 
                                      iou_threshold, class_index, class_label, 
                                      opt.subtract_bg, student, params, resize=False,
                                      filter_small_box=opt.filter_small_box,
                                      x_y_threshold=opt.x_y_threshold,
                                      bike_speed=opt.bike_speed,
                                      activate_kalman_filter=opt.active_kalman_filter)
        track_counter.run(frames, "frames/track_count_stat", False,
                          "frames/track_count_video.avi", show, 
                          use_precalculated_detection=detection_path, 
                          predefine_line=predefine_line)
    
    if opt.save_video == True:
        video_path = "frames/track_count_video/"
        video_path = [video_path + v for v in sorted(os.listdir(video_path)) and '.i' not in v]
        print("--------------------------------------------------------------")
        print("-------------------saving video %s" % video_path[0])
        print("--------------------------------------------------------------")
        
        vu.give_input_to_vispostim(video_path[0], "temp", None, opt.ooi, class_label, 
                                   show=False, save_video=True, use_name=opt.ooi,
                                   algo="boundary", return_count=False, return_stat=False, 
                                   jupyter=False, box_standard = [], specify_direc=[], 
                                   return_id_remove=False, predefine_line=predefine_line,
                                   im_mom=opt.frame_path)
        

if __name__ == '__main__':
    opt = get_args()
    run(opt)
        
    
    