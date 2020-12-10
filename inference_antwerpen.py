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
import utils.utils as input_utils
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
    parser.add_argument('--frame_path', type=str, default="/project_scratch/bo/normal_data/camera/")
    parser.add_argument('--expdir', type=str, default="Results/")
    parser.add_argument('--camera', type=str, default="CAM11_1")
    parser.add_argument('--date', type=str, default="Nov_28_2020")
    parser.add_argument('--sequence', type=int, default=0)
    parser.add_argument('-cc', '--compound_coef', type=int, default=0)
    parser.add_argument('--subtract_bg', default=False, type=str2bool,
                        help="whether preprocess the input image by subtracting the background")
    parser.add_argument('--skip', type=int, default=1, help="every %skip frame is processed")    
    parser.add_argument('--ckpt', help="detector ckpt", default=None)    
    parser.add_argument('--ooi', default="car", type=str, help="objects of interest")
    parser.add_argument('--filter_small_box', default=True, type=str2bool)
    parser.add_argument('--x_y_threshold', default=[1400, 0], 
                       help="throw out predictions that are outside of region of interest")
    parser.add_argument('--save_video', default=False, type=str2bool,
                        help="whether save the statistics for creating a video")
    parser.add_argument('--active_kalman_filter', default=True, type=str2bool,
                        help="whether the kalman filter is activated during tracking stage")
    parser.add_argument('--real_time', default=False, type=str2bool,
                        help="whether the input is coming from a streaming video (true) or an offline video (false)")
    parser.add_argument('--program', default="detect_track_count",
                        help="which stages do you want to evaluate? detect, track_count, detect_track_count")
    args = parser.parse_args()
    return args



def run(opt):
    #-----------------------------------------------------------#
    #--------------------Prepare Videos-------------------------#
    #-----------------------------------------------------------#
    image_path = opt.frame_path + "/%s/%s/%s/" % (opt.camera, opt.date, opt.sequence)
    frames = [image_path + v for v in sorted(os.listdir(image_path)) if '.jpg' in v]
    
    exp_path = opt.expdir + "Detections/%s/%s/" % (opt.camera, opt.date)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    detection_path = exp_path + "detections_%s.obj" % opt.sequence
    track_path = opt.expdir + "TrackCounts/%s/%s/" % (opt.camera, opt.date)
    if not os.path.exists(track_path):
        os.makedirs(track_path)
                
    if opt.program == "save_csv":
        video_path = track_path + "%s/" % opt.sequence
        count_path = opt.expdir + "CountStatistics/"
        _c_path = count_path + "%s/%s/" % (opt.camera, opt.date)
        if not os.path.exists(_c_path):
            os.makedirs(_c_path)
        stat = [video_path + v for v in sorted(os.listdir(video_path)) if '.i' not in v][0]
        vu.write_csv_file(stat, opt.ooi, 1, 
                          "ostime", ["pedestrian", "cyclist", "car"], "boundary", 3,  
                          box_standard=[], specify_direc=[], count_path=_c_path, 
                          remove_id=False, save=True)
        return 0
        

    
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
                                          opt.ckpt, opt.camera)
    mean, std =  params['mean'], params['std']
    if opt.subtract_bg == True:
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    frames = frames[::opt.skip]
    
    if opt.program == "detect" or opt.program == "detect_track_count":
        perform_detection = True
    else:
        perform_detection = False
    
    if opt.program == "track_count" or opt.program == "detect_track_count":
        perform_track_count = True
    else:
        perform_track_count = False
    
    if perform_detection == False:
        model = None
    #-----------------------------------------------------------#
    #------------------Save detections--------------------------#
    #-----------------------------------------------------------#
    roi_interest = np.load('/home/jovyan/bo/dataset/%s/annotations/rois.npy' % opt.camera)
    roi_npy = input_utils.aspectaware_resize_padding(np.repeat(roi_interest*255.0, 3, -1), input_sizes[opt.compound_coef],
                                                    input_sizes[opt.compound_coef])[0]/255.0
    roi_tensor = torch.tensor(roi_npy, requires_grad=False, dtype=torch.float32).permute(2, 0, 1).cuda() # 3, 512, 512
    
    num_frame = len(frames)
    old_image_shape = np.shape(cv2.imread(frames[0]))[:2]
    if not os.path.isfile(detection_path) and opt.real_time == False and perform_detection == True:
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
                framed_metas, _ = model_arch.get_prediction_batch(_imsubset, input_sizes[opt.compound_coef],
                                                                  mean, std, None, model, threshold,
                                                                  nms_threshold, regressboxes, clipboxes,
                                                                  count_class, student=student,
                                                                  filter_small_box=opt.filter_small_box,
                                                                  x_y_threshold=opt.x_y_threshold,
                                                                  roi_interest=roi_tensor,
                                                                  only_detection=True, 
                                                                  old_image_shape=old_image_shape)
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
    frames = frames[:-1]
    if not os.path.isfile(track_path + "%s.obj" % opt.sequence) and perform_track_count == True:
        max_objects = 30
        show=False    
        bike_speed = np.load("/home/jovyan/bo/dataset/%s/annotations/rois_bike.npy" % opt.camera)
        track_counter = cc.TrackCount(model, opt.compound_coef, threshold, nms_threshold, 
                                      count_class, max_objects, 
                                      iou_threshold, class_index, class_label, 
                                      opt.subtract_bg, student, params, resize=False,
                                      filter_small_box=opt.filter_small_box,
                                      x_y_threshold=opt.x_y_threshold,
                                      bike_speed=bike_speed,
                                      activate_kalman_filter=opt.active_kalman_filter)
        print(frames[0])
        track_counter.run(frames, track_path+"%s.obj" % opt.sequence, False,
                          track_path+"%s.avi" % opt.sequence, show, 
                          use_precalculated_detection=detection_path, 
                          predefine_line=None)
        
    
    if opt.save_video == True and perform_track_count == True:
        video_path = track_path + "%s/" % opt.sequence
        video_path = [video_path + v for v in sorted(os.listdir(video_path)) if '.i' not in v]
        print("--------------------------------------------------------------")
        print("-------------------saving video %s" % video_path[0])
        print("--------------------------------------------------------------")
        if not os.path.isfile(video_path[0]+"%s.avi" % opt.ooi):
            vu.give_input_to_vispostim(video_path[0], opt.camera, 
                                       opt.date + "/%s" % opt.sequence, opt.ooi, class_label, 
                                       show=False, save_video=True, use_name=opt.ooi,
                                       algo="boundary", return_count=False, return_stat=False, 
                                       jupyter=False, box_standard = [], specify_direc=[], 
                                       return_id_remove=False, predefine_line=[],
                                       im_mom=opt.frame_path)

if __name__ == '__main__':
    opt = get_args()
    opt.sequence = "Sequence_%04d" % opt.sequence
    run(opt)
        
    
    