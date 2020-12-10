import torch
from torch.backends import cudnn
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from tqdm import tqdm
import time
import pickle
import json
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.utils import preprocess, invert_affine, postprocess
import efficientdet.utils as eff_utils
import prediction as model_arch
import vis_utils as vt
import utils.create_coco_type_data as cctd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

    
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
    parser = argparse.ArgumentParser('Evaluate EfficientDet model on raw data')
    parser.add_argument('--project', type=str, default="aicity")
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--params_project', type=str)
    parser.add_argument('--camera_name', type=str)
    parser.add_argument('--date', type=str)
    parser.add_argument('-wp', '--weights_path', type=str, default=None)
    parser.add_argument('--subtract_bg', type=str2bool)    
    parser.add_argument('-cc', '--compound_coef', type=int)
    parser.add_argument('--ooi', type=str)
    parser.add_argument('--coco_eval', type=str2bool, default=False)
    parser.add_argument('--skip', type=int, default=1)
    args = parser.parse_args()
    return args


def get_antwerp_data(args):
    camera_name = args.camera_name
    date = args.date
    skip = args.skip
    path_mom = args.datadir + '/%s/%s/' % (camera_name, date)
    path_all = sorted([path_mom + v for v in sorted(os.listdir(path_mom)) if 'Sequence' in v])
    image_all = []
    for single_path in path_all:
        _images = [single_path + '/' + v for iterr, v in enumerate(
            sorted(os.listdir(single_path))) if iterr % skip == 0 and '.jpg' in v]
        image_all.append(_images)
    data_path = [v for j in image_all for v in j]
    save_name = "%s_%s_eval" % (camera_name, date)
    print("--------------------------------------------")
    print("The selected camera name", "%s" % camera_name)
    print("The selected date:", date)
    print("There are %d images" % len(data_path))
    print("The save name:", save_name)
    return data_path, save_name


def get_raw_data(args):
    pass

    
def check_webcam(args, x_y_threshold=[960, 350], category_id=[], accuracy=None):
    if args.weights_path == None:
        save_dir = "/home/jovyan/bo/exp_data/" + 'teacher_result/%s/' % (args.camera_name)
        weights_path = "checkpoints/efficientdet-d%d.pth" % args.compound_coef
        ckpt_step = 0
        params = yaml.safe_load(open(f'parameter/coco.yml'))   
        args.coco_eval = False
        
    else:
        save_dir = args.weights_path.strip().split("D%d/" % args.compound_coef)[0] + "D%d/" % args.compound_coef + "tds/" 
        save_dir = save_dir + args.weights_path.split("D%d/" % args.compound_coef)[1].split('/eff')[0] + '/'
        weights_path = args.weights_path
        ckpt_step = int(weights_path.split('.pth')[0].split('_')[-1])        
        params = yaml.safe_load(open(f'parameter/{args.project}.yml'))
        args.coco_eval = True
    print("The saved tds dir", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.subtract_bg == True:
        _mean = [0, 0, 0]
        _std = [1, 1, 1]
        background = np.load('/home/jovyan/bo/dataset/%s/annotations/bg_%s.npy' % (camera_name, date))
    else:
        _mean = params['mean'][::-1]
        _std = params['std'][::-1]
        background = [0.0, 0.0, 0.0]
    preprocess_params = [_mean, _std, background]
    print("---mean............", _mean)
    print("---std.............", _std)
    print("---the max of background %.2f, the min of the background %.2f" % (np.max(background), 
                                                                             np.min(background)))
    if args.coco_eval == False:
        pred_path = save_dir + "/d%d_%s" % (args.compound_coef, args.date)
    else:
        pred_path = save_dir + "d%d_%s_ckpt_%d.json" % (args.compound_coef, args.date, ckpt_step)
    if os.path.isfile(pred_path) == False:
        if args.project  == "Antwerp":
            im_select, save_name = get_antwerp_data(args)
        else:
            im_select, save_name = get_raw_data(args)
        print("There are %d images" % len(im_select))
        if not os.path.isfile(pred_path):
            infer = InferenceOwnData(params, args.compound_coef, weights_path, device, save_dir, 
                                     preprocess_params, x_y_threshold, args.ooi, 
                                     args.coco_eval)
            infer.run_detect(im_select, args.date)
        else:
            print("File already exists")
                
    if args.project == "Antwerp" and args.coco_eval == False:
        cctd.create_antwerpen(opt.camera_name, [opt.date], opt.ooi, opt.date+"_eval", 
                              opt.compound_coef, opt.skip, 1, [0, -1])
        
    
    if args.coco_eval == True:
        gt_path = args.datadir + "/%s/annotations/instances_%s_eval.json" % (args.camera_name, args.date)
        pred_path = save_dir + "d%d_%s_ckpt_%d.json" % (args.compound_coef, args.date, ckpt_step)
        if accuracy != None:
            accuracy["ckpt_step"].append(ckpt_step)
        accuracy = _eval(gt_path, pred_path, args.ooi, category_id, accuracy)
        
        return accuracy
    
class InferenceOwnData(object):
    def __init__(self, params, compound_coef, ckpt_dir, device, exp_dir, 
                 preprocess_params, x_y_threshold=[960, 350], ooi="all", coco_eval=False):
        super(InferenceOwnData, self).__init__()
        self.params = params
        self.compound_coef = compound_coef
        self.device = device
        self.obj_list = params['obj_list']
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.exp_dir = exp_dir
        self.threshold = 0.4
        self.nms_threshold = 0.3
        if ooi == "car_truck" or "AIC" in ckpt_dir:
            self.threshold = 0.2
            self.nms_threshold = 0.5
        self.im_size = self.input_sizes[self.compound_coef]
        self.preprocess_params = preprocess_params        
        self.model = model_arch.recover_model(ckpt_dir, params, compound_coef)
        self.ooi = ooi
        self.student = [True if 'exp_data' in ckpt_dir else False][0]
        self.filter_small_box = [True if self.student == True else False][0]
        self.x_y_threshold = x_y_threshold
        self.coco_eval = coco_eval
        if len(self.obj_list) < 10:
            ckpt_step = ckpt_dir.split('.pth')[0].split('_')[-1]
            self.ckpt_step = int(ckpt_step)
        else:
            self.ckpt_step = 0        
        print("---------------Statistics----------------------")
        print("The compound coef is", self.compound_coef)
        print("The ckpt dir is", ckpt_dir)
        print("I am using student model:", self.student)
        print("Statistics are saved at", exp_dir)
        print("Classification threshold", self.threshold)
        print("NMS threshold", self.nms_threshold)
        print("I only need the predictions for", self.ooi)
        print("----------------------------------------------")


    def run_detect(self, frames, seq_name, check_freq=False):
        """This function run object detection on each images in the filenames list"""
        num_frame = len(frames)
        if self.coco_eval:
            results = []
        else:
            label_group, box_group, pred_prob_group = [], [], []
        regressboxes = eff_utils.BBoxTransform()
        clipboxes = eff_utils.ClipBoxes()
        opt_batch_size = [i for i in range(50)[2:] if num_frame % i == 0]
        if self.student == False:
            opt_batch_size = [21, 1]

        if len(opt_batch_size) > 0 and max(opt_batch_size) > 20:
            opt_batch_size = opt_batch_size[-1]
        else:
            opt_batch_size = 50
            frames = frames[:opt_batch_size * (num_frame // opt_batch_size)]
        print("1. saving the detections with batch size %d.............." % opt_batch_size)
        preds = {}
        fps_detection = 0.0
        old_image_shape = np.shape(cv2.imread(frames[0]))[:2]
        height, width = old_image_shape
        rand_show_image = np.random.choice(np.arange(num_frame), 5, replace=False)
        for i in tqdm(range(num_frame // opt_batch_size)):
            _imsubset = frames[i * opt_batch_size:(i+1)*opt_batch_size]
            time_init = time.time()
            _pred_oois, \
                framed_metas, _ = model_arch.get_prediction_batch(_imsubset, self.input_sizes[opt.compound_coef],
                                                               self.preprocess_params[0], self.preprocess_params[1],
                                                               None, self.model, self.threshold,
                                                               self.nms_threshold, regressboxes, clipboxes,
                                                               self.ooi, student=self.student,
                                                               filter_small_box=self.filter_small_box,
                                                               x_y_threshold=self.x_y_threshold,
                                                               roi_interest=[],
                                                               only_detection=True,
                                                               old_image_shape=old_image_shape)
            
            for j, stat_single_image in enumerate(_pred_oois):
                _clsid, _score, _roi = stat_single_image
                if i * opt_batch_size + j in rand_show_image:
                    show_bbox(cv2.imread(_imsubset[j])[:,:,::-1]/255.0, 
                              _roi, _clsid.astype('int32'), _score, self.obj_list, 
                              self.exp_dir + 'd%d_exp_images/' % self.compound_coef, '%d' % (i * opt_batch_size + j),True)
                if len(_roi) > 0:
                    _roi[:, 2] -= _roi[:, 0]
                    _roi[:, 3] -= _roi[:, 1]
                if not self.coco_eval:
                    for single_group, single_stat in zip([pred_prob_group, label_group, box_group], 
                                                        [_score, _clsid, _roi]):
                        single_group.append(single_stat)
                else:
                    for roi_id in range(len(_roi)):
                        score = float(_score[roi_id])
                        label = int(_clsid[roi_id])
                        box = _roi[roi_id, :]
                        image_result = {
                        'image_name': '/'.join(frames[i * opt_batch_size + j].strip().split('/')[-2:]),
                        'image_id': int(i * opt_batch_size + j +1),
                        'category_id': label + 1,
                        'score': score,
                        'bbox': box.tolist()}
                        results.append(image_result)

        if not self.coco_eval:
            pred_stat = {}
            pred_stat["filename"] = ['/'.join(v.strip().split('/')[-2:]) for v in frames]
            pred_stat["score"] = pred_prob_group
            pred_stat["label"] = label_group
            pred_stat["boxes"] = box_group
            pred_stat["heightwidth"] = [[height, width] for i in range(len(frames))]
            with open(self.exp_dir + "/d%d_%s" % (self.compound_coef, seq_name), 'wb') as f:
                pickle.dump(pred_stat, f)
        else:
            if not len(results):
                raise Exception('the model does not provide any valid output, check model architecture and the data input')
            print("write predictions into json file..............")
            json.dump(results, open(self.exp_dir+"/d%d_%s_ckpt_%d.json" % (self.compound_coef, seq_name, 
                                                                           self.ckpt_step), 'w'), indent=4)

            
def show_bbox(ori_img, roi, class_id, pred_score, class_name, save_dir, save_name, save=False, return_im=False):
    ori_img = np.array(ori_img)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if np.shape(roi)[0] > 0:
        for j in range(np.shape(roi)[0]):
            (x1, y1, x2, y2) = roi[j].astype(np.int)
            cv2.rectangle(ori_img, pt1=(x1, y1), 
                          pt2=(x2, y2), color=(1, 1, 0), thickness=4)
            label = f"{class_name[class_id[j]]}: {pred_score[j]:.2f}"
            cv2.putText(ori_img, label, 
                        (x1+20, y1+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (1, 1, 0), 2)
    if save is True:
        cv2.imwrite(save_dir + '%s.jpg' % save_name, (ori_img * 255.0).astype('uint8')[:, :, ::-1])
    if return_im is True:
        return ori_img
    
    
def transfer_coco_prediction(pred_file, ooi):
    pred_data = pickle.load(open(pred_file, 'rb'))
    boxes_g = pred_data["boxes"]
    label_g = pred_data["label"]
    score_g = pred_data["score"]
    results = []
    if ooi == "all":
        object_name_group = ["person", "bike", "car", "car", "car", "car", "car", "car"]
        act_label_group = np.array(["car", "person", "bike"])
    elif ooi == "car_truck":
        object_name_group = ["car", "truck"]
        act_label_group = np.array(["car", "truck"])
    elif ooi == "ped_car":
        object_name_group = ["person", "car"]
        act_label_group = np.array(["person", "car"])
    print(len(boxes_g), len(label_g), len(score_g))
    for image_id in range(len(boxes_g)):
        rois = boxes_g[image_id]
        class_ids = label_g[image_id]
        scores = score_g[image_id]
        for roi_id in range(len(rois)):
            score = float(scores[roi_id])
            label = class_ids[roi_id]
            box = rois[roi_id, :]
            image_result = {'image_id': int(image_id + 1),
                            'category_id': int(label + 1),
                            'score': score,
                            'bbox': box.tolist()}
            results.append(image_result)
    print("finish transfering coco-type predictions to the desired format.......")
    print("The shape of the results", len(results))
    print("The min and max of image ids", results[0]["image_id"], results[-1]["image_id"])
    print("The unique category id", np.unique([v["category_id"] for v in results]))
    
    json.dump(results, open(pred_file + '.json', 'w'), indent=4)

    
def _eval(val_gt, pred_json_path, ooi, category_id, accuracy):
    print("----the gt path", val_gt)
    print("----the prediction path", pred_json_path)
    if "teacher_result" in pred_json_path and '.json' not in pred_json_path:
        transfer_coco_prediction(pred_json_path.strip().split('.json')[0], ooi)
        pred_json_path = pred_json_path + '.json'
        print("prediction path", pred_json_path)
    coco_gt = COCO(val_gt)
    image_ids = coco_gt.getImgIds()[:10000]  ## [:10000]
    coco_pred = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    if len(category_id) != 0:
        coco_eval.params.catIds = category_id
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if type(accuracy) is dict:        
        accuracy["AP"].append(coco_eval.stats[:6])
        accuracy["AR"].append(coco_eval.stats[6:])
        return accuracy
    
    iou = coco_eval.ious
    iou_npy = []
    for i in iou.keys():
        value = iou[i]
        iou_npy.append(value)
    q = [v for j in iou_npy for v in j]
    q = [v for j in q for v in j]
    print("iou----", np.mean(q))
    if type(accuracy) is not dict:
        return coco_eval.stats


if __name__ == '__main__':
    opt = get_args()
    print("-------------------------------------------------------------------")
    print("------------------argument for current experiment------------------")
    print("-------------------------------------------------------------------")
    for arg in vars(opt):
        print(arg, getattr(opt, arg))
    print("-------------------------------------------------------------------")
    if opt.weights_path == None:
        check_webcam(opt)
    elif os.path.isfile(opt.weights_path):
        check_webcam(opt)
    else:
        accuracy = {}
        accuracy["camera"] = opt.camera_name + "_" + opt.date
        accuracy["ckpt_step"] = []
        accuracy["AP"] = []
        accuracy["AR"] = []

        weight_mom = opt.weights_path
        weight_all = [v for v in sorted(os.listdir(opt.weights_path)) if '.pth' in v]
        weight_all = sorted(weight_all, key=lambda s:int(s.split('_')[-1].split('.pth')[0]))
        for single_v in weight_all[4:]:
            opt.weights_path = weight_mom + '/' + single_v
            accuracy = check_webcam(opt, accuracy=accuracy)
        vt.write_detection_accuracy_csv(accuracy, weight_mom + "/accuracy_%s.csv" % opt.date)
    
    