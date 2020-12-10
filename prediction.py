import torch
from torch.backends import cudnn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from backbone import EfficientDetBackbone
import utils.utils as input_utils
import time
# from utils.utils import preprocess, invert_affine, postprocess

import efficientdet.utils as eff_utils
import vis_utils as vt


input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

id_box = np.array([[9.60343517e+01, 2.86560849e+02, 2.34430635e+02, 3.89942676e+02],
                   [1.56201363e-01, 3.60303131e+02, 9.56258087e+01, 4.79096100e+02]])

id_threshold = 0.75


def get_parameters(compound_coef, only_person, ckpt_dir, aicity=False):
    if only_person is "car" or only_person is "caronly":
        threshold, nms_threshold = 0.5, 0.3
    elif only_person is "person":
        threshold, nms_threshold = 0.5, 0.3
    else:
        threshold, nms_threshold = 0.5, 0.3
    if aicity:
        threshold = [0.35 if not ckpt_dir else 0.6][0]
    else:
        threshold = [0.4 if not ckpt_dir else 0.6][0]
    print("-----------------------------------------------------------------------------------")
    print("----Get parameter for EfficientDet%d to detect %s-----" % (compound_coef, only_person))
    print("-----------------------------------------------------------------------------------")
    if not ckpt_dir:
        student = False
        params, model = get_model("coco", compound_coef, None)
    else:
        student = True
        params, model = get_model("cam11_1", compound_coef, ckpt_dir)
    if aicity == "aicity":
        iou_threshold = [0.15 for _ in range(3)]
    else:
        iou_threshold = [0.1 for _ in range(3)]
    print("iou threshold filter pred.......", threshold)
    print("nms threshold...................", nms_threshold)
    print("iou threshold...................", iou_threshold)
    print("I am using student model........", student)

    return threshold, nms_threshold, iou_threshold, student, params, model


def get_model(project_name, compound_coef, ckpt_dir):
    params = yaml.safe_load(open(f'parameter/{project_name}.yml'))
    print(params['obj_list'])
    model = recover_model(ckpt_dir, params, compound_coef)
    return params, model


def recover_model(weights_path, params, compound_coef):
    obj_list = params['obj_list']
    if not weights_path:
        weights_path = 'checkpoints/efficientdet-d%d.pth' % compound_coef
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    num_par = 0.0
    for name, p in model.named_parameters():
        num = np.prod(p.shape)
        num_par += num
    print('%.7f' % (num_par / 1e6))
    model.load_state_dict(torch.load(weights_path))  # , map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    if use_float16:
        model.half()
    return model

def find_ooi(preds, ooi, filter_small_box, x_y_threshold, student):
    """Finds the object-of-interest (ooi)
    preds: an object, keys rois, class_ids, scores
    """
    scores = preds["scores"]
    class_ids = preds["class_ids"]
    rois = preds["rois"]  # this is on the actual image scale
    if ooi is "every" or len(rois) == 0:
        return scores, class_ids, rois
    else:
        if ooi == "person":
            ooi_id = np.where(class_ids == 0)[0]
        elif "car" in ooi and "ped" not in ooi:
            ooi_cls_id = [[0, 1] if student == True else [2, 7]][0]
            ooi_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in ooi_cls_id])
            if ooi == "car":
                class_ids = np.ones(len(class_ids))
            elif ooi == "car_truck" and student == False:
                class_ids[class_ids == 2] = 0
                class_ids[class_ids == 7] = 1
        elif ooi == "ped_car":
            if not student:
                ooi_cls_id = [0, 2, 7]
                ooi_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in ooi_cls_id])
                class_ids[class_ids == 2] = 1
                class_ids[class_ids == 7] = 1 
            else:
                ooi_id = np.arange(len(class_ids))
        if len(ooi_id) > 0:
            if filter_small_box:
                scale = (rois[ooi_id, 2] - rois[ooi_id, 0]) * (rois[ooi_id, 3] - rois[ooi_id, 1])
                ooi_id = ooi_id[np.where(scale > 2500)[0]]
            if len(x_y_threshold) > 0:
                _index = np.where(np.logical_and(rois[ooi_id][:, 3] >= x_y_threshold[1],
                                                 rois[ooi_id][:, 2] <= x_y_threshold[0]))[0] #height, width
                if len(_index) > 0:
                    ooi_id = ooi_id[_index]
                else:
                    ooi_id = []
        if len(ooi_id) == 0:
            ooi_id = []
        rois = rois[ooi_id]
        scores = scores[ooi_id]
        class_ids = class_ids[ooi_id]
        return class_ids, scores, rois
    
    
def get_inference_speed(batch_size, compound_coef):
    import os
    import cv2
    impath = "/home/jovyan/bo/dataset/CAM11_1/Nov_27_2020/Sequence_0000/"
    frames = [impath + v for v in sorted(os.listdir(impath)) if '.jpg' in v][:400]
    num_iter = len(frames) // batch_size
    time_preprocess, time_prediction = 0.0, 0.0
    params, model = get_model("coco", compound_coef, None)
    threshold = 0.6
    nms_threshold = 0.3
    old_image_shape = np.shape(cv2.imread(frames[0]))[:2]
    imsize = input_sizes[compound_coef]
    mean, std =  params['mean'], params['std']
    only_detection =True
    with torch.no_grad():
        for i in range(num_iter):
            time_init = time.time()
            filenames = frames[i * batch_size : (i+1) * batch_size]
            orig_im, framed_im, framed_metas = [], [], []
            for single_im in filenames:
                _s_orig_im, _s_framed_im, \
                    _s_framed_metas = input_utils.preprocess_ts(single_im, old_image_shape=old_image_shape,
                                                                max_size=imsize, mean=mean, std=std, 
                                                                only_detection=only_detection)
                orig_im.append(_s_orig_im)
                framed_im.append(_s_framed_im)
                framed_metas.append(_s_framed_metas)
            im_fake = torch.cat(framed_im, dim=0).to(device)
            time_mid = time.time()
            regression, classification, anchors = model.forward_test(im_fake)
            preds = input_utils.postprocess(anchors, regression, 
                                            classification, threshold, nms_threshold)
            preds = input_utils.invert_affine(framed_metas, preds)
            time_prediction += (time.time() - time_mid)
            time_preprocess += (time_mid - time_init)

        print("=========================================================")
        print("Preprocess fps", len(frames)/time_preprocess)
        print("Prediction fps", len(frames)/time_prediction)
        print("=========================================================")
    


def get_prediction_batch(images, imsize, mean, std, background, model, threshold, nms_threshold, 
                         regressboxes, clipboxes, ooi, student=False,
                         filter_small_box=True, x_y_threshold=[959, 0], roi_interest=[],
                         only_detection=True, old_image_shape=[]):
    orig_im, framed_im, framed_metas = [],[], []
    time_init = time.time()
    for single_im in images:
        _s_orig_im, _s_framed_im, \
            _s_framed_metas = input_utils.preprocess_ts(single_im, old_image_shape=old_image_shape,
                                                        max_size=imsize, mean=mean, std=std, 
                                                        only_detection=only_detection)
        orig_im.append(_s_orig_im)
        framed_im.append(_s_framed_im)
        framed_metas.append(_s_framed_metas)
    framed_im = torch.cat(framed_im, dim=0).to(device)
    if len(roi_interest) > 0:
        framed_im = framed_im.mul(roi_interest)
    time_mid = time.time()
    regression, classification, anchors = model.forward_test(framed_im)
    preds = input_utils.postprocess(anchors, regression, 
                                    classification, threshold, nms_threshold)
    preds = input_utils.invert_affine(framed_metas, preds)
    preds_ooi = [find_ooi(v, ooi, filter_small_box, x_y_threshold, student) for v in preds]
    time_post = time.time() - time_mid
    return preds_ooi, _s_framed_metas, [time_mid - time_init, time_post]


def get_prediction_ts(image, imsize, mean, std, background, model, threshold, nms_threshold,
                      regressboxes, clipboxes, ooi, student=False, filter_small_box=True,
                      x_y_threshold=[1300, 0], roi_interest=[]):
    orig_im, framed_im, framed_metas = input_utils.preprocess_ts(image, max_size=imsize, 
                                                                 mean=mean, std=std)
    framed_im = framed_im.to(device)
    regression, classification, anchors = model.forward_test(framed_im)
    preds = input_utils.postprocess(anchors, regression, 
                                    classification, threshold, nms_threshold)
    preds = input_utils.invert_affine([framed_metas], preds)
    preds_ooi = [find_ooi(v, ooi, filter_small_box, x_y_threshold, student) for v in preds]
    return preds_ooi[0], [framed_metas], orig_im # [ classids, scores, rois]
        
    