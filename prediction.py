import torch
from torch.backends import cudnn
import torch.nn.functional as F
import numpy as np
import yaml
import time
import matplotlib.pyplot as plt
from backbone import EfficientDetBackbone, EfficientDetBackboneMultiHeads 
from utils.utils import preprocess, invert_affine, postprocess, patch_preprocess, transfer_coord_back_to_norm, preprocess_batch_tensor, preprocess_batch_t2
import efficientdet.utils as eff_utils
import vis_utils as vt
import efficientdet.tracking_utils as tracking_utils
import efficientdet.count_utils as counting_utils

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

id_box = np.array([[9.60343517e+01, 2.86560849e+02, 2.34430635e+02, 3.89942676e+02],
                   [1.56201363e-01, 3.60303131e+02, 9.56258087e+01, 4.79096100e+02]])

id_threshold=0.75


def refined_combined_rois(rois_group):
    """rois_group is a list of rois in each patch from a single image
    the scale is already transformed. This step mainly wants to 
    filter out those boxes that are highly overlapped
    """
    select_index = []
    for iterr, roi in enumerate(rois_group):
        if iterr == 0:
            roi_g = roi
        else:
            if len(roi) > 0:
#                 overlaps = counting_utils.compute_intersection(roi_g, roi, for_counting=True)
#                 print(overlaps)
                overlaps = tracking_utils.compute_overlaps(roi_g, roi)
                select_index.append(np.arange(len(roi))[np.max(overlaps, axis=0) < 0.5])
                roi = roi[np.max(overlaps, axis=0) < 0.5]
                roi_g = np.concatenate([roi_g, roi], axis=0)
    select_index = np.array([v for j in select_index for v in j])
    return roi_g, select_index


def concate_stat(rois, indexs, scores, class_ids, roi, index, score, class_id):
    rois = np.concatenate([rois, roi], axis=0)
    indexs = np.concatenate([indexs, index], axis=0)
    scores = np.concatenate([scores, score], axis=0)
    class_ids = np.concatenate([class_ids, class_id], axis=0)
    return rois, indexs, scores, class_ids


def refined_combined_rois_intersection(rois_group, index_group, scores_group, class_ids_group):
    for iterr, roi in enumerate(rois_group):
        if iterr == 0:
            roi_g = roi
            index_g = index_group[iterr]
            score_g = scores_group[iterr]
            class_ids_g = class_ids_group[iterr]
        else:
            if len(roi) > 0:
                _keep_index = []
                _delete_index = []
                for j, single_roi in enumerate(roi):
                    intersect = counting_utils.compute_intersection(single_roi, roi_g, for_counting=True)
                    if np.max(intersect) > 0.15:
                        _ind = np.argmax(intersect)
                        _roi = roi_g[_ind]
                        now_size = (single_roi[3] - single_roi[1]) * (single_roi[2] - single_roi[0])
                        before_size = (_roi[3] - _roi[1]) * (_roi[2] - _roi[0])
                        if now_size > before_size:
                            if now_size < 10000 and before_size < 10000:
#                         if (single_roi[3] - single_roi[1]) * (single_roi[2] - single_roi[0]) > (_roi[3] - _roi[1]) * (_roi[2] - _roi[0]) :
#                             print("delete the roi in the original", _ind, single_roi, _roi)
                                _delete_index.append(_ind)
                                _keep_index.append(j)
                            
                    else:
                        _keep_index.append(j)
                left_index = np.delete(np.arange(len(roi_g)), _delete_index)
                left_index = [[] if len(left_index) == 0 else left_index][0]
                _keep_index = [[] if len(_keep_index) == 0 else _keep_index][0]
                roi_g, index_g, score_g, class_ids_g = concate_stat(roi_g[left_index], index_g[left_index], score_g[left_index], 
                                                                    class_ids_g[left_index], roi[_keep_index], 
                                                                    index_group[iterr][_keep_index],
                                                                    scores_group[iterr][_keep_index],
                                                                    class_ids_group[iterr][_keep_index])

    return roi_g, index_g, score_g, class_ids_g


def get_parameters(compound_coef, only_person, ckpt_dir, aicity=False, use_feature_maps=True):
    if only_person is "car" or only_person is "caronly":
        threshold, nms_threshold = 0.5, 0.3
    elif only_person is "person":
        threshold, nms_threshold = 0.5, 0.3
    else:
        threshold, nms_threshold = 0.5, 0.3
    if aicity:
        if not ckpt_dir:
            threshold = 0.35
        else:
            threshold = 0.5
    else:
        if not ckpt_dir:
            threshold = 0.4
        else:
            threshold = 0.5
    print("-----------------------------------------------------------------------------------")
    print("----Get parameter for EfficientDet%d to detect %s-----" % (compound_coef, only_person))
    print("-----------------------------------------------------------------------------------")
    if compound_coef >= 3:
        if "_" not in only_person:
            if only_person is "car" or only_person is "caronly":
                iou_threshold = 0.75 #75
                sim_threshold = 0.51 #0.57 before CAM1_1 0.57
            if only_person is "person":
                iou_threshold = 0.80  #0.80 
                sim_threshold = 0.51
            else:
                iou_threshold = 0.75
                sim_threshold = 0.51
            iou_threshold = [iou_threshold]
            sim_threshold = [sim_threshold]
        else:
            iou_threshold = [0.80, 0.80, 0.80] # person, car
            sim_threshold = [0.57, 0.57, 0.57] # person, car
        if not ckpt_dir:
            student = False
            params, model = get_model("coco", compound_coef, None)
        else:
            student = True
            params, model = get_model("aicity_cam2", compound_coef, ckpt_dir)
    elif compound_coef < 3:
        if "_" not in only_person:
            if only_person is "car" or only_person is "caronly":
                iou_threshold = 0.47
                sim_threshold = 0.42 #0.50
            if only_person is "person":
                iou_threshold = 0.50
                sim_threshold = 0.50
            iou_threshold = [iou_threshold]
            sim_threshold = [sim_threshold]
        else:
            iou_threshold = [0.50, 0.50]
            sim_threshold = [0.50, 0.50]
        print("Restore the fine-tuned model............")
        if not ckpt_dir:
            params, model = get_model("coco", compound_coef, None)
            student = False
        else:
            params, model = get_model("smartcity_all", compound_coef, ckpt_dir)
            student = True
    
#     if not use_feature_maps:
#         iou_threshold = [0.15 for _ in iou_threshold]
    if aicity == "aicity":
        iou_threshold = [0.15 for _ in iou_threshold]
    else:
        iou_threshold = [0.1 for _ in iou_threshold]
    print("iou threshold filter pred.......", threshold)
    print("nms threshold...................", nms_threshold)
    print("iou threshold...................", iou_threshold)
    print("similarity threshold............", sim_threshold)
    print("I am using student model........", student)
    
    return threshold, nms_threshold, iou_threshold, sim_threshold, student, params, model

        

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

    model.load_state_dict(torch.load(weights_path)) #, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    if use_float16:
        model.half()
    return model


def recover_multiheads_model(weights_path, box_level, params):
    obj_list = params['obj_list']
    compound_coef = 7
    model = EfficientDetBackbone(compound_coef=compound_coef, 
                                 box_level=box_level, num_classes=1,
                                 ratios=eval(params['anchors_ratios']), 
                                 scales=eval(params['anchors_scales']))
    
    model.load_state_dict(torch.load(weights_path), strict=True)
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    if use_float16:
        model.half()
    num_par = 0.0
    for name, p in model.named_parameters():
        num = np.prod(p.shape)
        num_par += num
    print('%.3f' % (num_par / 1000000))
    return model



def get_prediction_patch(im_filename, imsize, mean, std, background, model, threshold, nms_threshold,
                         regressBoxes, clipBoxes, only_person, get_anchors=False, student=False, 
                         roi_interest=[], minus_bg_norm=False, num_patch=2):
    ori_img, framed_imgs, framed_metas = patch_preprocess(im_filename, 
                                                          max_size=imsize, mean=mean, std=std,
                                                          background=background, roi_interest=roi_interest,
                                                          num_patch=num_patch)
    imh, imw = np.shape(ori_img)[:-1]
    x = torch.from_numpy(np.array(framed_imgs))
    x = x.to(device).permute(0, 3, 1, 2)
    q = ori_img[:, :, :]
    features, regression, classification, anchors = model.forward_test(x)
    preds = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes,
                        threshold, nms_threshold)
    preds = invert_affine(framed_metas, preds)
    rois_group = [v['rois'] for v in preds]
    if len(rois_group[1]) > 0:
        rois_group[-1] = transfer_coord_back_to_norm(rois_group[-1], 2 * imh - imw, imh)
    
    index_group = [v['index'] for v in preds]
    scores_group = [v['scores'] for v in preds]
    class_ids = [v['class_ids'] for v in preds]
    
    rois, index, scores, class_ids = refined_combined_rois_intersection(rois_group, index_group, scores_group, class_ids)
    if only_person is not "every":
        if only_person is "car" or only_person is "car_truck":
            if student:
                select_string = [0, 1]
            else:
                select_string = [2, 7]
            person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in select_string])
#             print(person_id, class_ids, len(rois))
            if only_person is "car_truck" and student is False:
                class_ids[class_ids == 2] = 0
                class_ids[class_ids == 7] = 1
            elif student is False:
                class_ids = np.ones(len(class_ids))
            if len(person_id) == 0:
                person_id = []

            if len(rois) > 0:
                rois = rois[person_id]
                index = index[person_id]        
                scores = scores[person_id]
                class_ids = class_ids[person_id]

    if get_anchors is True:
        classification = classification.detach().cpu().numpy()[0]
        output_stat = [features, rois, [], scores, index, framed_metas, classification, preds["trans_anchor"]]
    else:
        output_stat = [features, rois, [], scores, index, framed_metas, class_ids]
    return output_stat, q


def get_prediction(im_filename, imsize, mean, std, background, model, threshold, nms_threshold, 
                   regressBoxes, clipBoxes, only_person, seq=False, get_anchors=False, student=False,
                   input_filenames=True, resize=False, filter_small_box=True, 
                   x_y_threshold=[959, 0], roi_interest=[], minus_bg_norm=False, old_size=[]):
    """Give predictions based on the model
    only_person: bool, whether I only give the predictions that are people or all the predictions
    seq: bool, whether I have sequential output
    """  
    ori_imgs, framed_imgs, framed_metas = preprocess(im_filename, max_size=imsize,
                                                     mean=mean, std=std, background=background,
                                                     input_filenames=input_filenames,
                                                     resize=resize, roi_interest=roi_interest,
                                                     minus_bg_norm=minus_bg_norm)
    x = torch.from_numpy(framed_imgs[0])
    x = x.to(device)
    x = x.unsqueeze(0).permute(0, 3, 1, 2)
    q = ori_imgs[0][:, :, :] #/255.0

#     _im = cv2.imread(im_filename)
#     oldh, oldw = np.shape(cv2.imread(im_filename))[:-1]
#     ori_imgs, framed_imgs, framed_metas, _ = preprocess_batch_t2([im_filename], imsize, old_size[0], old_size[1], mean[::-1], std[::-1])
#     q = ori_imgs[0]
#     x = torch.cat(framed_imgs).cuda()
    features, regression, classification, anchors = model.forward(x)
    if seq is False:
        preds = postprocess(x, anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        orig_rois = preds[0]["rois"].copy()  # this is on the input scale
        preds = invert_affine(framed_metas, preds)[0]
        index = preds["index"]
        scores = preds["scores"]
        class_ids = preds["class_ids"]
        rois = preds["rois"]  # this is on the actual image scale
        if only_person is not "every":
            if only_person is "person":
                person_id = np.where(class_ids == 0)[0]
            elif only_person is "car":
                if student:
                    person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [0, 1]])
                else:
                    person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [2, 7]])
                class_ids = np.ones(len(class_ids))
            elif only_person is "caronly":
                person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [2]])
                class_ids = np.ones(len(class_ids))
            elif only_person is "car_truck":
                if not student:
                    person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [2, 7]])
                    class_ids[class_ids == 2] = 0
                    class_ids[class_ids == 7] = 1
                else:
                    person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [0, 1]])
            elif only_person is "bike":
                person_id = np.where(class_ids == 1)[0]
                class_ids = np.ones(len(class_ids)) * 2
            elif only_person is "ped_car":
                if not student:
                    person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [0, 2,  7]])
#                     class_ids[class_ids == 1] = 0
#                     class_ids[class_ids == 3] = 0
                    class_ids[class_ids == 2] = 1
                    class_ids[class_ids == 7] = 1
                else:
                    person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [0, 1]])
            elif only_person is "ped_car_bike":
                if not student:
                    person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [0, 1, 2, 7]])
                    class_ids[class_ids == 1] = 3
                    class_ids[class_ids == 2] = 1
                    class_ids[class_ids == 7] = 1
                    class_ids[class_ids == 3] = 2
                else:
                    person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [0, 1, 2]])
            elif only_person is "all":
                person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [0, 1, 2, 5, 7]])
                class_ids[class_ids == 1] = 3
                class_ids[class_ids == 2] = 1
                class_ids[class_ids == 5] = 1
                class_ids[class_ids == 7] = 1
            
            if len(person_id) > 0 and len(rois) > 0 and only_person is not "car_truck":
                if filter_small_box:
                    scale = (rois[person_id, 2] - rois[person_id, 0]) * (rois[person_id, 3] - rois[person_id, 1])
                    num_select = np.where(scale > 2500)[0]
                    person_id = person_id[num_select]
                if len(x_y_threshold) > 0:
                    _index = np.where(np.logical_and(rois[person_id][:, 3] >= x_y_threshold[1], 
                                                     rois[person_id][:, 2] <= x_y_threshold[0]))[0]
                    if len(_index) > 0:
                        person_id = person_id[_index]
                    if len(_index) == 0 or len(person_id) == 0:
                        person_id = []
            elif only_person is "car_truck" and len(person_id) != 0 and len(x_y_threshold) > 0:
                _index = np.where(np.logical_and(rois[person_id][:, 3] >= x_y_threshold[1], 
                                                 rois[person_id][:, 2] <= x_y_threshold[0]))[0]
                if len(_index) > 0:
                    person_id = person_id[_index]
                if len(_index) == 0 or len(person_id) == 0:
                    person_id = []
            elif len(person_id) == 0:
                person_id = []
            if len(rois) > 0:
                rois = rois[person_id]
                index = index[person_id]        
                scores = scores[person_id]
                orig_rois = orig_rois[person_id]
                class_ids = class_ids[person_id]
        if get_anchors is True:
            classification = classification.detach().cpu().numpy()[0]
            output_stat = [features, rois, orig_rois, scores, index, framed_metas, classification, preds["trans_anchor"]]
        else:
            output_stat = [features, rois, orig_rois, scores, index, framed_metas, class_ids]
    else:
        output_stat = []
        for i in range(2):
            preds = postprocess(x, anchors, regression[i], classification[i],
                                regressBoxes, clipBoxes,
                                threshold, nms_threshold)
            preds = invert_affine(framed_metas, preds)[0]
            index = preds["index"]
            scores = preds["scores"]
            class_ids = preds["class_ids"]
            rois = preds["rois"]
            if only_person is True:
                person_id = np.where(class_ids == 0)[0]
                rois = rois[person_id]
                index = index[person_id]        
                scores =scores[person_id]
            output_stat.append([rois, scores, index])
        output_stat = [features, output_stat]
    return output_stat, q


import os
import cv2
def run_infer_speed(batch_size, compound_coef, cluster=6):
    if cluster == 4:
        path_mom = "/tmp/bo/"
    elif cluster == 6:
        path_mom = "/project_scratch/bo/"
    path = path_mom + 'normal_data/aic2020/AIC20_track1/Dataset_A_Frame/eval_frames/'
    im_group = [v for v in os.listdir(path) if '.png' in v]
    im_group = [v for v in os.listdir(path) if 'cam_2_' in v]
    roi_interest = np.load('/project_scratch/bo/normal_data/aic2020/AIC20_track1/ROIs/cam_2.npy')
    background = np.load('/project_scratch/bo/normal_data/aic2020/AIC20_track1/Dataset_A_Frame/annotations/bg_cam_2.npy')
    im_group = [path + v for v in im_group]
    im_group = np.reshape(np.repeat(im_group,4, -1), [-1])
    print("There are %d images" % len(im_group))
    params_coco, model_coco = get_model("coco", compound_coef, None)
    
    inference_speed(im_group, batch_size, model_coco, compound_coef, background, roi_interest,
                   False)


def inference_speed(im_group, batch_size, model, compound_coef, use_background, 
                    roi_interest=[], minus_bg_norm=False):
    threshold = 0.6
    nms_threshold = 0.3
    if use_background is False or np.sum(use_background) == 0:
        mean_ =(0.406, 0.456, 0.485)
        std_ =(0.225, 0.224, 0.229)
        background = [0.0, 0.0, 0.0]
    else:
        if minus_bg_norm:
            mean_ =(0.406, 0.456, 0.485)
            std_ =(0.225, 0.224, 0.229)
        else:
            mean_ = (0.0, 0.0, 0.0)
            std_ = (1.0, 1.0, 1.0)
        background = use_background
        print("----subtracting the background from the input image........")
    print(mean_, std_, np.max(background), np.min(background))
    regressBoxes = eff_utils.BBoxTransform()
    clipBoxes = eff_utils.ClipBoxes()
    time_group = []
    time_preprocess = []
#     old_h, old_w = np.shape(cv2.imread(im_group[0]))[:-1]
    with torch.no_grad():
        for j in range(len(im_group) // batch_size):
            t0 = time.time()
            _impath = im_group[j * batch_size : (j+1) * batch_size]
            _, framed_imgs, framed_metas = preprocess_batch_tensor(_impath, max_size=input_sizes[compound_coef],
                                                                   background=background, roi_interest=roi_interest, 
                                                                   mean=mean_[::-1], std=std_[::-1])
            x = torch.cat(framed_imgs, 0).cuda()
            t1 = time.time()
            time_preprocess.append(t1 - t0)
            for i in range(10):
                features, regression, classification, anchors = model.forward(x)
                preds = postprocess(x, anchors, regression, classification,
                                    regressBoxes, clipBoxes,
                                    threshold, nms_threshold)        
                preds = invert_affine(framed_metas, preds)
            
            time_group.append((time.time() - t1) / 10.0)

    print("Preprocess time", np.sum(time_preprocess[10:]))
    print("Prediction time", np.sum(time_group[10:]))
    
    print("Preprocessing FPS", 1 / (np.mean(time_preprocess[10:]) / batch_size))
    print("Prediction FPS", 1 / (np.mean(time_group[10:]) / batch_size))

    print("Total FPS", 1 / (np.mean(time_group[10:]) + np.mean(time_preprocess[10:])) * batch_size) 

def predict_single_image(im_path, model, compound_coef, 
                         use_background=False, seq=False, only_person=True, show=True,
                         im_future_path=None, get_anchors=False, 
                         student=False, resize=False, filter_small_box=True,
                         x_y_threshold=[959, 0],
                         roi_interest=[], minus_bg_norm=False, patch=False):
    threshold = 0.3
    nms_threshold = 0.5
    if use_background is False or np.sum(use_background) == 0:
        mean_ =(0.406, 0.456, 0.485)
        std_ =(0.225, 0.224, 0.229)
        background = [0.0, 0.0, 0.0]
    else:
        if minus_bg_norm:
            mean_ =(0.406, 0.456, 0.485)
            std_ =(0.225, 0.224, 0.229)
        else:
            mean_ = (0.0, 0.0, 0.0)
            std_ = (1.0, 1.0, 1.0)
        background = use_background
        print("----subtracting the background from the input image........")
#     print(mean_, std_, np.max(background), np.min(background))
    regressBoxes = eff_utils.BBoxTransform()
    clipBoxes = eff_utils.ClipBoxes()
    input_filenames = [True if type(im_path) is str else False][0]
    if seq is False:
        if not patch:
            output_stat, im = get_prediction(im_path, input_sizes[compound_coef], 
                                             mean_, std_, background, 
                                             model, threshold, nms_threshold, 
                                             regressBoxes, clipBoxes, only_person, 
                                             seq=False, get_anchors=get_anchors, 
                                             student=student, input_filenames=input_filenames,
                                             resize=resize, filter_small_box=filter_small_box, 
                                             x_y_threshold=x_y_threshold, 
                                             roi_interest=roi_interest, minus_bg_norm=minus_bg_norm)
        else:
            output_stat, im = get_prediction_patch(im_path, input_sizes[compound_coef],
                                                   mean_, std_, background, 
                                                   model, threshold, nms_threshold, regressBoxes, 
                                                   clipBoxes, only_person, 
                                                   get_anchors=False, student=student, 
                                                   roi_interest=roi_interest, 
                                                   minus_bg_norm=minus_bg_norm, num_patch=2)
        features, output_stat = output_stat[0], output_stat[1:]
        if show is True:
            rois = output_stat[0]
            print("---prediction status--------")
            _pred_class = np.unique(output_stat[-1])
            [print("class id %d: %d" % (i, len(np.where(output_stat[-1] == i)[0]))) for i in _pred_class]
            img_annotate=vt.show_bbox(im, rois, output_stat[5]+1, "pred")
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.imshow(img_annotate)
    else:
        [features, output_stat], im = get_prediction(im_path, input_sizes[compound_coef], 
                                                     mean_, std_, background, 
                                                     model, threshold, nms_threshold, 
                                                     regressBoxes, clipBoxes, only_person, seq=False,
                                                     get_anchors=get_anchors, 
                                                     student=student, input_filenames=input_filenames)
        frame_txt = ["current", "future"]
        if show is True:
            if im_future_path:
                q_fu = cv2.imread(im_future_path)[:,:,::-1]/255.0
                q_g = [im, q_fu]
            else:
                q_g = [im]
            for iterr, single_q in enumerate(q_g):
                img_annotate = vt.show_bbox(single_q, output_stat[iterr][0], 
                                            output_stat[iterr][5],
                                            "preds")
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111)
                ax.imshow(img_annotate)
                ax.set_title(frame_txt[iterr])
    return features, output_stat, im