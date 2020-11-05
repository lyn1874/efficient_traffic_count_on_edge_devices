import torch
from torch.backends import cudnn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from backbone import EfficientDetBackbone
from utils.utils import preprocess, invert_affine, postprocess
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
    if not ckpt_dir:
        student = False
        params, model = get_model("coco", compound_coef, None)
    else:
        student = True
        params, model = get_model("aicity_cam2", compound_coef, ckpt_dir)
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


def get_prediction(im_filename, imsize, mean, std, background, model, threshold, nms_threshold,
                   regressboxes, clipboxes, only_person, student=False,
                   input_filenames=True, filter_small_box=True,
                   x_y_threshold=[959, 0], roi_interest=[], minus_bg_norm=False):
    """Give predictions based on the model
    """
    ori_imgs, framed_imgs, framed_metas = preprocess(im_filename, max_size=imsize,
                                                     mean=mean, std=std, background=background,
                                                     input_filenames=input_filenames,
                                                     resize=False, roi_interest=roi_interest,
                                                     minus_bg_norm=minus_bg_norm)
    x = torch.from_numpy(framed_imgs[0])
    x = x.to(device)
    x = x.unsqueeze(0).permute(0, 3, 1, 2)
    q = ori_imgs[0][:, :, :]  # /255.0
    features, regression, classification, anchors = model.forward(x)
    preds = postprocess(x, anchors, regression, classification,
                        regressboxes, clipboxes,
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

        elif only_person is "ped_car":
            if not student:
                person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [0, 2, 7]])
                class_ids[class_ids == 2] = 1
                class_ids[class_ids == 7] = 1
            else:
                person_id = np.array([_i for _i, _id in enumerate(class_ids) if _id in [0, 1]])
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
    output_stat = [[], rois, orig_rois, scores, index, framed_metas, class_ids]

    return output_stat, q


def predict_single_image(im_path, model, compound_coef,
                         use_background=False, only_person="car", show=True,
                         student=False, filter_small_box=True,
                         x_y_threshold=[959, 0], roi_interest=[], minus_bg_norm=False):
    threshold = 0.3
    nms_threshold = 0.5
    if use_background is False or np.sum(use_background) == 0:
        mean_ = (0.406, 0.456, 0.485)
        std_ = (0.225, 0.224, 0.229)
        background = [0.0, 0.0, 0.0]
    else:
        if minus_bg_norm:
            mean_ = (0.406, 0.456, 0.485)
            std_ = (0.225, 0.224, 0.229)
        else:
            mean_ = (0.0, 0.0, 0.0)
            std_ = (1.0, 1.0, 1.0)
        background = use_background
        print("----subtracting the background from the input image........")
        print(mean_, std_, np.max(background), np.min(background))
    regressboxes = eff_utils.BBoxTransform()
    clipboxes = eff_utils.ClipBoxes()
    input_filenames = [True if type(im_path) is str else False][0]

    output_stat, im = get_prediction(im_path, input_sizes[compound_coef],
                                     mean_, std_, background,
                                     model, threshold, nms_threshold,
                                     regressboxes, clipboxes, only_person,
                                     student=student, input_filenames=input_filenames,
                                     filter_small_box=filter_small_box,
                                     x_y_threshold=x_y_threshold,
                                     roi_interest=roi_interest, minus_bg_norm=minus_bg_norm)

    features, output_stat = output_stat[0], output_stat[1:]
    if show is True:
        rois = output_stat[0]
        print("---prediction status--------")
        _pred_class = np.unique(output_stat[-1])
        [print("class id %d: %d" % (i, len(np.where(output_stat[-1] == i)[0]))) for i in _pred_class]
        img_annotate = vt.show_bbox(im, rois, output_stat[5] + 1, "pred")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.imshow(img_annotate)

    return features, output_stat, im
