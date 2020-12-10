# Author: Zylo117
import os
import cv2
import numpy as np
import torch
from glob import glob
from torch import nn
from torchvision.ops import nms
from typing import Union
import uuid
from PIL import Image
import time
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

from utils.sync_batchnorm import SynchronizedBatchNorm2d


def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


def invert_affine_npy(metas, preds):
    for i in range(len(preds)):
        if len(preds[i]) == 0:
            continue
        else:
            new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
            preds[i][:, [0, 2]] = preds[i][:, [0, 2]] / (new_w / old_w)
            preds[i][:, [1, 3]] = preds[i][:, [1, 3]] / (new_h / old_h)
    return preds


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def aspectaware_resize_padding_t3(image, width, height, mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225), old_image_shape=[]):
    """Image preprocessing based on torch
    Note: Input image has to be RGB instead of BGR!!
    1. Transform it to a tensor
    2. Normalize with the given mean and std
    3. Resize it to a shape that its longest axis is as same as the required height/width
    4. Pad the image with zeros on the residual part, because the final input to the network
    should be a square
    # for Jetson, it works like this
    # but for app, I should only to the normalization part, because the resize part is done by applevision comp
    """
    old_h, old_w = old_image_shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
        padding_w = 0
        padding_h = height - new_h
        pad_value = [0, 0, padding_w, padding_h]
    else:
        new_w = int(height / old_h * old_w)
        new_h = height
        padding_w = width - new_w
        padding_h = height - new_h
        pad_value = [0, 0, padding_w, padding_h]
    transform_list = [T.ToTensor(),
                      T.Normalize(mean, std),
                      T.Resize((new_h, new_w)),
                      T.Pad(pad_value)] # the default resize behaviour is bilinear
#     transform_list = [T.ToTensor(),
#                       T.Normalize(mean, std)] # the default resize behaviour is bilinear

    image = T.Compose(transform_list)(image)
    image = image.unsqueeze(0)
    return image, new_w, new_h, old_w, old_h, padding_w, padding_h


def preprocess_ts(image, old_image_shape, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                  only_detection=False):
    """Preprocess an image
    Args:
        image: RGB, uint8, [0, 255]
        max_size: the input size for the efficientdet
        mean: RGB order
        std: RGB order
    Ops:
        1. convert image to a tensor
        2. Normalise -> resize along the longer axis -> padding    
        3. unsqueeze along axis 0
    """
    time_init = time.time()
    if '.jpg' in image or '.png' in image:
        image = Image.open(image)
#         image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
#     time_mid = time.time()
#     print("open time ", time_mid-time_init)
    img_meta = aspectaware_resize_padding_t3(image, max_size, max_size, mean, std, old_image_shape)
#     print("preprocess time", time.time() - time_mid)
    framed_imgs = img_meta[0]
    framed_metas = img_meta[1:] 
    if only_detection:
        image = [0.0]
    return np.asarray(image)/255.0, framed_imgs, framed_metas


def increase_brightness(img, value=30):
    if np.max(img) < 10.0:
        img = (img * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def patch_preprocess(image_path, max_size=512, mean=(0.406, 0.456, 0.485),
                     std=(0.225, 0.224, 0.229),
                     background=[0, 0, 0], roi_interest=[], num_patch=2):
    img = cv2.imread(image_path) / 255
    ori_img = img.copy()
    imh, imw = np.shape(ori_img)[:-1]

    if np.sum(roi_interest) != 0:
        ori_img *= roi_interest

    if np.sum(background) != 0:
        ori_img -= background[:, :, ::-1]
    normalized_img = (ori_img - mean) / std
    normalized_imgs = []
    normalized_imgs.append(normalized_img)
    if num_patch == 2 and imw > imh:
        [normalized_imgs.append(normalized_img[:, i:(i + imh)]) for i in [0, imw - imh]]

    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_img, framed_imgs, framed_metas



def preprocess(*image_path, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229), background=[0, 0, 0],
               input_filenames=True, resize=False, roi_interest=[], minus_bg_norm=False):
    if input_filenames is True:
        ori_imgs = [cv2.imread(img_path) for img_path in image_path]
        if resize:
            ori_imgs = [cv2.resize(v, dsize=(960, 960)) for v in ori_imgs]
    else:
        ori_imgs = image_path
    float_imgs = [img / 255 for img in ori_imgs]
    if np.sum(background) != 0:
        float_imgs = [(img - background[:, :, ::-1]) for img in float_imgs]
        if minus_bg_norm is True:
            float_imgs = [(img - np.min(img)) / (np.max(img) - np.min(img)) for img in float_imgs]
    if np.sum(roi_interest) != 0:
        float_imgs = [v * roi_interest for v in float_imgs]
    ori_imgs = float_imgs

    normalized_imgs = [(img - mean) / std for img in float_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return [v[:, :, ::-1] for v in ori_imgs], framed_imgs, framed_metas


def nms_cpu(boxes, scores, overlap_threshold=0.5, min_mode=False):
    boxes = boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        keep.append(order[0])
        xx1 = np.maximum(x1[order[0]], x1[order[1:]])
        yy1 = np.maximum(y1[order[0]], y1[order[1:]])
        xx2 = np.minimum(x2[order[0]], x2[order[1:]])
        yy2 = np.minimum(y2[order[0]], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if min_mode:
            ovr = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]
    return keep

def postprocess_npy(transformed_anchors, regression, classification, threshold, iou_threshold):
    scores = np.max(classification, axis=2, keepdims=True)
    scores_over_thresh = (scores > threshold)[:,:,0]
    out = []
    for i in range(regression.shape[0]):  # batch_size
        if scores_over_thresh[i].sum() == 0:
            pass
        else:
            classification_per = np.transpose(classification[i, scores_over_thresh[i, :], ...], (1, 0))
            transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            anchors_nms_idx = nms_cpu(transformed_anchors_per, scores_per[:, 0],
                                      overlap_threshold=iou_threshold)
            
            if len(anchors_nms_idx) != 0:
                scores_ = np.max(classification_per[:, anchors_nms_idx], axis=0)
                classes_ = np.argmax(classification_per[:, anchors_nms_idx], axis=0)
                boxes_ = transformed_anchors_per[anchors_nms_idx, :]
                _value = np.concatenate([boxes_, np.expand_dims(classes_, axis=-1), np.expand_dims(scores_, axis=-1)], axis=-1)
                print(np.shape(_value))
                out.append(_value)
    return out


def postprocess(transformed_anchors, regression, classification, threshold, iou_threshold):
    scores = torch.max(classification, dim=2, keepdim=True)[0]  # return max value and the corresponding index
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(regression.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
        else:
            sub_index = np.where(scores_over_thresh[i, :].cpu().numpy())[0]
            classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
            transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            time_init = time.time()
            anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)
            if anchors_nms_idx.shape[0] != 0:
                scores_, classes_ = classification_per[:, anchors_nms_idx].max(dim=0)
                boxes_ = transformed_anchors_per[anchors_nms_idx, :]
                out.append({
                    'rois': boxes_.cpu().numpy(),
                    'class_ids': classes_.cpu().numpy(),
                    'scores': scores_.cpu().numpy(),
                })
            else:
                out.append({
                    'rois': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                })

    return out


def transfer_coord_back_to_norm(rois_orig, append, plus):
    rois = rois_orig.copy()
    rois[:, 0] = rois[:, 0] - append + plus
    rois[:, 2] = rois[:, 2] - append + plus
    return rois


def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])


def replace_w_sync_bn(m):
    for var_name in dir(m):
        target_attr = getattr(m, var_name)
        if type(target_attr) == torch.nn.BatchNorm2d:
            num_features = target_attr.num_features
            eps = target_attr.eps
            momentum = target_attr.momentum
            affine = target_attr.affine

            # get parameters
            running_mean = target_attr.running_mean
            running_var = target_attr.running_var
            if affine:
                weight = target_attr.weight
                bias = target_attr.bias

            setattr(m, var_name,
                    SynchronizedBatchNorm2d(num_features, eps, momentum, affine))

            target_attr = getattr(m, var_name)
            # set parameters
            target_attr.running_mean = running_mean
            target_attr.running_var = running_var
            if affine:
                target_attr.weight = weight
                target_attr.bias = bias

    for var_name, children in m.named_children():
        replace_w_sync_bn(children)


class CustomDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                 inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))
                for device_idx in range(len(devices))], \
               [kwargs] * len(devices)


def get_last_weights(weights_path):
    weights_path = glob(weights_path + f'/*.pth')
    weights_path = sorted(weights_path,
                          key=lambda x: int(x.rsplit('_')[-1].rsplit('.')[0]),
                          reverse=True)[0]
    print(f'using weights {weights_path}')
    return weights_path


def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                module.bias.data.zero_()
