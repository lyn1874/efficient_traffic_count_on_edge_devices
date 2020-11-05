# Author: Zylo117

import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from torch import nn
from torchvision.ops import nms
from typing import Union
import uuid
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


def aspectaware_resize_padding_tensor(image, width, height, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    padding_h = height - new_h
    padding_w = width - new_w
    
    image = cv2.resize(image, (new_w, new_h)).astype('float32')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform_val_list = [
        T.ToTensor(),
        T.Normalize(mean, std)
    ]
    transform_compose = T.Compose(transform_val_list)
    image = transform_compose(image)
    image = F.pad(image, (0, padding_w, 0, padding_h), "constant", 0)
    return image, new_w, new_h, old_w, old_h, padding_w, padding_h


def aspectaware_resize_padding_t2(image, width, height, old_h, old_w, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height
    transform_val_list = [T.Resize((new_h, new_w)),
                          T.ToTensor(),
                          T.Normalize(mean, std)]

    padding_h = height - new_h
    padding_w = width - new_w
    transform_compose = T.Compose(transform_val_list)
    image = transform_compose(image)
    image = F.pad(image, (0, padding_w, 0, padding_h), "constant", 0)
    return image, new_w, new_h, old_w, old_h, padding_w, padding_h


def aspectaware_resize_padding_t3(image, width, height, old_h, old_w, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height
    transform_list = [T.ToTensor(),
                      T.Normalize(mean, std)]
    image = T.Compose(transform_list)(image)
    image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w))
    padding_h = height - new_h
    padding_w = width - new_w
    image = F.pad(image, (0, padding_w, 0, padding_h), "constant", 0)
    return image, new_w, new_h, padding_w, padding_h    


def preprocess_batch_tensor(samples, max_size, background, roi_interest, mean, std, input_videos=False):
    if input_videos:
        images = [v for v in samples]
    else:
        images = [cv2.imread(img_path)[:,:,::-1]/255.0 for img_path in samples]
    image_origin = images
    images = [(v-background)*roi_interest for v in images]
    imgs_meta = [aspectaware_resize_padding_tensor(image, max_size, max_size, mean, std) for image in images]
    framed_imgs = [img_meta[0].unsqueeze(0) for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]
    return image_origin, framed_imgs, framed_metas


def preprocess_batch_t3(samples, max_size, background, roi_interest, old_h, old_w, mean, std):
    """"""
    images = [cv2.imread(v)[:,:,::-1]/255.0 for v in samples]
    images = [(v-background) * roi_interest for v in images]
    t0 = time.time()
    imgs_meta = [aspectaware_resize_padding_t3(image, max_size, max_size, old_h, old_w, mean, std) for image in images]
    framed_imgs = [img_meta[0]for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]
    return images, framed_imgs, framed_metas, t0


def preprocess_batch_t2(samples, max_size, old_h, old_w, mean, std, input_videos=False):
    images = [Image.open(v).convert('RGB') for v in samples]
    orig_img = [np.asarray(v) for v in images]
    
    t0 = time.time()
    imgs_meta = [aspectaware_resize_padding_t2(image, max_size, max_size, old_h, old_w,  mean, std) for image in images]
    framed_imgs = [img_meta[0].unsqueeze(0) for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]
    return images, framed_imgs, framed_metas, t0


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
    img = cv2.imread(image_path)/255
    ori_img = img.copy()
    imh, imw = np.shape(ori_img)[:-1]

    if np.sum(roi_interest) != 0:
        ori_img *= roi_interest
    
    if np.sum(background) != 0:
        ori_img -= background[:,:,::-1]
    normalized_img = (ori_img - mean) / std
    normalized_imgs = []
    normalized_imgs.append(normalized_img)
    if num_patch == 2 and imw > imh:
        [normalized_imgs.append(normalized_img[:, i:(i+imh)]) for i in [0, imw-imh]]
    
    
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_img, framed_imgs, framed_metas 


def preprocess_batch(image_path, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229), background=[0, 0, 0], input_filenames=True, resize=False, roi_interest=[], minus_bg_norm=False):
    float_imgs = [cv2.imread(v)/255.0 for v in image_path]
    ori_imgs = float_imgs
    if np.sum(background) != 0:
        float_imgs = [[(v - background[:, :, ::-1])] for v in float_imgs]
    if np.sum(roi_interest) != 0:
        float_imgs = [v * roi_interest for v in float_imgs]
    normalized_imgs = [(img - mean) / std for img in float_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


class Augment(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, background_, mean_, std_, roi_interst):
        self.background_ = background_
        self.mean = mean_
        self.std = std_
        self.roi_interst = roi_interest
        if np.sum(background_) != 0:
            self.mean = background_
        
    def __call__(self, image):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


    

def preprocess(*image_path, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229), background=[0, 0, 0], input_filenames=True, resize=False, roi_interest=[], minus_bg_norm=False):
    if input_filenames is True:
        ori_imgs = [cv2.imread(img_path) for img_path in image_path]
        if resize:
            ori_imgs = [cv2.resize(v, dsize=(960, 960)) for v in ori_imgs]
    else:
        ori_imgs = image_path
    float_imgs = [img/255 for img in ori_imgs]
    if np.sum(background) != 0:
        float_imgs = [(img - background[:, :, ::-1]) for img in float_imgs]
        if minus_bg_norm is True:
            float_imgs = [(img - np.min(img))/(np.max(img) - np.min(img)) for img in float_imgs]
    if np.sum(roi_interest) != 0:
        float_imgs = [v * roi_interest for v in float_imgs]
    ori_imgs = float_imgs

    normalized_imgs = [(img - mean) / std for img in float_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return [v[:,:,::-1] for v in ori_imgs], framed_imgs, framed_metas


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
                'index': np.array(()),
                'trans_anchor': transformed_anchors[i].detach().cpu().numpy(),
            })
        
        sub_index = np.where(scores_over_thresh[i, :].cpu().numpy())[0]
        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)
        if anchors_nms_idx.shape[0] != 0:
            scores_, classes_ = classification_per[:, anchors_nms_idx].max(dim=0)
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]
            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
                'index': sub_index[anchors_nms_idx.cpu().numpy()],
                'trans_anchor': transformed_anchors[i].detach().cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
                'index': np.array(()),
                'trans_anchor': transformed_anchors[i].detach().cpu().numpy(),
            })

    return out


def transfer_coord_back_to_norm(rois_orig, append, plus):
    rois = rois_orig.copy()
    rois[:, 0] = rois[:, 0] - append + plus
    rois[:, 2] = rois[:, 2] - append + plus
    return rois


def get_mot_gt(gt, gt_or_pred):
    print("The gt path---------", gt)
    f = open(gt)
    gt = [np.array(v.rstrip().split(',')).astype(np.float) for v in f.readlines()]
    gt = np.array(gt)
    num_frame = np.max(gt[:, 0])
    bbox_group = []
    for i in range(int(num_frame)):
        _bbox = gt[gt[:, 0] == (i + 1)]
        if gt_or_pred is "gt":
            _bbox = _bbox[_bbox[:, -3] == 1, :]
        bbox_group.append(_bbox)   # [image_index, person_identity, tlx, tly, w, h, class, class, prob]
    print("There are %d people in total with %d unique people" % (np.sum([len(v) for v in bbox_group]), np.max([max(v[:, 1]) for v in bbox_group])))
    person_id, box_use = get_tlbr_bbox(bbox_group)
    return person_id, box_use


def get_tlbr_bbox(bbox_group):
    box_use = [v[:, 2:6].astype(np.int) for v in bbox_group]
    for iterr, single_box in enumerate(box_use):
        single_box[:, 2] += single_box[:, 0]
        single_box[:, 3] += single_box[:, 1]
        box_use[iterr] = single_box
    person_id = [v[:, 1] for v in bbox_group]
    return person_id, box_use


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
