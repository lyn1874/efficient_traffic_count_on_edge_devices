# Author: Zylo117
import os
import cv2
import numpy as np
import time

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
