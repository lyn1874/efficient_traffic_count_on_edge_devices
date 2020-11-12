# This script contains all the utility function for visualization
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import csv
import pickle
from efficientdet.postprocess_count_utils import *
import efficientdet.count_utils as count_utils


def show_bbox(ori_img, roi, pred_score, gt_or_pred, show=False):
    if np.max(ori_img) > 100:
        ori_img = ori_img / 255.0
    color = [(1, 0, 0) if gt_or_pred is "gt" or gt_or_pred is "d0" else (0, 1, 0)][0]
    if gt_or_pred is "pred2":
        color = (1, 1, 1)
    imh, imw = np.shape(ori_img)[:2]
    ori_img = cv2.UMat(ori_img).get()
    if np.shape(roi)[0] > 0:
        for j in range(np.shape(roi)[0]):
            x1, y1, x2, y2 = roi[j].astype(np.int)
            cv2.rectangle(ori_img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2)
            if len(pred_score) > 0:
                label = '%d' % pred_score[j]
                cv2.putText(ori_img, label,
                            (x1 + 20, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            color, 2)
    if "d" in gt_or_pred and "pre" not in gt_or_pred:
        if gt_or_pred is "d0":
            cv2.putText(ori_img, gt_or_pred + ": %d" % len(roi), (imw - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color,
                        2)
        elif gt_or_pred is "d7":
            cv2.putText(ori_img, gt_or_pred + ": %d" % len(roi), (imw - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color,
                        2)
    if show is True:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.imshow(ori_img)
    else:
        return ori_img


def get_video_writer(shape, out_path, name):
    if "aicity" in out_path:
        fps_use = 12.0
    elif "jovyan" in out_path:
        fps_use = 12.0
    else:
        fps_use = 6.0
    shape_use = (shape[1], shape[0])
    if '.mp4' or '.avi' in out_path:
        out_name = out_path
    else:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_name = out_path + '%s.avi' % name
    if '.mp4' in out_path:
        api_prefer = cv2.VideoWriter_fourcc(*'MP4V')
    elif '.avi' in out_path:
        api_prefer = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(filename=out_name, apiPreference=cv2.CAP_FFMPEG,
                          fourcc=api_prefer,
                          fps=fps_use, frameSize=shape_use)
    return out


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


def blur_image(image, rois):
    """This function is used to blur the image w.r.t the rois
    Args:
    image: [imh, imw, 3]
    rois: [number_rois, 4]
    """
    for iterr, single_roi in enumerate(rois):
        x1, y1, x2, y2 = single_roi.astype('int')
        subset = image[y1:y2, x1:x2]
        subset_blur = cv2.GaussianBlur(subset, (13, 13), 2.5)
        image[y1:y2, x1:x2] = subset_blur
    return image


def put_multiple_stat_on_im(image, bbox, current_person_id, new_object_index, count, ep, class_group, show=False,
                            show_direction_count=False, direction_count_stat=None, im_index=0,
                            resize=False):
    image = np.array(image)
    for iterr, single_class in enumerate(class_group):
        if iterr < len(class_group) - 1 and len(class_group) > 1:
            show_ = False
        else:
            show_ = show
        image = put_stat_on_im(image, bbox[iterr], current_person_id[iterr], new_object_index[iterr],
                               count[iterr], ep[iterr], show_, show_direction_count, direction_count_stat[iterr],
                               im_index, resize, single_class[0], single_class)
    return image


def put_stat_on_im(image, bbox, current_person_id, new_object_index, count, ep, show=False,
                   show_direction_count=False, direction_count_stat=None, im_index=0,
                   resize=False, box_text="id", object_text="person"):
    color_group = [[1, 0, 0], [0, 0, 1]]
    imw = np.shape(image)[1]
    if object_text == "car":
        left_coord = imw - 145
    elif object_text == "person":
        left_coord = 5
    elif object_text == "bike":
        left_coord = imw - 290
    elif object_text == "truck":
        left_coord = 5
    if show_direction_count:
        text_coordinate = np.array([[left_coord, 190]])
    else:
        text_coordinate = np.array([[left_coord, 40]])

    if show_direction_count:
        line_group, count_movement, direction_arrow = direction_count_stat
        line_color = [1, 0, 0]
        direction_arrow = direction_arrow.astype(np.int)
        color_move_group = [[153 / 255, 0, 0], [1, 0.5, 0], [0, 255 / 255, 128 / 255],
                            [255 / 255, 255 / 255, 128 / 255], [1, 1, 1],
                            np.random.random([3]), np.random.random([3]), np.random.random([3])]

        for single_line in line_group:
            x11, y11, x12, y12 = single_line
            cv2.line(image, (x11, y11), (x12, y12), line_color, thickness=2)

        count_coord = [[left_coord, v] for v in [30, 70, 110, 150, 190, 230]]

        ac = [[80, -15, 80, -45],
              [80, -45, 80, -15],
              [100, -15, 80, -15],
              [80, -15, 100, -15],
              [80, -15, 80, -45],
              [80, -45, 80, -15],
              [100, -15, 80, -15],
              [80, -15, 100, -15],
              ]  # 40, 60 for antwerpen dataset
        text_coordinate = np.concatenate([count_coord, text_coordinate],
                                         axis=0)
        text, count_use = [], []
        text.append([object_text])

        if len(count_movement) == 5:
            text.append(["1u", "1d", "2u", "2d"])
            count_use.append(count_movement[1:])
        elif len(count_movement) == 4:
            text.append(["1u", "1d", "2u", "2d"])
            count_use.append(count_movement)
        else:
            text.append(["up", "down"])
            count_use.append(count_movement[-2:])
        text.append(["total"])
        count_use.append([count])
        text = [v for j in text for v in j]
        count_group = [v for j in count_use for v in j]
    else:
        text = ["count"]
        count_group = [count]
    if np.sum(bbox) > 0:
        for j in range(len(bbox)):
            (x1, y1, x2, y2) = bbox[j].astype(np.int)
            if j in new_object_index:
                _b_color = color_group[0]
            else:
                if ep[j] == 0:
                    _b_color = color_group[1]
                else:
                    _b_color = [0, 1, 0]
            if show_direction_count:
                if direction_arrow[j] != 0:
                    _direc = direction_arrow[j] - 1
                    _b_color = color_move_group[_direc]
                    image = cv2.arrowedLine(image, tuple([x1 + ac[_direc][0], y1 + ac[_direc][1]]),
                                            tuple([x1 + ac[_direc][2], y1 + ac[_direc][3]]), tuple(_b_color), 2,
                                            tipLength=0.4)
            if ep[j] != 0:
                _b_color = [0, 1, 0]
            cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2),
                          color=_b_color, thickness=1)
            text_use = "%s%d" % (box_text, current_person_id[j])
            cv2.putText(image, text_use,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, tuple(_b_color), 2)
    text_width, text_height = 140, 30  # 115, 30
    for j in range(len(text)):
        text_offset_x, text_offset_y = text_coordinate[j]
        box_coords = ((text_offset_x, text_offset_y + 4), (text_offset_x + text_width + 2, text_offset_y - text_height))
        image[box_coords[1][1]:box_coords[0][1], box_coords[0][0]:box_coords[1][0]] = [192 / 255, 192 / 255, 192 / 255]
        text_color = [[1, 1, 1] if not show_direction_count else color_move_group[j - 1]][0]
        if j != 0:
            text_use = text[j] + ' %d' % count_group[j - 1]
            cv2.putText(image, text_use, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(text_color),
                        2)
        else:
            text_use = text[j]
            cv2.putText(image, text_use, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple([1, 1, 1]),
                        2)
    if resize:
        image = cv2.resize(image, dsize=(960, 720))
    if show is True:
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111)
        ax.imshow(image)
        ax.set_title(im_index)
    return image


# -------------------------------------------------------------------------------------------------#
# -------------------------------Create videos-----------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
def assign_direction_to_string(_stat):
    _movement_string = np.array(["static", "up", "down", "left", "right"])
    for i in range(3):
        new_movement = {}
        for single_key in _stat[-1][i].keys():
            _value = np.array(_stat[-1][i][single_key])
            _value = _value[_value != 0].astype('int32')
            if len(np.unique(_value)) > 0:
                new_movement[single_key] = [_movement_string[np.unique(_value)][0]]
        _stat[-1][i] = new_movement
    return _stat


def vis_postprocessing_im(image, bbox, count_stat, direction_stat, box_text, line_group, color_move_group=[]):
    if len(color_move_group) == 0:
        color_move_group = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 128 / 255, 0], [255 / 255, 255 / 255, 0],
                            [0, 76 / 255, 153 / 255], [0.4, 0.3, 0.2], [0.5, 0.7, 0.9]]
    if np.sum(bbox) > 0:
        bbox_ = bbox.copy()
        bbox_[:, :2] += (bbox_[:, 2:] - bbox_[:, :2]) / 2.0
    ac = [[],
          [0, -35, 0, -55],
          [0, -55, 0, -35],
          [20, -55, 0, -55],
          [0, -55, 20, -55],
          [20, -55, 0, -55],
          [0, -55, 20, -55],
          [0, -35, 0, -55],
          [0, -55, 0, -35],
          [0, -35, 0, -55],
          [0, -55, 0, -35],
          [0, -35, 0, -55],
          [0, -55, 0, -35],
          [0, -35, 0, -55],
          [0, -55, 0, -35]]

    #           [20, -55, 0, -55],
    #           [0, -55, 20, -55],
    #           [20, -55, 0, -55],
    #           [0, -55, 20, -55],
    #            ]  # 40, 60 for antwerpen dataset
    radius = 4
    if len(line_group) > 0:
        for line_iter, single_line in enumerate(line_group):
            x11, y11, x12, y12 = single_line
            cv2.line(image, (x11, y11), (x12, y12), color_move_group[1 + line_iter * 2], thickness=2)

    if np.sum(bbox) > 0:
        for j in range(len(bbox)):
            x_c, y_c = bbox_[j, :2].astype(np.int)
            _direc = int(direction_stat[j])
            image = cv2.circle(image, (x_c, y_c), radius, color=tuple(color_move_group[_direc]),
                               thickness=-1)  # what is this circle?
            text_use = "%s%d" % (box_text[j][0], count_stat[j])
            cv2.putText(image, text_use, (x_c, y_c - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        tuple(color_move_group[_direc]), 2)
            image = cv2.arrowedLine(image, tuple([x_c + ac[_direc][0], y_c + ac[_direc][1]]),
                                    tuple([x_c + ac[_direc][2], y_c + ac[_direc][3]]),
                                    tuple(color_move_group[_direc]), 2, tipLength=0.4)

    return image


def put_accumulate_num_on_im(im, only_person, count_stat, movement_string):
    imw = np.shape(im)[1]
    color_move_group = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 128 / 255, 0], [255 / 255, 255 / 255, 0],
                        [0, 76 / 255, 153 / 255], [0.4, 0.3, 0.2], [0.5, 0.7, 0.9]]
    if not movement_string:
        text_use = ["s", "U", "D", "L", "R"]
    else:
        text_use = ["s", "m1U", "m1D", "m2U", "m2D", "m3U", "m3D", "m4U", "m4D"]
    c_left = imw - 120
    b_left = 135
    p_left = 5
    count_loc = [70 + 40 * i for i in range(len(movement_string))]
    if only_person is "person":
        c_group = ["person", "bike"]
        count_coord_p = [[p_left, v] for v in count_loc]
        count_coord_b = [[b_left, v] for v in count_loc]
        count_coord_g = [count_coord_p, count_coord_b]
    elif only_person is "car":
        c_group = ["car"]
        count_coord_c = [[c_left, v] for v in count_loc]
        count_coord_g = [count_coord_c]
    elif only_person is "ped_car":
        count_coord_p = [[p_left, v] for v in count_loc]
        count_coord_b = [[b_left, v] for v in count_loc]
        count_coord_c = [[c_left, v] for v in count_loc]
        c_group = ["person", "bike", "car"]
        count_coord_g = [count_coord_p, count_coord_b, count_coord_c]

    text_width, text_height = 120, 30  # 115, 30
    for iterr, single_cls in enumerate(c_group):
        text_coordinate = count_coord_g[iterr]
        _text = [single_cls]
        _text = np.concatenate(
            [_text, ["%s:%d" % (text_use[q + 1], count_stat[iterr][q + 1]) for q in range(len(movement_string) - 1)]],
            axis=0)  # why only range 2
        for j in range(len(movement_string)):
            text_offset_x, text_offset_y = text_coordinate[j]
            box_coords = (
            (text_offset_x, text_offset_y + 4), (text_offset_x + text_width + 2, text_offset_y - text_height))
            im[box_coords[1][1]:box_coords[0][1], box_coords[0][0]:box_coords[1][0]] = [192 / 255, 192 / 255, 192 / 255]
            text_color = color_move_group[j]
            cv2.putText(im, _text[j], (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(text_color), 2)
    return im


def give_input_to_vispostim(filename, camera, seq, only_person, class_group, show, save_video, use_name, algo,
                            movement_string=None, return_count=False,
                            return_stat=False, jupyter=True,
                            box_standard=[], specify_direc=[], return_id_remove=False, predefine_line=[],
                            im_mom=None):
    if algo is "boundary":
        if "CAM" in camera:
            line_group, counted_direc = count_utils.give_line_group_antwerp(camera, only_person)
        elif camera == "aicity":
            line_group, counted_direc = count_utils.give_aicity_linegroup(seq)
        else:
            line_group = predefine_line
            counted_direc = np.arange(len(line_group))
        movement_string = [["static"]]
        [movement_string.append(["move-%d-up" % i, "move-%d-down" % i]) for i in range(len(line_group) + 1)[1:]]
        movement_string = [v for j in movement_string for v in j]
    else:
        line_group = []
        if not movement_string:
            movement_string = ["static", "up", "down", "left", "right"]
        else:
            movement_string = [["static"]]
            [movement_string.append(["move-%d-up" % i, "move-%d-down" % i]) for i in range(3)[1:]]
            movement_string = [v for j in movement_string for v in j]
    if camera is "aicity":
        im_mom = "/project_scratch/bo/normal_data/aic2020/AIC20_track1/Dataset_A_Frame/%s/" % seq
    elif "CAM" in camera:
        im_mom = '/home/jovyan/bo/dataset/%s/%s/' % (camera, seq)
    else:
        im_mom = im_mom
    try:
        stat_original = pickle.load(open(filename, 'rb'))
    except Exception as error:
        print(filename, "does not exist")
        return []
    #     print("The used statistics", filename)
    #     print("The shape of the saved statistics", len(stat_original))
    stat, count_table_frameindex = rearrange_stat(stat_original, only_person, class_group, algo, movement_string)
    if return_stat:
        return stat
    if len(box_standard) > 0:
        id_remove = count_utils.filter_wrong_prediction(stat, box_standard, specify_direc, only_person)
        id_exist = stat[-1][-1]
        for single_id in id_remove:
            del id_exist["id%d" % single_id]
        stat_original[-1][-1] = id_exist
        stat, count_table_frameindex = rearrange_stat(stat_original, only_person, class_group, algo, movement_string)
        if return_id_remove:
            return id_remove
    else:
        id_remove = []

    count, time_use = give_count(stat, only_person, movement_string)
    if return_count:
        return count, time_use, count_table_frameindex, id_remove
    if only_person is "person":
        count = count[:, :2]
    elif only_person is "car":
        count = count[:, -1:]
    num_im = len(stat) - 1
    shape = tuple(np.shape(cv2.imread(im_mom + stat_original[0]["frame"]))[:-1])
    _epnum = 1
    if save_video:
        video_writer = get_video_writer(shape, filename + '%s.avi' % use_name, [])
    print("There are %d images" % num_im)
    for i in range(num_im):  # [12:13]:
        _s = stat[i]
        im = cv2.imread(im_mom + _s["frame"])[:, :, ::-1] / 255.0
        for _citer, _cls in enumerate(class_group):
            _id = _s["count_id_%s" % _cls]
            if len(_id) > 0 and np.sum(_id) > 0:
                _ep = _s["ep_%s" % _cls][_id != 0]
                _box = np.array(_s["rois_%s" % _cls])[_id != 0][_ep != _epnum]
                _direc = _s["direction_arrow_%s" % _cls][_id != 0][_ep != _epnum]
                im = blur_image(im, _box)
                im = vis_postprocessing_im(im, _box, _id[_id != 0][_ep != _epnum], _direc,
                                           np.array(_s["identity_%s" % _cls])[_ep != _epnum], line_group)
            else:
                im = vis_postprocessing_im(im, [], [], [], [], line_group)
        im = put_accumulate_num_on_im(im, only_person, count[i], movement_string)
        if show is True:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111)
            ax.imshow(im)
        if save_video:
            if not jupyter:
                im = im[:, :, ::-1]
            video_writer.write((im * 255.0).astype('uint8'))

    if save_video:
        video_writer.release()    


def run0():
    img_path="frames/clip/frame_00000020.jpg"
    time_g = 0.0
    for i in range(20):
        time_i = time.time()
        ori_imgs, framed_imgs, \
            framed_metas = input_utils.preprocess(img_path, max_size=512)
        x = torch.from_numpy(framed_imgs[0]).unsqueeze(0).permute(0, 3, 1, 2)
        if i > 4:
            time_g += (time.time() - time_i)
    print("Original Preprocess speed..................")
    print("FPS %.2f" % ((20 - 5)/time_g))
    

def run1():
    impath = "frames/clip/frame_00000020.jpg"
    time_g = 0.0
    for i in range(20):
        time_i = time.time()
        _, x, framed_metas = input_utils.preprocess_t3(impath) #, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        if i > 4:
            time_g += (time.time() - time_i)
    print("Updated Preprocess speed..................")
    print("FPS %.2f" % ((20 - 5)/time_g))
    
    
