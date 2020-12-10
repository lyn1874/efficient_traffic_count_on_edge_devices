# This script contains all the utility function for visualization
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import csv
import pickle
import time
from efficientdet.postprocess_count_utils import *
import efficientdet.count_utils as count_utils

    
def ax_global_get(fig):
    ax_global = fig.add_subplot(111, frameon=False)
    ax_global.spines['top'].set_color('none')
    ax_global.spines['bottom'].set_color('none')
    ax_global.spines['left'].set_color('none')
    ax_global.spines['right'].set_color('none')
    ax_global.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    return ax_global


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
        fps_use = 24.0
    else:
        fps_use = 24.0
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
    if only_person == "person":
        c_group = ["person", "bike"]
        count_coord_p = [[p_left, v] for v in count_loc]
        count_coord_b = [[b_left, v] for v in count_loc]
        count_coord_g = [count_coord_p, count_coord_b]
    elif only_person == "car":
        c_group = ["car"]
        count_coord_c = [[c_left, v] for v in count_loc]
        count_coord_g = [count_coord_c]
    elif only_person == "ped_car":
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
                            im_mom=None, segment=[]):
    if algo == "boundary":
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
    if camera == "aicity":
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
        single_time = stat_original[0]['time']
        q = single_time.split(' ')
        read_date = '-'.join((q[1], q[2], q[-1]))
        return count, time_use, count_table_frameindex, read_date
    if only_person is "person":
        count = count[:, :2]
    elif only_person is "car":
        count = count[:, -1:]
    num_im = len(stat) - 1
    shape = tuple(np.shape(cv2.imread(im_mom + stat_original[0]["frame"]))[:-1])
    _epnum = 1
    if len(segment) > 0:
        print(time.ctime(os.path.getmtime(im_mom + stat[segment[0]]['frame'])))
        print(time.ctime(os.path.getmtime(im_mom + stat[segment[1]]['frame'])))
    if len(segment) > 0:
        interval = range(num_im)[segment[0]:segment[1]]
        use_str = "%s_%s" % (time.ctime(os.path.getmtime(im_mom + stat[segment[0]]['frame'])), 
                             time.ctime(os.path.getmtime(im_mom + stat[segment[1]]['frame'])))
    else:
        interval = range(num_im)
        use_str = use_name    
    if save_video:
        video_writer = get_video_writer(shape, filename + '%s.avi' % use_str, [])
    print("There are %d images" % num_im)
    for i in interval:  # [12:13]:
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
        


def sorted_stat(path, camera, date, only_person, class_group, box_standard, specify_direc, filter_id):    
    count_num, time_use, _, read_date = give_input_to_vispostim(path, camera, date, only_person, 
                                                     class_group, False, False, "ok", "boundary", return_count=True,
                                                     box_standard=box_standard, 
                                                     specify_direc=specify_direc, return_id_remove=filter_id)
    aggre_count = {}
    aggre_count["time"] = np.array(time_use)
    aggre_count["pedestrian"] = count_num[:, 0, :]
    aggre_count["cyclist"] = count_num[:, 1, :]
    aggre_count["car"] = count_num[:, 2, :]
    return aggre_count, read_date



def get_count_every_k_min(k, stat, class_group, input_date):
    """This function is used to get the count every k minutes
    stat: a dict, with key 'frameid', 'current count', 'aggregate count', 'current direction count', 
        'aggregated direction count', 'current_time'
    k: int, defines the minute we want
    """
    time_int = []
    date_group = []
    second = np.array([v.split(':')[-1] for v in stat['time']])
    index = np.where(second.astype('int32') <= 2)[0][0]
    for single_key in stat.keys():
        stat[single_key] = stat[single_key][index:]
    time_per_frame = stat['time']
    hour = np.array([v.split(':')[0] for v in stat['time']])
    transfer = "00" in hour and "23" in hour
    for iterr, single_time in enumerate(time_per_frame):
        if ' ' in single_time:
            single_time = single_time.strip().split(' ')[3]
            time_per_frame[iterr] = single_time
            q = single_time.split(' ')
            date = '-'.join((q[1], q[2], q[-1]))
        else:
            if transfer == False:
                date = input_date
            else:
                if "23" in single_time.split(":")[:1]:
                    date = input_date
                elif "00" in single_time.split(":")[:1]:
                    date = '-'.join((input_date.split('-')[0], str(int(input_date.split('-')[1]) + 1),
                                     input_date.split('-')[-1]))
        time_int.append([int(v) for v in single_time.strip().split(':')])
        date_group.append(date)
    time_int = np.array(time_int)
    date_group = np.array(date_group)
    if transfer == False:
        time_single = [v[0] * 60 + v[1] + v[2]/60 for v in time_int]
    else:
        time_single = [v[0] * 60 + v[1] + v[2]/60 if v[0] == 23 else (v[0] + 24) * 60 + v[1] + v[2]/60 for v in time_int]
    diff = np.diff(time_single, axis=0)
    _cum = np.cumsum(diff, axis=0)
    count_sort = {}
    count_sort["time"] = [time_per_frame[0]]
    count_sort["date"] = [date_group[0]]
    for iterr, single_class in enumerate(class_group):
        count_sort["count_%s" % single_class] = [stat[single_class][0, 1:3]]  # static, move-1-up, move-1-down
    for i in range(int(_cum[-1]/k)+1)[1:]:
#     for i in range(int(np.ceil(_cum[-1]/k))+1)[1:]:
        _iterr = np.where(_cum == i * k)[0][0]
        count_sort["time"].append(time_per_frame[_iterr])
        count_sort["date"].append(date_group[_iterr])
        for iterr, single_class in enumerate(class_group):
            count_sort["count_%s" % single_class].append(stat[single_class][_iterr, 1:3])
    if np.max(_cum) - (i * k) > 0.95:
        count_sort["time"].append(time_per_frame[-1])
        count_sort["date"].append(date_group[-1])
        for iterr, single_class in enumerate(class_group):
            count_sort["count_%s" % single_class].append(stat[single_class][-1, 1:3])
        
        
    for single_class in class_group:
        count_sort["count_%s" % single_class] = np.vstack(count_sort["count_%s" % single_class])
        count_sort["count_%s" % single_class] = np.vstack([
            [count_sort["count_%s" % single_class][0]], np.diff(count_sort["count_%s" % single_class], axis=0)])
    return count_sort


def arrange_time(_time):
    sp = _time.split(':')
    sp[0] = '%02d' % (int(sp[0]) + 2)
    _time = ':'.join(sp)
    return _time


def fix_error(count_table_stat, class_group):
    for s in class_group:
        value = count_table_stat[s]
        diff = np.diff(value, axis=0)
        if len(np.where(diff[:, 1] < 0)[0]) > 0 or len(np.where(diff[:, 2] < 0)[0])>0:
#             print("fix error for class %s" % s)
            stat = value.copy()
            for j in [1, 2]:
                for iterr, _va in enumerate(stat[:-1, j]):
                    if stat[iterr + 1, j] < _va:
                        stat[iterr+1, j] = _va
            count_table_stat[s] = stat
    return count_table_stat


def write_csv_file(json_filename, only_person, k, datetime_or_ostime, 
                   class_group, algo, num_movement, box_standard, 
                   specify_direc, count_path, remove_id=False, save=False, return_stat=False):
    only_person="ped_car"
    camera = json_filename.split('/')[-4]
    date = json_filename.split('/')[-3]
    csv_filename = count_path + "/%s.csv" % json_filename.split('/')[-2]
#     csv_filename = "/".join(json_filename.split('/')[:-2]) + "/count_statistics/%s.csv" % json_filename.split('/')[-2]
    if only_person == "ped_car":
        stat_class_group = ["person", "car"]
    elif only_person == "car":
        stat_class_group = ["car"]
    elif only_person == "ped":
        stat_class_group = ["person"]    
    stat, input_date = sorted_stat(json_filename, camera, date, only_person, stat_class_group,
                       box_standard, specify_direc, remove_id)
    stat = fix_error(stat, class_group)
    print(stat["time"][-1])
    stat = get_count_every_k_min(k, stat, class_group, input_date)
    if return_stat == True:
        return stat
    row_name = ["Date", "start", "end"]
    for single_class in class_group:
        row_name = np.hstack([row_name, [single_class for _ in range(num_movement)]])
    row_name2 = [["up", "down", "count"] for _ in range(len(class_group))]
    row_name2 = np.hstack([" ", " ", " ", [v for j in row_name2 for v in j]])
    _content = []
    for i in range(len(stat["time"]) - 1):
        _time_init = stat["time"][i]
        _time_end = stat["time"][i+1]
        _single_stat = [stat["date"][i], _time_init, _time_end]
        for single_class in class_group:
            _value = stat["count_%s" % single_class][i+1]
            _single_stat = np.hstack([_single_stat, _value, np.sum(_value)])
        _content.append(_single_stat)
    if save:
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)    
            writer.writerow(row_name)
            writer.writerow(row_name2)
            for i in range(len(stat["time"]) - 1):
                writer.writerow(_content[i])
    else:
        print(np.shape(row_name), np.shape(row_name2), np.shape(_content))
        return np.vstack([np.expand_dims(row_name, 0), np.expand_dims(row_name2, 0), _content])
    

def load_csv(filename, load_name=False):
    value = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')        
        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                name = row       
                if load_name:
                    return name
            else:
                value.append(row)
                line_count += 1
#         print(f'Processed {line_count} lines.')
    return np.array(value), np.array(name)


def save_csv(csv_filename, row_name, value):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)    
        writer.writerow(row_name)
        for i in range(len(value)):
            writer.writerow(value[i])
        
    
def get_aggregate(csv_folder, savepath, savename, return_stat=False):
    file = []
    for s_fold in csv_folder:
        _file = [s_fold + v for v in sorted(os.listdir(s_fold)) if '.csv' in v]
        file.append(_file)
    file = [v for j in file for v in j]
#     file = [csv_folder + v for v in sorted(os.listdir(csv_folder)) if '.csv' in v]
    value_group = []
    for i, single_file in enumerate(file):
        value, name = load_csv(single_file)
        if i != 0:
            value = value[1:]
        value_group.append(value)
    value_group = np.array([v for j in value_group for v in j])
    date = value_group[1:, 0]
    date_num = np.zeros([len(value_group), 1])
    for q_iter, q in enumerate(np.unique(date)):
        date_num[np.where(value_group[:, 0] == q)[0], :] = q_iter
    value_group = np.concatenate([date_num, value_group], axis=-1)
    name = np.concatenate([["index"], name], axis=0)
    csv_new = savepath + "/%s.csv" % savename
    save_csv(csv_new, name, value_group)
    if return_stat:
        return name, value_group
    

def arrange_gt(camera, readdate, savedate, load_hour):
    path = "/home/jovyan/bo/dataset/%s/%s.csv" % (camera, readdate)
    value, name = load_csv(path)
    value = np.array(value)
    index = np.arange(len(value))
    index = np.reshape(index, [-1, 4])

    time_suppose = ['%02d:00:00' % i for i in range(24)]
    time_suppose.append("00:00:00")
    time_init = value[index[:,0], 2]
    date_read = [v.strip().split(' ')[0:1] for v in time_init]
    time_init = [v.strip().split(' ')[-1].split('+')[:1] for v in time_init]
    count = value[:, 3]
    count_per_class = np.reshape(count, [-1, 4]).astype('float32') # cyclist, pedestrian, car, truck
    obj = {}
    row_name = ["Date", "Start Time", "End Time", "Cyclist", "Pedestrian", "Car", "Truck"]
    if load_hour is True:
        num_hour = len(date_read) // 12
        value = []
        for i in range(num_hour-1):
            init_time = time_init[i * 12]
            end_time = time_init[(i+1) * 12]
            _date = date_read[(i+1) * 12 - 1]
            _c = np.sum(count_per_class[i*12:(i+1)*12, :], axis=0)
            value.append(np.hstack([_date, init_time, end_time, _c]))
    else:
        value = np.hstack([date_read, time_init, time_init, count_per_class])

    obj["stat"]=[]
    value = np.array(value)
    for j, every_time in enumerate(time_suppose[:-1]):
        if every_time in value[:, 1]:
            l = np.where(value[:, 1] == every_time)[0][0]
            obj["stat"].append(value[l])
        else:
            obj["stat"].append([date_read[0][0], every_time, time_suppose[j+1], 0, 0, 0, 0])

    obj["stat"] = np.array(obj["stat"])
    save_object = {}
    save_object["date"] = obj["stat"][0, 0]
    save_object["time"] = obj["stat"][:, 1]
    save_object["Cyclist"] = obj["stat"][:, 3].astype('float32')
    save_object["Pedestrian"] = obj["stat"][:, 4].astype('float32')
    save_object["Car"] = obj["stat"][:, 5].astype('float32')

    csv_name = path.split('/2020')[0] + "/%s.csv" % savedate 
    save_csv(csv_name, row_name, value)
    pickle.dump(save_object, open(csv_name.split(".csv")[0] + ".obj", 'wb'))
    
    
def arrange_pred(camera, k):
    csv_folder = "Results/CountStatistics/%s/" % camera
    csv_folder = [csv_folder + v + "/" for v in sorted(os.listdir(csv_folder)) if '_' in v]
    savedir = "/home/jovyan/bo/dataset/%s/" % camera
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    name, value_g = get_aggregate(csv_folder, savedir, "pred_count", True)
    minu = np.array([v.strip().split(":")[1] for v in value_g[1:, 2]])
    index = np.where(minu == "00")[0][0]
    value_g = value_g[index+1:, :]
    time_init = [v.split(":") for v in value_g[:, 2]]
    index_ = (value_g[:, 0]).astype('float32')
    time_num = np.zeros([len(index_)])
    for i, s_t in enumerate(time_init):
        _va = (float(s_t[0]) + (index_[i] * 24)) * 60 + float(s_t[1]) + float(s_t[2])/60
        time_num[i] = _va
    time_diff = np.diff(time_num, axis=0)
    _cum = np.cumsum(time_diff, axis=0)
    accu_count = np.cumsum(value_g[:, 4:].astype('float32'), axis=0)
    count_sort = {}
    count_sort["time"] = [value_g[0, 2]]
    count_sort["date"] = [value_g[0, 1]]
    count_sort["count"] = [value_g[0, 4:]]
    k = 60
    for i in range(int(_cum[-1]/k)+1)[1:]:
        _iterr = np.where(_cum == i * k)[0]
        if len(_iterr) > 0:
            _iterr = _iterr[0]
        else:
            _iterr = np.argsort(abs(_cum - i * k))[1]
        count_sort["time"].append(value_g[_iterr, 2])
        count_sort["date"].append(value_g[_iterr, 1])
        count_sort["count"].append(accu_count[_iterr])

    count_sort["count"] = np.array(count_sort["count"])
    count_sort["count"][1:] = np.diff(count_sort["count"].astype('float32'), axis=0)
    save_stat = []
    for i in range(len(count_sort["time"]) - 1):
        init_time = count_sort["time"][i]
        end_time = count_sort["time"][i+1]
        _count = count_sort["count"][i+1]
        _date = count_sort["date"][i+1]
        save_stat.append(np.hstack([_date, init_time, end_time, _count]))
    
    num_lost = int(count_sort["time"][1].split(":")[0])
    time_lost = ["%02d:00:00" % i for i in range(num_lost)]
    save_object = {}
    
    save_object["date"] = np.hstack([[count_sort["date"][0] for _ in range(num_lost)], count_sort["date"][1:]])
    save_object["time"] = np.hstack([time_lost, count_sort["time"][:-1]])
    save_object["Cyclist"] = np.hstack([[0 for _ in range(num_lost)], count_sort["count"][1:, 5]]).astype('float32')
    save_object["Pedestrian"] = np.hstack([[0 for _ in range(num_lost)], count_sort["count"][1:, 2]]).astype('float32')
    save_object["Car"] = np.hstack([[0 for _ in range(num_lost)], count_sort["count"][1:, -1]]).astype('float32')
    pickle.dump(save_object, open(savedir + "pred_count.obj", 'wb'))
    namecsv = np.hstack([["Date", "Start Time", "End Time"],name[-9:]])
    save_csv(savedir + "pred_count.csv", namecsv, save_stat)
    

def write_detection_accuracy_csv(accuracy, csv_file):
    a = np.expand_dims(accuracy["ckpt_step"], axis=1)
    b = np.array(accuracy["AP"])
    c = np.array(accuracy["AR"])
    d = np.hstack([a, b, c])
    row_name = ["ckpt_step", "AP 0.5:0.95", "AP 0.5", "AP 0.75", "AP 0.5:0.95 S", "AP 0.5:0.95 M", "AP 0.5:0.95 L",
                "AR 0.5:0.95", "AR 0.5", "AR 0.75", "AR 0.5:0.95 S", "AR 0.5:0.95 M", "AR 0.5:0.95 L"]
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_name)
        for i in range(len(d)):
            writer.writerow(d[i])
    


    
        