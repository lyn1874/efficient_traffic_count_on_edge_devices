# This script contains all the utility function for visualization
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import csv
import pickle
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
                            (x1+20, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                            color, 2)
    if "d" in gt_or_pred and "pre" not in gt_or_pred:
        if gt_or_pred is "d0":
            cv2.putText(ori_img, gt_or_pred+": %d" % len(roi), (imw-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 
                        2)
        elif gt_or_pred is "d7":
            cv2.putText(ori_img, gt_or_pred+": %d" % len(roi), (imw-200, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 
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
        out_name=out_path
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
        color_move_group = [[153/255, 0, 0], [1, 0.5, 0], [0, 255/255, 128/255], [255/255, 255/255, 128/255], [1, 1, 1], 
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
                    image = cv2.arrowedLine(image, tuple([x1+ac[_direc][0], y1+ac[_direc][1]]), 
                                            tuple([x1+ac[_direc][2], y1+ac[_direc][3]]), tuple(_b_color), 2, tipLength=0.4)
            if ep[j] != 0:
                _b_color = [0, 1, 0]
            cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), 
                          color=_b_color, thickness=1)
            text_use = "%s%d" % (box_text, current_person_id[j])
            cv2.putText(image, text_use, 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, tuple(_b_color), 2)
    text_width, text_height = 140, 30  #115, 30
    for j in range(len(text)):
        text_offset_x, text_offset_y = text_coordinate[j]
        box_coords = ((text_offset_x, text_offset_y + 4), (text_offset_x + text_width + 2, text_offset_y - text_height))
        image[box_coords[1][1]:box_coords[0][1], box_coords[0][0]:box_coords[1][0]] = [192/255, 192/255, 192/255]
        text_color = [[1, 1, 1] if not show_direction_count else color_move_group[j - 1]][0]
        if j != 0:
            text_use = text[j] + ' %d' % count_group[j - 1]
            cv2.putText(image, text_use, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(text_color), 2)
        else:
            text_use = text[j]
            cv2.putText(image, text_use, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple([1, 1, 1]), 2)
    if resize:
        image = cv2.resize(image, dsize=(960, 720))
    if show is True:
        fig = plt.figure(figsize=(14,12))
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
        color_move_group = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 128/255, 0], [255/255, 255/255, 0], [0, 76/255, 153/255], [0.4, 0.3, 0.2], [0.5, 0.7, 0.9]]
    if np.sum(bbox) > 0:
        bbox_ = bbox.copy()    
        bbox_[:, :2] += (bbox_[:, 2:] - bbox_[:, :2])/2.0
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
            image = cv2.circle(image, (x_c, y_c), radius, color=tuple(color_move_group[_direc]), thickness=-1)  #what is this circle?
            text_use = "%s%d" % (box_text[j][0], count_stat[j])
            cv2.putText(image, text_use, (x_c, y_c - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        tuple(color_move_group[_direc]), 2)
            image = cv2.arrowedLine(image, tuple([x_c + ac[_direc][0], y_c + ac[_direc][1]]),
                                    tuple([x_c + ac[_direc][2], y_c+ac[_direc][3]]), 
                                    tuple(color_move_group[_direc]), 2, tipLength=0.4)
            
    return image


def put_accumulate_num_on_im(im, only_person, count_stat, movement_string):
    imw = np.shape(im)[1]
    color_move_group = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 128/255, 0], [255/255, 255/255, 0], [0, 76/255, 153/255], [0.4, 0.3, 0.2], [0.5, 0.7, 0.9]]
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
        
    text_width, text_height = 120, 30  #115, 30
    for iterr, single_cls in enumerate(c_group):
        text_coordinate = count_coord_g[iterr]
        _text = [single_cls]
        _text = np.concatenate([_text, ["%s:%d" % (text_use[q+1], count_stat[iterr][q+1]) for q in range(len(movement_string) - 1)]], axis=0)  # why only range 2        
        for j in range(len(movement_string)):
            text_offset_x, text_offset_y = text_coordinate[j]
            box_coords = ((text_offset_x, text_offset_y + 4), (text_offset_x + text_width + 2, text_offset_y - text_height))
            im[box_coords[1][1]:box_coords[0][1], box_coords[0][0]:box_coords[1][0]] = [192/255, 192/255, 192/255]
            text_color = color_move_group[j]
            cv2.putText(im, _text[j], (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(text_color), 2)
    return im


def give_input_to_vispostim(filename, camera, seq, only_person, class_group, show, save_video, use_name, algo, 
                            movement_string=None, return_count=False, 
                            return_stat=False, jupyter=True, 
                            box_standard=[], specify_direc=[], return_id_remove=False, predefine_line=[]):
    if algo is "boundary":
        if "CAM" in camera:
            line_group, counted_direc = count_utils.give_line_group_antwerp(camera, only_person)            
        elif camera == "aicity":            
            line_group, counted_direc = count_utils.give_aicity_linegroup(seq)
        else:
            line_group = predefine_line
            counted_direc = np.arange(len(line_group))
        movement_string = [["static"]]
        [movement_string.append(["move-%d-up" % i, "move-%d-down" % i]) for i in range(len(line_group)+1)[1:]]
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
        im_mom = "frames/clip/"
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
            del id_exist["id%d"%single_id]
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
        video_writer = get_video_writer(shape, filename+'%s.avi' % use_name, [])
    print("There are %d images" % num_im)
    for i in range(num_im): #[12:13]:
        _s = stat[i]
        im = cv2.imread(im_mom + _s["frame"])[:,:,::-1]/255.0
        for _citer, _cls in enumerate(class_group):
            _id = _s["count_id_%s" % _cls]
            if len(_id) > 0 and np.sum(_id) > 0:
                _ep = _s["ep_%s" % _cls][_id != 0]
                _box = np.array(_s["rois_%s" % _cls])[_id != 0][_ep != _epnum]
                _direc = _s["direction_arrow_%s" % _cls][_id != 0][_ep != _epnum]
                im = blur_image(im, _box)
                im = vis_postprocessing_im(im, _box, _id[_id!=0][_ep!= _epnum], _direc, 
                                           np.array(_s["identity_%s" % _cls])[_ep!=_epnum], line_group)
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
        
    
def _give_table_as_aicity_format(count_tab, wrong_prediction, specified_direction):
    id_ = np.array([v[1] for v in count_tab])
    unique_id_ = np.unique(id_)
    unique_id_ = unique_id_[np.argsort([float(v) for v in unique_id_])]
    unique_id_ = [i for i in unique_id_ if i != 0]
    count_tab2 = []
    for single_id in unique_id_:
        sub = np.max(np.where(id_ == single_id)[0])
        _f_index = count_tab[sub][0]
        _f_index = _f_index.split("_")[0] + "_%06d" % (int(_f_index.split("_")[1].split(".jpg")[0]) + 2)
        
        if single_id in wrong_prediction and count_tab[sub][2] == specified_direction:
            print("pass", single_id)
        else:
            count_tab2.append([_f_index , count_tab[sub][2], count_tab[sub][3]])
    frame_index = [v[0] for v in count_tab2]
    sorted_frame_index = np.argsort(frame_index)
    count_tab2 = [count_tab2[i] for i in sorted_frame_index]
    return count_tab2


def give_video_id():
    video_id_file = '/project_scratch/bo/normal_data/aic2020/AIC20_track1/Dataset_A/list_video_id.txt'
    with open(video_id_file, 'r') as f:
        content = f.readlines()
        content = [v.rstrip("\n") for v in content]
        content = [v.strip().split(" ") for v in content]
        num_num = [int(v[0]) for v in content]
        cam_str = [v[1].split(".mp4")[0] for v in content]
    f.close()
    return np.array(num_num), np.array(cam_str)


def write_to_txt(cam, student, compound_coef, algo, interval, only_person, class_group, subtract_bg, finetune_part, 
                 cls_prob, use_feature_maps, use_precalculate, activate_kalman_filter, box_standard, specify_direction=[],
                 return_path=False, camera_specific="specific_2"):
    _, json_path, video_avipath, _, txt_outpath = get_path("aicity",
                                                           cam, student, compound_coef, "offline", algo, 
                                                           interval, "test", subtract_bg, finetune_part, cls_prob,
                                                           use_feature_maps, use_precalculate, activate_kalman_filter,
                                                           cluster=6,
                                                           camera_specific=camera_specific)
    if cam == "cam_13":
        path = video_avipath.split('.avi')[0] + '/' + "use_feature_similarity_%s_boundary_car_1965" % use_feature_maps
    else:
        path = json_path + "_%s_%s" % (algo, only_person)
    if not os.path.isfile(path):
        return None
    if os.path.isfile(txt_outpath + "%s.txt" % cam):
        return txt_outpath
    required_direction, _ = give_required_direction(cam)

    _, _a, _count, wrong_prediction = give_input_to_vispostim(path, "aicity", cam+"/", only_person, class_group, False, False, None, algo, 
                                            box_standard=box_standard, specify_direc=specify_direction, 
                                            return_count=True)
    _predict_direct = [v[2] for v in _count]
    _use_iter = [iterr for iterr, v in enumerate(_predict_direct) if v in required_direction]
    _count = [_count[i] for i in _use_iter]    
    count_tab = _give_table_as_aicity_format(_count, wrong_prediction, specify_direction)
    #print(np.unique([v[1] for v in count_tab]))
    count_npy = give_desired_format(count_tab, cam, len(_a))
    savefile = txt_outpath + "%s.txt" % cam
    dst_out = open(savefile, 'w')
    results = count_npy    
    for res in results:
        res_str = " ".join(str(k) for k in res)
        print(res_str, file=dst_out)
    dst_out.close()
#     print("Vehicle counting on %s is done" % cam)
    if return_path:
        return txt_outpath
    

def give_required_direction(cam):
    ms = ["move-1-up", "move-1-down", "move-2-up", "move-2-down", "move-3-up", "move-3-down", "move-4-up", "move-4-down"]
    if cam == "cam_1" or cam == "cam_1_dawn" or cam == "cam_1_rain":
        direction_group = [ms[0], ms[1], ms[2], ms[4], ms[5]]
        direction_string = np.array(["move-1-down", "move-1-up", "move-2-up", "move-3-up-move-3-down"])
    elif cam == "cam_2" or cam == "cam_2_rain":
        direction_group = [ms[0], ms[1], ms[2], ms[3], ms[4]]
        direction_string = np.array(["move-1-down", "move-1-up", "move-2-down-move-2-up", "move-3-up"])
    elif cam == "cam_3" or cam == "cam_3_rain":
        direction_group = [ms[0], ms[1], ms[3], ms[4]]
        direction_string = np.array(["move-1-down", "move-1-up", "move-2-down", "move-3-up"])
    elif cam == "cam_10":
        direction_group = [ms[1], ms[3], ms[5]]
        direction_string = np.array(["move-1-down", "move-2-down", "move-3-down"])
    elif cam == "cam_11":
        direction_group = [ms[1], ms[3], ms[5]]
        direction_string = np.array(["move-1-down", "move-2-down", "move-3-down"])
    elif cam == "cam_12":
        direction_group = [ms[1], ms[3], ms[5]]
        direction_string = np.array(["move-1-down", "move-2-down", "move-3-down"])
    elif cam == "cam_13":
        direction_group = [ms[0], ms[3], ms[5]]
        direction_string = np.array(["move-1-up", "move-2-down", "move-3-down"])
    elif cam in ["cam_14", "cam_15", "cam_16", "cam_17", "cam_18", "cam_19", "cam_20"]:
        direction_group = [ms[0], ms[1]]
        direction_string = np.array(["move-1-down", "move-1-up"])
    elif cam == "cam_8":
        direction_group = [ms[0], ms[1], ms[2], ms[3], ms[4], ms[7]]
        direction_string = np.array(["move-1-down", "move-2-down", "move-3-up", "move-2-up", "move-4-down", "move-1-up"])
    else:
        print("the direction for camera %s is not defined" % cam)
    return direction_group, direction_string

    
def give_desired_format(count_tab, cam, num_frame):
    num_num, cam_str = give_video_id()
    #direct_string = np.array(["down", "up"])
    _, direct_string = give_required_direction(cam)
    class_group = np.array(["car", "truck"])
    count_npy = np.zeros([len(count_tab), 4], dtype=np.int32)
    specific_id = num_num[np.where(cam_str == cam)[0]]
    for iterr, single in enumerate(count_tab):
        count_npy[iterr, 0] = specific_id
        _l = int(single[0].strip().split("_")[1])
        if _l >= num_frame:
            _l = num_frame - 1
        count_npy[iterr, 1] = _l
#         count_npy[iterr, 1] = int(single[0].strip().split("_")[1])
        count_npy[iterr, 2] = [iterr for iterr, v in enumerate(direct_string) if single[1] in v][0] + 1
#         count_npy[iterr, 2] = np.where(single[1] in direction_string)[0]+1
        count_npy[iterr, 3] = np.where(class_group == single[2])[0] + 1
    return count_npy
        
            

# -------------------------------------------------------------------------------------------------- #
# ------------- Give the json filename based on the url--------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #


def get_urls():
    urls = ["rtsp://143.169.169.3:1935/smartcity/CAM1.1-Groenplaats.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM1.2-Groenplaats.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM1.3-Groenplaats.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM1.4-Groenplaats.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM2.1-Hoogstraat-Reynderstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM2.2-Hoogstraat-Reynderstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM2.3-Hoogstraat-Reynderstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM2.4-Hoogstraat-Reynderstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM3.1-Hoogstraat-St.Jansvliet.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM3.2-Hoogstraat-St.Jansvliet.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM3.3-Hoogstraat-St.Jansvliet.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM3.4-Hoogstraat-St.Jansvliet.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM4.1-Kloosterstraat-Kromme-elleboogstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM4.2-Kloosterstraat-Kromme-elleboogstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM4.3-Kloosterstraat-Kromme-elleboogstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM4.4-Kloosterstraat-Kromme-elleboogstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM5.1-Kloosterstraat-KorteVlierstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM5.2-Kloosterstraat-KorteVlierstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM5.3-Kloosterstraat-KorteVlierstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM5.4-Kloosterstraat-KorteVlierstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM6.1-Andriesplaats.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM6.2-Andriesplaats.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM6.3-Andriesplaats.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM6.4-Andriesplaats.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM7.1-Kloosterstraat-Riemstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM7.2-Kloosterstraat-Riemstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM7.3-Kloosterstraat-Riemstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM7.4-Kloosterstraat-Riemstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM8.1-Kloosterstraat-Scheldestraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM8.2-Kloosterstraat-Scheldestraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM8.3-Kloosterstraat-Scheldestraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM8.4-Kloosterstraat-Scheldestraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM9.1-LeopoldDeWaelstraat-Schildersstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM9.2-LeopoldDeWaelstraat-Schildersstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM9.3-LeopoldDeWaelstraat-Schildersstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM9.4-LeopoldDeWaelstraat-Schildersstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM10.1-Nationalestraat-Kronenburgstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM10.2-Nationalestraat-Kronenburgstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM10.3-Nationalestraat-Kronenburgstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM10.4-Nationalestraat-Kronenburgstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM11.1-Nationalestraat-Prekerstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM11.2-Nationalestraat-Prekerstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM11.3-Nationalestraat-Prekerstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM11.4-Nationalestraat-Prekerstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM12.1-Nationalestraat-Ijzerwaag.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM12.2-Nationalestraat-Ijzerwaag.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM12.3-Nationalestraat-Ijzerwaag.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM12.4-Nationalestraat-Ijzerwaag.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM13.1-Nationalestraat-Steenhouwersvest.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM13.2-Nationalestraat-Steenhouwersvest.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM13.3-Nationalestraat-Steenhouwersvest.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM13.4-Nationalestraat-Steenhouwersvest.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM14.1-Kloosterstraat-WillemLepelstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM14.2-Kloosterstraat-WillemLepelstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM14.3-Kloosterstraat-WillemLepelstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM14.4-Kloosterstraat-WillemLepelstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM15.1-Kronenburgstraat-volkstraat.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM16.1-StAndriesplaats.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM17.1-StAndriesplaats.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM18.1-StAndriesplaats-costa.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM18.2-StAndriesplaats-costa.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM18.3-StAndriesplaats-costa.stream",
            "rtsp://143.169.169.3:1935/smartcity/CAM18.4-StAndriesplaats-costa.stream"]
    return urls


def get_path(camera, seq, student, compound_coef, on_off, angle_line, interval, tr_or_tt="train", subtract_bg=False, finetune_part="head", 
             cls_prob=0.5, use_feature_maps=True, use_precalculate=False, activate_kalman_filter=True, cluster=4, camera_specific="specific"):
    if camera is not "aicity":
        urls = get_urls()
        url = [v for v in urls if '.'.join(camera.split('_'))+'-' in v][0]
        mom_path_string = "/home/jovyan/bo/"
        im_mom = '/home/jovyan/bo/dataset/%s/%s/' % (camera, seq)
    else:
        url = "aicity2020"
        mom_path_string = "/project/bo/"
        if cluster == 6:
            im_mom = "/project_scratch/bo/normal_data/aic2020/AIC20_track1/Dataset_A_Frame/%s/" % seq
        elif cluster == 4:
            im_mom = "/tmp/bo/normal_data/aic2020/AIC20_track1/Dataset_A_Frame/%s/" % seq
        
    path_shortcut = ["%sDet_student_subtractBG_%s_finetune_%s_clsprob_%.2f" % (camera, subtract_bg,
                                                                               finetune_part, cls_prob) if student else "%sDet_teacher_precalc_%s" % (camera, 
                                                                                                                                                      use_precalculate)][0]
    if not use_feature_maps:
        path_shortcut += "_wsimilarity_%s" % use_feature_maps
    if not activate_kalman_filter:
        path_shortcut += "_wkalman_%s" % activate_kalman_filter
    
    json_path = mom_path_string + 'exp_data/counting_result_%s/%s/track_result_%d_gap%d/' % (camera_specific, path_shortcut, compound_coef, interval)
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    json_filename = json_path + "%s_prediction" % seq
    video_outpath = mom_path_string + 'exp_data/counting_result_%s/%s/videos_%d_gap%d/' % (camera_specific, path_shortcut, compound_coef, interval)
    txt_outpath = mom_path_string + "exp_data/counting_result_%s/%s/result_txt_%d_gap%d/" % (camera_specific, path_shortcut, compound_coef, interval)
    if not os.path.exists(video_outpath):
        os.makedirs(video_outpath)
    if not os.path.exists(txt_outpath):
        os.makedirs(txt_outpath)
    name = "iou_sim_kf_%s_d%d_filter_boxes_%s_%s" % (seq, compound_coef, on_off, angle_line)
    csv_filename = json_path + url.strip().split('/')[-1].strip().split('.stream')[0] + "_%s" % seq + '.csv'
    if tr_or_tt is "train":
        print("camera name........................", camera)
        print("date................................", seq)
        print("track statistics are saved at.......\n", json_filename)
        print("videos are saved at.................\n", video_outpath+name)
        print("csv file will be saved at...........\n", csv_filename)

    return im_mom, json_filename, video_outpath + name + '.avi', csv_filename, txt_outpath


#------------------------------------------------------------------------------------------------------#
#----------------------------------Write the results into csv file-------------------------------------#
#------------------------------------------------------------------------------------------------------#

def sorted_stat(path):
    stat = pickle.load(open(path, 'rb'))
    aggre_count = {}
    aggre_count["pedestrian"] = np.zeros([len(stat)- 1, 5])
    aggre_count["cyclist"] = np.zeros([len(stat) - 1, 5])
    aggre_count["car"] = np.zeros([len(stat) - 1, 5])
    aggre_count["time"] = []
    aggre_count["frame"] = []
    car_string = "aggregate_movement_car"
#     car_string = ["count_movement_car" if "angle" in path else "aggregate_movement_car"][0]
    with_iter = [iterr for iterr, v in enumerate(stat[:-1]) if "aggregate_movement_pedestrian" in v.keys()]
    for i in range(len(stat) - 1):
        if i in with_iter:
            aggre_count["pedestrian"][i] = stat[i]["aggregate_movement_pedestrian"]
            aggre_count["cyclist"][i] = stat[i]["aggregate_movement_cyclists"]
        else:
            aggre_count["pedestrian"][i] = aggre_count["pedestrian"][i - 1]
            aggre_count["cyclist"][i] = aggre_count["cyclist"][i - 1]
        aggre_count["car"][i] = stat[i][car_string]
        aggre_count["time"].append(stat[i]["time"])
        aggre_count["frame"].append(stat[i]["frame"])
    return aggre_count


def get_count_every_k_min(k, stat, class_group):
    """This function is used to get the count every k minutes
    stat: a dict, with key 'frameid', 'current count', 'aggregate count', 'current direction count', 
        'aggregated direction count', 'current_time'
    k: int, defines the minute we want
    """
    time_per_frame = stat['time']
    time_int = []
    for iterr, single_time in enumerate(time_per_frame):
        if ' ' in single_time:
            single_time = single_time.strip().split(' ')[3]
            time_per_frame[iterr] = single_time
        time_int.append([int(v) for v in single_time.strip().split(':')])
    time_int = np.array(time_int)
    time_single = [v[0] * 60 + v[1] + v[2]/60 for v in time_int]
    diff = np.diff(time_single, axis=0)
    _cum = np.round(np.cumsum(diff, axis=0))
    count_sort = {}
    count_sort["time"] = [time_per_frame[0]]
    for iterr, single_class in enumerate(class_group):
        count_sort["count_%s" % single_class] = [stat[single_class][0, 1:3]]
    for i in range(int(_cum[-1]/k)+1)[1:]:
        _iterr = np.where(_cum == i * k)[0][0]
        count_sort["time"].append(time_per_frame[_iterr])
        for iterr, single_class in enumerate(class_group):
            count_sort["count_%s" % single_class].append(stat[single_class][_iterr, 1:3])
    for single_class in class_group:
        count_sort["count_%s" % single_class] = np.vstack(count_sort["count_%s" % single_class])
        count_sort["count_%s" % single_class] = np.vstack([[count_sort["count_%s" % single_class][0]], np.diff(count_sort["count_%s" % single_class], axis=0)])
    return count_sort


def arrange_time(_time):
    sp = _time.split(':')
    sp[0] = '%02d' % (int(sp[0]) + 2)
    _time = ':'.join(sp)
    return _time


def write_detection_accuracy_csv(accuracy, csv_file, arranged=False):
    if not arranged:
        a = np.expand_dims(accuracy["ckpt_step"], axis=1)
        b = np.array(accuracy["AP"])
        c = np.array(accuracy["AR"])
        d = np.hstack([a, b, c])
    else:
        d = accuracy
    row_name = ["ckpt_step", "AP 0.5:0.95", "AP 0.5", "AP 0.75", "AP 0.5:0.95 S", "AP 0.5:0.95 M", "AP 0.5:0.95 L",
                "AR 0.5:0.95", "AR 0.5", "AR 0.75", "AR 0.5:0.95 S", "AR 0.5:0.95 M", "AR 0.5:0.95 L"]
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_name)
        for i in range(len(d)):
            writer.writerow(d[i])
    
    
def write_aggregate_detection_accuracy(path_mom):
    """This function writes the detection accuracy for each camera in a single sheet
    path_mom: /../../../tds/../
    """
    camera = ["1_rain",  "1_dawn", "2", "2_rain", "3", "3_rain", "11", "12", "13",
          "14", "15", "16", "17", "18", "19", "20"]
    camera = ["cam_%s" % v for v in camera]
    aggre_stat = []

    for s_c in camera:
        value, name = vt.load_csv(path_mom + s_c + '_all.csv')
        value = np.array(value).astype('float32')
        max_value = value[np.argmax(value, axis=0)[1],:]
        aggre_stat.append(max_value)

    csv_file = path_mom + "aggregated.csv"
    vt.write_detection_accuracy_csv(aggre_stat, csv_file, True)
    

def write_csv_file(json_filename, csv_filename, k, datetime_or_ostime, class_group, algo, save=False):
    stat = sorted_stat(json_filename)
    
    stat = get_count_every_k_min(k, stat, class_group)
    if not csv_filename:
        return stat
    else:
        csv_filename = csv_filename.strip().split('.csv')[0] + "%s.csv" % algo
    row_name = ["start", "end"]
    for single_class in class_group:
        row_name = np.hstack([row_name, [single_class for _ in range(len(class_group))]])
    row_name2 = [["up", "down", "count"] for _ in range(len(class_group))]
    row_name2 = np.hstack([" ", " ", [v for j in row_name2 for v in j]])
    
    _content = []
    for i in range(len(stat["time"]) - 1):
        _time_init = stat["time"][i]
        _time_end = stat["time"][i+1]
        if datetime_or_ostime is "ostime":
            _time_init = arrange_time(_time_init)
            _time_end = arrange_time(_time_end)
        _single_stat = [_time_init, _time_end]
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
    return value, name


def vis_lines(line_coord_g, im):
    if type(im) == str:
        im = cv2.imread(im)[:,:,::-1]/255.0
    line_color = [0, 1, 0]
    for single_line in line_coord_g:
        x11, y11, x12, y12 = single_line
        cv2.line(im, (x11, y11), (x12, y12), line_color, thickness=2)
    for iterr, line_coord in enumerate(line_coord_g):
        print("Line %d" % iterr, line_coord)
        x0 = sorted([line_coord[0], line_coord[2]])
        x1 = sorted([line_coord[1], line_coord[3]])
        temp_box = [x0[0], x1[0], x0[1], x1[1]]
        if line_coord[0] == line_coord[2]:
            temp_box = [line_coord[0]-100, line_coord[1], line_coord[2] + 100, line_coord[2]]
        elif line_coord[1] == line_coord[3]:
            temp_box = [line_coord[0], line_coord[1] - 100, line_coord[2], line_coord[3] + 100]
        if iterr < len(line_coord_g) - 1:
            show=False
        else:
            show=True
        im = show_bbox(im, np.array([temp_box]), np.array([1, 2, 3, 4]), "pred", show=show)
    
    
                
def draw_standard_moveline(path, move_, _im=[]):
    if np.sum(_im) == 0:
        _im = cv2.imread(path[0])[:,:,::-1]
    value = 4
    for iterr, single_coord in enumerate(move_["left_3"]):
        single_coord = np.array(single_coord).astype(np.int32)
        if single_coord[1] - value < 0:
            h_l = 0
        else:
            h_l = single_coord[1] - value
        if single_coord[0] - value < 0:
            w_l = 0
        else:
            w_l = single_coord[0] - value
        if iterr == 0:
            color = [0, 255, 0]
        else:
            color = [255, 0, 0]
        _im[h_l:single_coord[1]+value, w_l:single_coord[0]+value, :] = color
    plt.imshow(_im)
    
    
def vis_mask(pt, im_, save_name, return_mask=False):
    im = im_.copy()
    pt = np.array(pt)
    mask = np.zeros(np.shape(im)).astype('uint8')
    pts = pt
    pts = pts.reshape((-1,1,2))
    mask = cv2.fillPoly(mask, [pts], color=(255, 255, 255))
    mask = np.sum(mask, axis=-1, keepdims=-1)
    mask = (mask != 0).astype('int32')
    im[:, :, 0] = im[:, :, 0] * (mask[:,:,0] + 0.2)
    plt.imshow(im)
    if save_name:
        np.save('/home/jovyan/bo/dataset/%s' % save_name, mask)
    if return_mask:
        return mask
    
    
def give_cam():
    cam_group = ["cam_1_dawn", "cam_1_rain", "cam_2", "cam_2_rain", "cam_3", "cam_3_rain", "cam_8", "cam_10",
             "cam_11", "cam_12", "cam_13", "cam_14", "cam_15", "cam_16", "cam_17", "cam_18", "cam_19", "cam_20"]
    return cam_group