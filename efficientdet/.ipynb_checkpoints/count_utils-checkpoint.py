import numpy as np
import matplotlib.pyplot as plt
import cv2


def calculate_percentage(box, line):
    """Caclulate the percentage above the line, and below the line
    box: [x1, y1, x2, y2]
    line: [a, b]
    """
    pixel_right_or_up = 0
    for ix, x in enumerate(range(int(box[2]))[int(box[0]):]):
        for iy, y in enumerate(range(int(box[3]))[int(box[1]):]):
            if x * line[0] + line[1] - y > 0:
                pixel_right_or_up += 1
    percent = (int(box[3]) - int(box[1])) * (int(box[2]) - int(box[0]))
    percent = pixel_right_or_up / percent
    return percent


def test_assign_bbox_to_border():
    line1_coord = [500, 650, 20, 300]
    line2_coord = [800, 100, 750, 50]
    line3_coord = [1100, 280, 1250, 200]
    mask = np.zerso([1080, 1920, 3])
    # mask = cv2.imread(
    #     '/project_scratch/bo/normal_data/aic2020/AIC20_track1/screen_shot_with_roi_and_movement/cam_2.jpg')[:, :,
    #        ::-1] / 255.0
    region_0_color = [1, 0, 0]
    region_1_color = [0, 0, 1]
    color_group = [[0, 1, 0], region_0_color, region_1_color, [1, 1, 1]]
    line_group = [line1_coord, line2_coord, line3_coord]
    for l_iter, single_line in enumerate(line_group):
        if len(single_line) > 0:
            x11, y11, x12, y12 = single_line
            cv2.line(mask, (x11, y11), (x12, y12), color_group[l_iter], thickness=2)
    boxes_group = np.array([[157.3077, 354.80957, 483.7765, 610.5921],
                            [296.7928, 296.06042, 367.18036, 340.24933],
                            [580.0094, 262.40854, 638.5357, 299.55234],
                            [605.12244, 165.15363, 649.6768, 195.85562],
                            [612.90735, 172.48688, 662.3439, 208.46062],
                            [910.89484, 146.71783, 948.64185, 175.83723],
                            [516.5664, 100.74517, 559.84467, 122.77205],
                            [822.569, 160.62735, 861.25464, 188.1642],
                            [894.79913, 130.89241, 930.5981, 157.19673],
                            [1125.457, 176.51405, 1165.0243, 211.689],
                            [1051.5833, 49.311363, 1080.1599, 71.18177]],
                           dtype=np.int32)

    box_to_border, percent_to_border = assign_bbox_to_border_single_line(boxes_group, line_group, 0.6)
    for i in range(len(boxes_group)):
        (x1, y1, x2, y2) = boxes_group[i]
        c_use = color_group[int(np.max(box_to_border[i]))]
        text_use = "id %d R%d" % (i + 1, np.max(box_to_border[i]))
        cv2.rectangle(mask, pt1=(x1, y1), pt2=(x2, y2),
                      color=c_use, thickness=3)
        cv2.putText(mask, text_use,
                    (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, tuple(c_use), 2)
    fig = plt.figure(figsize=(17, 12))
    ax = fig.add_subplot(111)
    ax.imshow(mask)


def give_aicity_linegroup(camera_name):
    if "cam_16/" in camera_name:
        line_group = [[0, 800, 1919, 800]]
        counted_direction = np.array([0])
    elif "cam_2/" in camera_name or "cam_2_rain" in camera_name:
        line_group = [[500, 650, 20, 300], [200, 170, 149, 50], [1000, 310, 1279, 270]]
        counted_direction = np.array([0, 1, 2])
    elif "cam_14/" in camera_name:
        line_group = [[0, 750, 2050, 1919]]
        counted_direction = np.array([0])
    elif "cam_15/" in camera_name:
        line_group = [[750, 800, 1500, 300]]
        counted_direction = np.array([0])
    elif "cam_17/" in camera_name:
        line_group = [[600, 450, 1919, 450]]
        counted_direction = np.array([0])
    elif "cam_18/" in camera_name:
        line_group = [[0, 750, 1300, 850]]
        counted_direction = np.array([0])
    elif "cam_19/" in camera_name:
        line_group = [[400, 1000, 1500, 1000]]
        counted_direction = np.array([0])
    elif "cam_20/" in camera_name:
        line_group = [[0, 650, 1500, 810]]
        counted_direction = np.array([0])
    elif "cam_1/" in camera_name or "cam_1_dawn" in camera_name:
        line_group = [[300, 400, 249, 200], [800, 450, 900, 350], [500, 600, 600, 430]]
        counted_direction = np.array([0, 1, 2])
    elif "cam_1_rain/" in camera_name:
        line_group = [[150, 400, 100, 200], [800, 450, 900, 350], [500, 600, 600, 430]]
        counted_direction = np.array([0, 1, 2])
    elif "cam_3_rain" in camera_name:
        line_group = [[460, 400, 300, 200], [450, 550, 510, 410], [620, 430, 750, 400]]
        counted_direction = np.array([0, 1, 2])
    elif "cam_3/" in camera_name:
        line_group = [[380, 420, 300, 300], [450, 550, 500, 450], [620, 430, 750, 400]]
        counted_direction = np.array([0, 1, 2])
    elif "cam_10/" in camera_name:
        line_group = [[270, 570, 100, 380], [830, 700, 220, 700], [1000, 800, 1500, 600]]
        counted_direction = np.array([0, 1, 2])
    elif "cam_11/" in camera_name:
        line_group = [[400, 440, 0, 350], [800, 700, 240, 700], [1000, 800, 1500, 600]]
        counted_direction = np.array([0, 1, 2])
    elif "cam_12/" in camera_name:
        line_group = [[130, 490, 30, 330], [380, 500, 750, 500], [850, 400, 1050, 300]]
        counted_direction = np.array([0, 1, 2])
    elif "cam_13/" in camera_name:
        line_group = [[620, 380, 385, 330], [500, 750, 1000, 700], [1550, 700, 1919, 560]]
        counted_direction = np.array([0, 1, 2])
    elif "cam_8/" in camera_name:
        line_group = [[750, 500, 600, 100], [950, 550, 1400, 400], [1400, 300, 1750, 249], [500, 800, 750, 600]]
        counted_direction = np.array([0, 1, 2, 3])
    elif "cam_9/" in camera_name:
        line_group = [[500, 600, 100, 400], [825, 725, 625, 600], [1500, 900, 1050, 650], [1125, 500, 1300, 350],
                      [1750, 400, 1450, 310],
                      [1000, 400, 875, 320], [800, 400, 750, 200]]
        counted_direction = np.array([0, 1, 2, 3, 4, 5, 6])
    else:
        line_group = [[10, 10, 20, 20]]
        counted_direction = np.array([0])
        print("=============================================")
        print("The boundary for this camera is not ready yet")
        print("=============================================")
    return line_group, counted_direction


def give_line_group_antwerp(im_filename, only_person):
    if "12_4" in im_filename:
        if only_person is "person":
            line_group = [[0, 600, 959, 600]]
        elif only_person is "car":
            line_group = [[0, 500, 959, 500]]
        counted_direction = np.array([0])
    elif "11_1" in im_filename:
        line_group = [[0, 240, 740, 466]]
        counted_direction = np.array([0])
    elif "11_2" in im_filename:
        line_group = [[180, 560, 800, 210]]
        counted_direction = np.array([0])
    elif "14_4" in im_filename:        
        line0 = [380, 320, 550, 180]
        line1 = [700, 150, 900, 200]
        line2 = [0, 400, 180, 300]
        line_group = [line0, line1, line2]
        counted_direction = np.array([0, 1, 2])
    return line_group, counted_direction


def filter_out_cyclist_from_person(direction, speed, current_count_temp, current_bike_id, current_pedestrian_id,
                                   bikespeed):
    current_count = current_count_temp.copy()
    movement_string = np.array(["static", "up", "down", "left", "right"])
    id_with_direc = direction.keys()
    current_bike_count = np.zeros(len(current_count))
    for single_id in id_with_direc:
        if single_id not in current_bike_id.keys() and single_id not in current_pedestrian_id.keys():
            if np.max(abs(speed[single_id])) > bikespeed:
                _direc = direction[single_id]
                current_bike_count[np.where(movement_string == _direc)[0]] += 1
                current_bike_id[single_id] = _direc
            else:
                current_pedestrian_id[single_id] = direction[single_id]
    current_count -= current_bike_count
    return current_count, current_bike_count, current_bike_id, current_pedestrian_id


def compute_intersection(box, boxes, for_counting=False, bike=True):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    if type(boxes) is list:
        boxes = np.array(boxes)
    if not for_counting:
        boxes = boxes[:6, :]
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)
    union = np.array([(v[2] - v[0]) * (v[3] - v[1]) for v in boxes])
    iou = intersection / union
    if bike:
        if np.mean(iou) >= 0.8:
            return "bike"
        else:
            return "ped"
    else:
        return iou
    
    
def compute_intersection_2(benchmark_area, _rois, bike=True):
    _rois = _rois.astype('int32')
    _rois_size = (_rois[:, 3] - _rois[:, 1]) * (_rois[:, 2] - _rois[:, 0])
    _intersection = [np.sum(benchmark_area[v[1]:v[3], v[0]:v[2]]) for v in _rois]
    diff = abs(_intersection - _rois_size)
    if bike:
        if abs(np.mean(diff)) < 10:
            return "bike"
        else:
            return "ped"
    else:
        return diff


def filter_out_bike_by_iou(direction, roi_group, current_count_temp, current_bike_id, current_pedestrian_id,
                           benchmark_area):
    current_count = current_count_temp.copy()
    movement_string = np.array(["static", "up", "down", "left", "right"])
    id_with_direc = direction.keys()
    current_bike_count = np.zeros(len(current_count))
    for single_id in id_with_direc:
        if single_id not in current_bike_id.keys() and single_id not in current_pedestrian_id.keys():
            _rois = roi_group[single_id]
            if len(np.shape(benchmark_area)) == 1:
                _bike_or_ped = compute_intersection(benchmark_area, _rois, bike=True)
            elif len(np.shape(benchmark_area)) == 3:
                _bike_or_ped = compute_intersection_2(benchmark_area, _rois)
            if _bike_or_ped == "bike":
                _direc = direction[single_id]
                current_bike_count[np.where(movement_string == _direc)[0]] += 1
                current_bike_id[single_id] = _direc
            else:
                current_pedestrian_id[single_id] = direction[single_id]
    current_count -= current_bike_count
    return current_count, current_bike_count, current_bike_id, current_pedestrian_id


def assign_direction_num_to_string(direct_numeric, direct_string_group, counted_direction):
    _movement_string = [["static"]]
    [_movement_string.append(["move-%d-up" % (i + 1), "move-%d-down" % (i + 1)]) for i in counted_direction]
    _movement_string = np.array([v for j in _movement_string for v in j])
    for single_key in direct_numeric.keys():
        if single_key not in direct_string_group.keys():
            _value = np.array(direct_numeric[single_key])
            _value = _value[_value != 0].astype('int32')
            if len(np.unique(_value)) > 0:
                direct_string_group[single_key] = [_movement_string[np.unique(_value)][0]]
    return direct_string_group


def give_percent_single_boundary(line_coord, bboxes, box_to_border, percentage_to_border, direction_index,
                                 iou_threshold=0.6):
    a1 = (line_coord[3] - line_coord[1]) / (line_coord[2] - line_coord[0])
    b1 = line_coord[3] - a1 * line_coord[2]
    border_1_tl = bboxes[:, 0] * a1 + b1 - bboxes[:, 1]
    border_1_tr = bboxes[:, 2] * a1 + b1 - bboxes[:, 1]
    border_1_bl = bboxes[:, 0] * a1 + b1 - bboxes[:, -1]
    border_1_br = bboxes[:, 2] * a1 + b1 - bboxes[:, -1]
    border_1_crit1 = border_1_tl * border_1_bl
    border_1_crit2 = border_1_tr * border_1_br
    border_1_crit3 = border_1_tl * border_1_tr
    border_1_crit4 = border_1_bl * border_1_br
    border_group = np.concatenate([[border_1_crit1, border_1_crit2,
                                    border_1_crit3, border_1_crit4]], axis=0)
    border_group = np.transpose(border_group, (1, 0))

    x0 = sorted([line_coord[0], line_coord[2]])
    x1 = sorted([line_coord[1], line_coord[3]])

    temp_box = [x0[0], x1[0], x0[1], x1[1]]
    if abs(line_coord[0] - line_coord[2]) < 10:
        temp_box = [line_coord[0] - 100, line_coord[1], line_coord[2] + 100, line_coord[2]]
    elif abs(line_coord[1] - line_coord[3]) < 10:
        temp_box[1] = temp_box[1] - 100
        temp_box[-1] = temp_box[-1] + 100
    overlaps = compute_intersection(temp_box, bboxes, True, False)
    for i in range(len(bboxes)):
        _b = border_group[i]
        if np.sum(_b[:2] < 0) > 0:
            if overlaps[i] >= iou_threshold:
                percent = calculate_percentage(bboxes[i], [a1, b1])
                if percent > 0.05:
                    box_to_border[i, direction_index - 1] = direction_index
                    percentage_to_border[i, direction_index - 1] = percent
        if np.min([border_1_tl[i], border_1_tr[i], border_1_bl[i], border_1_br[i]]) > 0:
            percentage_to_border[i, direction_index - 1] = 1.0
        if np.sum(_b[2:] < 0) > 0:
            if overlaps[i] >= iou_threshold:
                box_to_border[i, direction_index - 1] = direction_index
                percentage_to_border[i, direction_index - 1] = calculate_percentage(bboxes[i], [a1, b1])
        if np.min([border_1_bl[i], border_1_br[i]]) > 0:
            if bboxes[i, -2] >= line_coord[-2] and bboxes[i, 0] >= line_coord[0]:
                percentage_to_border[i, direction_index - 1] = 1.0
    return box_to_border, percentage_to_border


def assign_bbox_to_border_single_line(bboxes, line_coord, iou_threshold=0.6):
    """This function assigns the detected to each border
    line1_coord: the line decides whether the person is moving downwards or upwards
    line2_coord: the line decides whether the person is moving leftwards or rightwards
    Args:
        bboxes: [num_box, [top_left_x, top_left_y, bottom_right_x, bottom_right_y]]
        line_coord: [num_boundaries], each boundary contain [left x, left y, right x, right y]
        iou_threshold: overlapping ratio with per region
    """
    num_bboxes = len(bboxes)
    if num_bboxes == 0:
        return np.zeros([0, 2]), np.zeros([0, 2])
    if isinstance(bboxes, list):
        bboxes = np.reshape(bboxes, [num_bboxes, 4])
    bboxes = bboxes.astype(np.int32)
    box_to_border = np.zeros([num_bboxes, len(line_coord)])
    percent_to_border = np.zeros([num_bboxes, len(line_coord)])
    for iterr, single_line in enumerate(line_coord):
        box_to_border, percent_to_border = give_percent_single_boundary(single_line, bboxes,
                                                                        box_to_border, percent_to_border,
                                                                        iterr + 1, iou_threshold)
    return box_to_border, percent_to_border


def direction_count_with_more_lines(old_percentage, old_roi_index, new_roi_index, percent_difference_group,
                                    direction_arrow, counted_direction):
    """count the vehicle movements
    """
    count_movement = np.zeros([len(counted_direction) * 2 + 1])  # static, ....
    feature_percentage_need_change = np.zeros([len(new_roi_index), len(counted_direction)])
    for iterr, _direction_iter in enumerate(counted_direction):
        use_border = _direction_iter
        percent_difference = percent_difference_group[:, _direction_iter]
        movement_index = [iterr * 2, iterr * 2 + 1]
        for _p_iter in range(len(percent_difference)):
            _map_to_current_roi_index = new_roi_index[_p_iter]
            _map_to_old_roi_index = old_roi_index[_p_iter]
            if old_percentage[_map_to_old_roi_index, use_border] != 100:
                if percent_difference[_p_iter] > 0:
                    #                     print("---object is moving %s %s" % (direction, text_use[0]))
                    count_movement[movement_index[0] + 1] += 1
                    direction_arrow[_map_to_current_roi_index] = movement_index[0] + 1
                    feature_percentage_need_change[_p_iter, use_border] = _map_to_old_roi_index + 1
                elif percent_difference[_p_iter] < 0:
                    #                     print("---object is moving %s %s" % (direction, text_use[1]))
                    count_movement[movement_index[1] + 1] += 1
                    direction_arrow[_map_to_current_roi_index] = movement_index[1] + 1
                    feature_percentage_need_change[_p_iter, use_border] = _map_to_old_roi_index + 1
            elif old_percentage[_map_to_old_roi_index, use_border] == 100:
                #                 print("---object has already been counted")
                feature_percentage_need_change[_p_iter, use_border] = _map_to_old_roi_index + 1
    return direction_arrow, count_movement, feature_percentage_need_change


def count_given_boundary(current_person_id, new_object_index, new_rois,
                         person_id_old, old_percentage, old_rois, percentage_difference_group,
                         counted_direction, line_group, iou_threshold=0.1):
    direction_arrow = np.zeros([len(new_rois)])
    if len(current_person_id) > 0:
        _new_roi_with_overlapping = np.delete(np.arange(len(current_person_id)), new_object_index)
        _corresponding_old_roi_index = np.array([np.where(person_id_old == v)[0][0] for v in
                                                 current_person_id[_new_roi_with_overlapping]]).astype(np.int)
        if len(_new_roi_with_overlapping) == 0:
            #             print("the current predictions have no overlapping with previous rois")
            _, percentage_new = assign_bbox_to_border_single_line(old_rois.copy(), line_group, iou_threshold)
            q1 = (old_percentage == 100)
            _index = np.where(np.sum(q1, axis=1) > 0)[0]
            percentage_new[_index] = old_percentage[_index]
            count_movement = np.zeros([len(counted_direction) * 2 + 1])
            return direction_arrow, count_movement, percentage_new

        roi_to_border, percentage_to_border = assign_bbox_to_border_single_line(new_rois[_new_roi_with_overlapping],
                                                                                line_group, iou_threshold)
        #         l0 = np.where(current_person_id[_new_roi_with_overlapping] == 8)[0]
        #         print("id8 percentage", percentage_to_border[l0])

        #         print("old percentage", old_percentage)
        #         print("the corresponding old index", _corresponding_old_roi_index)
        #         print("the new roi to each boundary", roi_to_border)
        #         print("the percentage to each boundary", percentage_to_border)
        #         print("the new roi that has already appeared in previous time steps", _new_roi_with_overlapping)

        percent_difference = percentage_to_border - old_percentage[_corresponding_old_roi_index, :]
        delete_index = []
        _num_use = 2
        for _ii_iter, _ii in enumerate(_new_roi_with_overlapping):
            _p_id = int(current_person_id[_ii])
            if len(percentage_difference_group["id%d" % _p_id]) >= _num_use + 1:
                _sub_percent = np.diff(percentage_difference_group["id%d" % _p_id], axis=0)[-_num_use:]
                _a = np.sum(np.logical_and(_sub_percent < 1.0, _sub_percent > 0.0), axis=0)
                _b = np.sum(np.logical_and(_sub_percent < 0.0, _sub_percent > -1.0), axis=0)
                _avg_value = np.mean(_sub_percent, axis=0)
                _with_moving_direction = 0
                for _ss in range(len(_a)):
                    if _a[_ss] >= _num_use - 1 or _b[_ss] >= _num_use - 1:
                        percent_difference[_ii_iter, _ss] = _avg_value[_ss]
                        _with_moving_direction += 1
                    elif percentage_difference_group["id%d" % _p_id][-1, _ss] == 100:
                        _with_moving_direction += 1
                if _with_moving_direction == 0:
                    delete_index.append(_ii_iter)
            else:
                delete_index.append(_ii_iter)
        if len(delete_index) > 0:
            _left_index = np.delete(np.arange(len(_new_roi_with_overlapping)), np.array(delete_index))
            _corresponding_old_roi_index = _corresponding_old_roi_index[_left_index]
            _new_roi_with_overlapping = _new_roi_with_overlapping[_left_index]
            percent_difference = percent_difference[_left_index]
        #         print("old percentage", old_percentage)
        #         print("percentage difference", percent_difference)
        if len(percent_difference) > 0:
            direction_arrow, count_movement, \
            feature_percentage_need_change = direction_count_with_more_lines(old_percentage,
                                                                             _corresponding_old_roi_index,
                                                                             _new_roi_with_overlapping,
                                                                             percent_difference, direction_arrow,
                                                                             counted_direction)
            _, percentage_new = assign_bbox_to_border_single_line(old_rois.copy(), line_group, iou_threshold)
            #             q0 = np.logical_and(old_percentage > 0, old_percentage < 1)
            q1 = (old_percentage == 100)
            _index = np.where(np.sum(q1, axis=1) > 0)[0]

            #             _index = np.where(np.mean(old_percentage, axis=1) > 40)[0]
            percentage_new[_index] = old_percentage[_index]
            for _single_count_direction in counted_direction:
                _f_c = np.array([v - 1 for v in feature_percentage_need_change[:, _single_count_direction] if v != 0])
                if len(_f_c) > 0:
                    percentage_new[_f_c.astype(np.int), _single_count_direction] = 100
        else:
            _, percentage_new = assign_bbox_to_border_single_line(old_rois.copy(), line_group, iou_threshold)
            # q0 = np.logical_and(old_percentage > 0, old_percentage < 1)
            q1 = (old_percentage == 100)
            _index = np.where(np.sum(q1, axis=1) > 0)[0]
            #             _index = np.where(np.mean(old_percentage, axis=1) > 20)[0]
            percentage_new[_index] = old_percentage[_index]
            count_movement = np.zeros([len(counted_direction) * 2 + 1])
    else:
        percentage_new = old_percentage
        count_movement = np.zeros([len(counted_direction) * 2 + 1])

    return direction_arrow, count_movement, percentage_new


def filter_wrong_prediction(stat_arrange, box_standard, specified_direction, only_person="car", return_stat=False):
    _p_id = [v['current_person_id_%s' % only_person] for v in stat_arrange[:-1]]
    _c_id = [v['count_id_%s' % only_person] for v in stat_arrange[:-1]]
    _roi = [v['rois_%s' % only_person] for v in stat_arrange[:-1]]
    _direc_arrow = [v['direction_arrow_%s' % only_person] for v in stat_arrange[:-1]]
    _direc_arrow = np.array([v for j in _direc_arrow for v in j])
    _roi = np.array([v for j in _roi for v in j])
    _p_id = np.array([v for j in _p_id for v in j])
    _c_id = np.array([v for j in _c_id for v in j])

    if return_stat:
        return _p_id, _c_id, _roi, _direc_arrow

    _p_id_wt_movement = _p_id[np.where(_c_id != 0)[0]]

    subset = []
    for single_id in np.unique(_p_id_wt_movement):
        _index = np.where(_p_id == single_id)[0]
        _roi_subset = _roi[_index]
        _direc_subset = _direc_arrow[_index]
        _roi_overlapping0 = np.mean(compute_intersection(box_standard, _roi_subset, for_counting=True, bike=False)[:3])
        _direc_sub = [v for v in np.unique(_direc_subset) if v != 0]
        if len(_direc_sub) != 0:
            _value = [single_id, _direc_sub[0], _roi_overlapping0]
            subset.append(_value)
    id_left = []
    for single_direc in specified_direction:
        _index = np.where(np.array([v[1] for v in subset]) == single_direc)[0]
        _roi_overlapping = np.array([v[2] for v in subset])[_index]
        _id_left = np.array([v[0] for v in subset])[_index][_roi_overlapping != 0]
        id_left.append(_id_left)
    id_left = [v for j in id_left for v in j]

    return id_left
