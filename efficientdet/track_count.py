import numpy as np
import efficientdet.tracking_utils as tracking_utils


def update_count_iou(im, old_rois, old_trajectory_stat,
                     old_std_value, new_rois, new_std_value, std_group,
                     framed_metas, count, iou_threshold,
                     person_id_old, kalman_filter,
                     x_y_threshold=[960, 0], activate_kalman_filter=True):
    """Update the count number using both roi iou and feature similarity
    Args:
        im: input image, it's used for calculating standard deviation in the detected bounding boxes
        old_rois: a list of historical detected rois, [N, 4] where x1, y1, x2, y2 are coordinates
        old_trajectory_stat: a list of [mean, std] for the past trajectories
        old_std_value: the standard deviation for each detected bounding box
        new_rois: a list of new detected rois, [N, 4]
        new_std_value: the standard deviation of the new detected bounding boxes
        std_group: object, the keys are IDs
        framed_metas: used in recover missing route
        count: int, the total count
        iou_threshold: the intersection over union threshold
        sim_threshold: feature similarity 
        person_id_old: an array, defines the unique person ID
        kalman_filter: function
        x_y_threshold: used for recovering missing route
        activate_kalman_filter: bool variable, True means activating kalman filter
    """
    if isinstance(old_rois, list):
        old_rois = np.concatenate([old_rois], axis=0)
    if isinstance(new_rois, list):
        new_rois = np.concatenate([new_rois], axis=0)
    if len(new_rois) == 0:
        new_rois = np.array([[0, 0, 0, 0]])
    # -- check iou -----#
    overlaps = tracking_utils.compute_overlaps(old_rois, new_rois)
    old_box_overlap = np.max(overlaps, axis=1)
    no_overlap = np.where(old_box_overlap <= 0.2)[0]  # 0.7 for MOT-02  # 0.1 for antwerpen dataset

    # -- check objects who lost their track --#
    if len(no_overlap) > 0 and activate_kalman_filter:
        _miss_detect_roi, _person_id_for_miss_detection, \
            _removed_index, old_rois, \
            old_trajectory_stat = tracking_utils.recover_missing_route(im, no_overlap, old_rois,
                                                                       old_trajectory_stat,
                                                                       std_group, person_id_old,
                                                                       framed_metas, kalman_filter, x_y_threshold)
    else:
        _removed_index = []

    # -------------------------------------------------------------------------------------------------------------#
    # ------------------------Tracking the object based on roi (+ feature maps similarity)-------------------------#
    # -------------------------------------------------------------------------------------------------------------#
    count_hist = np.max(person_id_old)  # count
    if np.sum(new_rois) != 0:
        max_iou, argmax_iou = np.max(overlaps, axis=0), np.argmax(overlaps, axis=0)
        current_person_id = np.zeros([len(new_rois)])
        for i in range(len(new_rois)):
            _iou = max_iou[i]
            if _iou >= iou_threshold:
                old_trajectory_stat, old_rois, old_std_value, \
                    current_person_id = tracking_utils.restore_state(old_trajectory_stat,
                                                                     person_id_old, old_rois, old_std_value,
                                                                     argmax_iou[i], i, new_rois[i],
                                                                     new_std_value[i], current_person_id, kalman_filter,
                                                                     activate_kalman_filter=activate_kalman_filter)

            else:
                old_trajectory_stat, person_id_old, old_rois, old_std_value, \
                    current_person_id, count = get_track_for_iou_lower(i, overlaps,
                                                                       old_trajectory_stat, person_id_old, old_rois,
                                                                       old_std_value, new_rois, new_std_value,
                                                                       current_person_id, kalman_filter,
                                                                       count,
                                                                       activate_kalman_filter=activate_kalman_filter)
    # -------------------------------------------------------------------------------------------------------------#
    # ----------------------Put the miss detect rois back into my rois pool----------------------------------------#
    # -------------------------------------------------------------------------------------------------------------#

    if len(no_overlap) > 0 and activate_kalman_filter:
        if np.sum(new_rois) == 0:
            new_rois = np.concatenate([_miss_detect_roi], axis=0)
            current_person_id = np.concatenate([_person_id_for_miss_detection], axis=0)
        else:
            if len(_miss_detect_roi) > 0:
                non_overlap_id = [iterr for iterr, v in enumerate(_person_id_for_miss_detection) if
                                  v not in current_person_id]
                #                 print("kalman recover boxes", non_overlap_id)
                _person_id_for_miss_detection = [_person_id_for_miss_detection[_qq] for _qq in non_overlap_id]
                _miss_detect_roi = [_miss_detect_roi[_qq] for _qq in non_overlap_id]
                if len(_miss_detect_roi) > 0:
                    current_person_id = np.concatenate([current_person_id, _person_id_for_miss_detection], axis=0)
                    new_rois = np.concatenate([new_rois, _miss_detect_roi], axis=0)

    if np.sum(new_rois) == 0:
        current_person_id = []
    # -------------------------------------------------------------------------------------------------------------#
    # ----------------------------------------Count w.r.t each object class----------------------------------------#
    # -------------------------------------------------------------------------------------------------------------#
    ep = np.zeros([len(new_rois)])
    if len(no_overlap) > 0 and activate_kalman_filter:
        if len(_miss_detect_roi) > 0:
            ep[-len(_miss_detect_roi):] = 1

    if len(_removed_index) > 0:
        current_left_id = np.array([_iterr for _iterr, v in enumerate(current_person_id) if v not in _removed_index])
        current_left_id = [[] if len(current_left_id) == 0 else current_left_id][0]
        if len(current_person_id) > 0 and len(current_person_id) != len(current_left_id):
            print("removin person id from current person group", _removed_index)
            current_person_id = \
                [np.array(current_person_id) if isinstance(current_person_id, list) else current_person_id][0]
            current_person_id = current_person_id[current_left_id]
            new_rois = new_rois[current_left_id]
            ep = ep[current_left_id]

    new_object_index = [np.where(current_person_id > count_hist)[0] if not isinstance(current_person_id, list) else []][
        0]

    stat = [old_rois, old_trajectory_stat, old_std_value,
            new_rois, current_person_id, new_object_index,
            person_id_old, count, ep, _removed_index]
    return stat


def get_track_for_iou_lower(i, overlaps, old_trajectory_stat, person_id_old,
                            old_rois, old_std_value, new_rois, new_std_value,
                            current_person_id, kalman_filter, count,
                            activate_kalman_filter=True):
    max_iou, argmax_iou = np.max(overlaps, axis=0), np.argmax(overlaps, axis=0)
    _iou = max_iou[i]

    old_trajectory_stat, person_id_old, \
        old_rois, old_std_value, \
        current_person_id, count = tracking_utils.update_state(old_trajectory_stat, person_id_old, old_rois,
                                                               old_std_value, i, new_rois[i],
                                                               new_std_value[i], current_person_id, count,
                                                               kalman_filter,
                                                               activate_kalman_filter=activate_kalman_filter)

    updated_stat = [old_trajectory_stat, person_id_old, old_rois, old_std_value, current_person_id, count]
    return updated_stat
