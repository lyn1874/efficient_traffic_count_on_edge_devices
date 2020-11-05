import numpy as np
import os
import efficientdet.utils as eff_utils
import efficientdet.kalmanfilter as kf
import efficientdet.count_utils as count_utils
import efficientdet.tracking_utils as tracking_utils


def update_count_iou_featuresimilarity(im, old_rois, old_trajectory_stat, old_feature_embedding, old_anchor_index, 
                                       old_std_value, old_percentage, percentage_difference_group,
                                       new_rois, new_features, new_anchor_index, new_std_value, std_group, 
                                       _aggregate_feature_maps_current_frame, framed_metas,
                                       count, iou_threshold, sim_threshold, 
                                       person_id_old, kalman_filter,
                                       input_size, line1_coord, line2_coord, dist_option="euc", counting=True,
                                       counted_direction=[0,1], x_y_threshold=[960, 0], use_feature_maps=True,
                                       activate_kalman_filter=True):
    """Update the count number using both roi iou and feature similarity
    Args:
        old_rois: a list of historical detected rois, [N, 4] where x1, y1, x2, y2 are coordinates
        old_trajectory_stat: a list of [mean, std] for the past trajectories 
        old_feature_embedding: a list of extracted features for the historical rois, 
            shape [N, 5] because there are 5 levels of featuers
        old_anchor_index: an array with value between 0 to 4, 
            it defines which feature maps are most representative for the corresponding objects
        new_rois: a list of new detected rois, [N, 4]
        new_features: a list of pyramid feature maps, each with shape [1, f_ch, fh, fw], there are 5 in total
        count: int
        iou_threshold: the intersection over union threshold
        sim_threshold: feature similarity 
        person_id_old: an array, defines the unique person ID
        kalman_filter: function
    """    
    imh, imw = framed_metas[0][3], framed_metas[0][2]
    if isinstance(old_rois, list):
        old_rois = np.concatenate([old_rois], axis=0)
    if isinstance(new_rois, list):
        new_rois = np.concatenate([new_rois], axis=0) 
    if len(new_rois) == 0:
        new_rois = np.array([[0, 0, 0, 0]])
    # -- check iou -----#
    overlaps = tracking_utils.compute_overlaps(old_rois, new_rois)
    old_box_overlap = np.max(overlaps, axis=1)
    no_overlap = np.where(old_box_overlap <= 0.2)[0] #0.7 for MOT-02  # 0.1 for antwerpen dataset  
    
    # -- check the feature similarity------#
    if use_feature_maps:
        similarity_new_old = tracking_utils.calc_feature_similarity(old_feature_embedding, old_anchor_index, 
                                                                    _aggregate_feature_maps_current_frame, new_anchor_index, 
                                                                    dist_option)
    else:
        similarity_new_old = []
    
    # -- check objects who lost their track --#
    if len(no_overlap) > 0 and activate_kalman_filter:
        _miss_detect_roi, _person_id_for_miss_detection, \
        _removed_index, _jumping_object, \
        old_rois, old_trajectory_stat = tracking_utils.recover_missing_route(im, no_overlap, old_rois, old_trajectory_stat, 
                                                                             old_feature_embedding, old_anchor_index, 
                                                                             std_group, person_id_old, new_features, 
                                                                             framed_metas, kalman_filter, dist_option,
                                                                             x_y_threshold, use_feature_maps=use_feature_maps)
    else:
        _removed_index = []
        
    
    #-------------------------------------------------------------------------------------------------------------#
    #------------------------Tracking the object based on roi (+ feature maps similarity)-------------------------#
    #-------------------------------------------------------------------------------------------------------------#   
    count_hist = np.max(person_id_old) #count
    if np.sum(new_rois) != 0:
        max_iou, argmax_iou = np.max(overlaps, axis=0), np.argmax(overlaps, axis=0)
        current_person_id = np.zeros([len(new_rois)])
        for i in range(len(new_rois)):
            _iou = max_iou[i]
            if _iou >= iou_threshold:
                old_trajectory_stat, old_anchor_index, \
                    old_feature_embedding, old_rois, old_std_value, \
                    current_person_id = tracking_utils.restore_state(old_trajectory_stat, old_anchor_index, 
                                                                     old_feature_embedding, 
                                                                     person_id_old, old_rois, old_std_value, 
                                                                     argmax_iou[i], i, new_rois[i], new_anchor_index[i], 
                                                                     _aggregate_feature_maps_current_frame[i], 
                                                                     new_std_value[i], current_person_id, kalman_filter,
                                                                     use_feature_maps=use_feature_maps,
                                                                     activate_kalman_filter=activate_kalman_filter)

            else:
                old_trajectory_stat, old_anchor_index, \
                    old_feature_embedding, person_id_old, old_rois, old_std_value, \
                    current_person_id, count = get_track_for_iou_lower(i, overlaps, similarity_new_old, sim_threshold, 
                                                                       old_trajectory_stat, old_anchor_index, 
                                                                       old_feature_embedding, person_id_old, old_rois, 
                                                                       old_std_value, new_rois, 
                                                                       new_anchor_index,
                                                                       _aggregate_feature_maps_current_frame, new_std_value,
                                                                       current_person_id, kalman_filter, 
                                                                       count, use_feature_maps=use_feature_maps,
                                                                       activate_kalman_filter=activate_kalman_filter)
    #-------------------------------------------------------------------------------------------------------------#
    #----------------------Put the miss detect rois back into my rois pool----------------------------------------#
    #-------------------------------------------------------------------------------------------------------------#

    if len(no_overlap) > 0 and activate_kalman_filter:
        if np.sum(new_rois) == 0:
            new_rois = np.concatenate([_miss_detect_roi], axis=0)
            current_person_id = np.concatenate([_person_id_for_miss_detection], axis=0)
        else:
            if len(_miss_detect_roi) > 0:
                non_overlap_id = [iterr for iterr, v in enumerate(_person_id_for_miss_detection) if v not in current_person_id]
#                 print("kalman recover boxes", non_overlap_id)
                _person_id_for_miss_detection = [_person_id_for_miss_detection[_qq] for _qq in non_overlap_id]
                _miss_detect_roi = [_miss_detect_roi[_qq] for _qq in non_overlap_id]
                if len(_miss_detect_roi) > 0:
                    current_person_id = np.concatenate([current_person_id, _person_id_for_miss_detection], axis=0)
                    new_rois = np.concatenate([new_rois, _miss_detect_roi], axis=0)
                
    if np.sum(new_rois) == 0:
        current_person_id = []
    direction_arrow = np.zeros([len(new_rois)])
    #-------------------------------------------------------------------------------------------------------------#
    #----------------------------------------Count w.r.t each object class----------------------------------------#
    #-------------------------------------------------------------------------------------------------------------#
    if counting is True and np.sum(new_rois) != 0: 
        if len(current_person_id) > 0:
            _new_roi_with_overlapping = np.where(current_person_id <= count_hist)[0]
            _corresponding_old_roi_index = np.array([np.where(person_id_old == v)[0][0] for v in 
                                                    current_person_id[_new_roi_with_overlapping]]).astype(np.int)
            roi_to_border, percentage_to_border = count_utils.assign_bbox_to_border(new_rois[_new_roi_with_overlapping], 
                                                                                    line1_coord, line2_coord)
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
            if len(percent_difference) > 0:
                direction_arrow, count_movement, \
                    feature_percentage_need_change = count_utils.direction_count(old_percentage, _corresponding_old_roi_index, 
                                                                                 _new_roi_with_overlapping, 
                                                                                 percent_difference, direction_arrow,
                                                                                 counted_direction)
                _, percentage_new = count_utils.assign_bbox_to_border(old_rois.copy(), line1_coord, line2_coord)
                _index = np.where(np.mean(old_percentage, axis=1) > 40)[0]
                percentage_new[_index] = old_percentage[_index]
                
                for _single_count_direction in counted_direction:
                    _f_c = np.array([v - 1 for v in feature_percentage_need_change[:, _single_count_direction] if v!=0])
                    if len(_f_c) > 0:
                        percentage_new[_f_c.astype(np.int), _single_count_direction] = 100
            else:
                _, percentage_new = count_utils.assign_bbox_to_border(old_rois.copy(), line1_coord, line2_coord)
                _index = np.where(np.mean(old_percentage, axis=1) > 40)[0]                
                percentage_new[_index] = old_percentage[_index]
                count_movement = np.zeros([5])
        else:
            percentage_new = old_percentage
            count_movement = np.zeros([5])
    else:
        count_movement = np.zeros([5])
        if counting is True:
            percentage_new = old_percentage
        else:
            percentage_new = []
    ep = np.zeros([len(new_rois)])
    if len(no_overlap) > 0 and activate_kalman_filter:
        if len(_miss_detect_roi) > 0:
            ep[-len(_miss_detect_roi):] = 1
    
    if len(_removed_index) > 0:
        left_index = np.array([_iterr for _iterr, v in enumerate(person_id_old) if v not in _removed_index])      
        current_left_id = np.array([_iterr for _iterr, v in enumerate(current_person_id) if v not in _removed_index])
        
        left_index = [[] if len(left_index) == 0 else left_index][0]
        current_left_id = [[] if len(current_left_id) == 0 else current_left_id][0]
        if len(current_person_id) > 0 and len(current_person_id) != len(current_left_id):
            print("removin person id from current person group", _removed_index)
            current_person_id = [np.array(current_person_id) if isinstance(current_person_id, list) else current_person_id][0]
            current_person_id = current_person_id[current_left_id]
            new_rois = new_rois[current_left_id]
            ep = ep[current_left_id]
            direction_arrow = direction_arrow[current_left_id]

    new_object_index = [np.where(current_person_id > count_hist)[0] if not isinstance(current_person_id, list) else []][0]

    stat = [old_rois, old_trajectory_stat, old_feature_embedding, old_anchor_index, old_std_value, 
            percentage_new, new_rois, current_person_id, new_object_index, 
            person_id_old, count, ep, count_movement, direction_arrow, _removed_index]
    return stat


def get_track_for_iou_lower(i, overlaps, similarity_new_old, sim_threshold, 
                            old_trajectory_stat, old_anchor_index, 
                            old_feature_embedding, person_id_old, old_rois, old_std_value, new_rois, 
                            new_anchor_index, _aggregate_feature_maps_current_frame, new_std_value,
                            current_person_id, kalman_filter, count, use_feature_maps=True, 
                            activate_kalman_filter=True):
    max_iou, argmax_iou = np.max(overlaps, axis=0), np.argmax(overlaps, axis=0)
    _iou = max_iou[i]
    if not use_feature_maps:
        old_trajectory_stat, old_anchor_index, \
            old_feature_embedding, person_id_old, \
            old_rois, old_std_value, \
            current_person_id, count = tracking_utils.update_state(old_trajectory_stat, old_anchor_index, 
                                                                   old_feature_embedding, person_id_old, old_rois, 
                                                                   old_std_value, i, new_rois[i], 
                                                                   new_anchor_index[i], old_feature_embedding, 
                                                                   new_std_value[i], current_person_id, count, 
                                                                   kalman_filter, use_feature_maps=False,
                                                                   activate_kalman_filter=activate_kalman_filter)
    else:
        if _iou != 0:
            _iou_control = np.argsort(overlaps[:, i])[-3:]
            _sim_subset = similarity_new_old[i, _iou_control]
            _sim_filter = np.where(_sim_subset >= sim_threshold)[0]
            _iou_select = overlaps[_iou_control[_sim_filter], i]
            if len(_iou_select) > 0 and np.sum(_iou_select > 0) > 1:
                if isinstance(_iou_select, list):
                    _ind = [1 if np.sorted(_iou_select)[-1] > 0 else 2][0]
                else:
                    _ind = 1
                _argmax_feat_sim = _iou_control[_sim_filter][np.argsort(_iou_select)[-_ind]]
            else:
                _argmax_feat_sim = _iou_control[np.argmax(_sim_subset)]
                _argmax_feat_sim = [_argmax_feat_sim if overlaps[_argmax_feat_sim, i] != 0 else
                                            np.argsort(overlaps[:, i])[-1]][0]
            if similarity_new_old[i, _argmax_feat_sim] >= sim_threshold:
                old_trajectory_stat, old_anchor_index, \
                    old_feature_embedding, old_rois, old_std_value, \
                    current_person_id = tracking_utils.restore_state(old_trajectory_stat, old_anchor_index, 
                                                                     old_feature_embedding, person_id_old, 
                                                                     old_rois, old_std_value, _argmax_feat_sim,
                                                                     i, new_rois[i], new_anchor_index[i], 
                                                                     _aggregate_feature_maps_current_frame[i], 
                                                                     new_std_value[i], current_person_id, 
                                                                     kalman_filter, use_feature_maps=True,
                                                                     activate_kalman_filter=activate_kalman_filter)
            else:
                old_trajectory_stat, old_anchor_index, \
                    old_feature_embedding, person_id_old, \
                    old_rois, old_std_value, \
                    current_person_id, count = tracking_utils.update_state(old_trajectory_stat, 
                                                                           old_anchor_index, old_feature_embedding, 
                                                                           person_id_old, old_rois, 
                                                                            old_std_value, i, new_rois[i], 
                                                                            new_anchor_index[i], 
                                                                            _aggregate_feature_maps_current_frame[i], 
                                                                            new_std_value[i], current_person_id, 
                                                                            count, kalman_filter, use_feature_maps=True,
                                                                            activate_kalman_filter=activate_kalman_filter)
        else:
            old_trajectory_stat, old_anchor_index, \
                old_feature_embedding, person_id_old, \
                old_rois, old_std_value, \
                current_person_id, count = tracking_utils.update_state(old_trajectory_stat, 
                                                                       old_anchor_index, old_feature_embedding, 
                                                                       person_id_old, old_rois, old_std_value, i, 
                                                                       new_rois[i], new_anchor_index[i], 
                                                                       _aggregate_feature_maps_current_frame[i], 
                                                                       new_std_value[i], current_person_id, 
                                                                       count, kalman_filter, use_feature_maps=True,
                                                                       activate_kalman_filter=activate_kalman_filter)
    updated_stat = [old_trajectory_stat, old_anchor_index, old_feature_embedding, person_id_old, 
                    old_rois, old_std_value, current_person_id, count]
    return updated_stat
        
            
