#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13.08.20 at 16:22
@author: li
"""
import torch
import cv2
import numpy as np
import time
import os
from tqdm import tqdm
import pickle
import prediction as model_arch
import efficientdet.utils as eff_utils
import efficientdet.count_utils as count_utils
import efficientdet.tracking_utils as tracking_utils
import efficientdet.track_count as tc
import vis_utils as vt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]


class TrackCount(object):
    def __init__(self, model, compound_coef, threshold, nms_threshold, only_person, max_objects,
                 iou_threshold_list, class_index, class_group, subtract_bg=False, student=False,
                 params=None, resize=False, filter_small_box=True, minus_bg_norm=False,
                 x_y_threshold=[960, 0], bike_speed=6, activate_kalman_filter=True):
        """Initialize the tracking and counting parameters
        model: the efficientdet model that gives the bounding box regression and prediction
        compound_coef: int
        threshold: the classification probability threshold, float
        nms_threshold: the nms threshold that is used in the postprocessing step
        only_person: str, it can be either "car", "person", "ped_car", "cyclist"
        max_objects: int, the maximum number of objects that are saved in the memory
        iou_threshold_list: a list of iou_threshold, it includes the iou threshold for counting different type of object
        sim_threshold_list: same format as the iou_threshold_list
        counting: bool, True or False
        num_class: int, how many types of object i need to count and track
        """
        self.model = model
        self.compound_coef = compound_coef
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.only_person = only_person
        self.max_num_objects = max_objects
        self.iou_threshold_list = iou_threshold_list
        self.subtract_bg = subtract_bg
        self.filter_small_box = filter_small_box
        self.student = student
        self.params = params
        self.resize = resize
        self.bikespeed = bike_speed
        self.x_y_threshold = x_y_threshold
        self.class_index = class_index
        self.class_group = class_group
        num_class = np.arange(len(self.class_index))
        self.num_class = num_class
        self.minus_bg_norm = minus_bg_norm
        self.activate_kalman_filter = activate_kalman_filter
        self.parking_lot = []
        self.preprocess_mean = self.params['mean'][::-1]
        self.preprocess_std = self.params['std'][::-1]
        if self.subtract_bg is True and self.minus_bg_norm is False:
            self.preprocess_mean = [0.0, 0.0, 0.0]
            self.preprocess_std = [1.0, 1.0, 1.0]
        print("-----------------------------------------------------------------------------")
        print("model level", self.compound_coef)
        print("classification probability threshold", self.threshold)
        print("box prediction nms threshold", self.nms_threshold)
        print("the objects that need to be counted", self.only_person, self.class_index,
              self.class_group, self.num_class)
        print("activating kalman filter to recover miss detect object", self.activate_kalman_filter)
        print("iou threshold to identify whether two people are same person", self.iou_threshold_list)
        print("the input needs to be resized", self.resize)
        print("there should be maximum %d objects in the memory" % self.max_num_objects)
        print("subtracting the background from the input frame", self.subtract_bg)
        print("The mean and standard deviation for the input frame", self.preprocess_mean, self.preprocess_std)
        print("Normalizing the input frames after subtracting the background", self.minus_bg_norm)
        print("-----------------------------------------------------------------------------")

        self.regressBoxes = eff_utils.BBoxTransform()
        self.clipBoxes = eff_utils.ClipBoxes()

        self.roi_group = [{} for _ in num_class]
        self.person_id_group = [{} for _ in num_class]
        self.std_group = [{} for _ in num_class]
        self.direction_group = [{} for _ in num_class]
        self.direction_string_group = [{} for _ in num_class]
        self.percentage_difference_group = [{} for _ in num_class]
        self.roi_std_group = [{} for _ in num_class]
        self.speed_group = [{} for _ in num_class]
        self.bike_id_group = {}
        self.pedestrian_id_group = {}
        self.car_id_group = {}

        # -------- old statistics------------------------------#
        self.old_rois = [[] for _ in num_class]
        self.count = np.zeros([len(num_class)])

        self.old_feature_embedding = [[] for _ in num_class]
        self.old_anchor_index = [[] for _ in num_class]
        self.old_std_value = [[] for _ in num_class]
        self.old_percentage = [[] for _ in num_class]
        self.person_id_old = [[] for _ in num_class]
        self.old_trajectory_stat = [[] for _ in num_class]

        self.pred_stat = []

        # ---------current statistics---------------------------#
        self.current_rois, self.current_orig_rois, self.current_featuremaps, self.current_anchor_index, \
        self.current_std_value, self.current_class_ids, self.framed_metas, self.im, \
        self.current_orig_featuremaps = [None for _ in range(9)]
        self.current_person_id = None
        self.new_object_index = None

    def _load_precalculated_stat(self, camera):
        path = '/project/bo/AICity_Counting/HCMUS/track1-multi-intersection-counting/CenterNet/info_tracking/info_%s.mp4.npy' % camera
        _stat = np.load(path)
        self.pred_stat = _stat

    def _give_prediction_baseon_precalculation(self, im_filename, roi_interest):
        imindex = int(im_filename.split('frame_')[1].split('.jpg')[0])
        _subindex = np.where(self.pred_stat[:, 1] == imindex)[0]
        class_ids = self.pred_stat[_subindex, 0]
        rois = self.pred_stat[_subindex, -4:]
        subindex = []
        for j, single_roi in enumerate(rois):
            x1, y1, x2, y2 = single_roi.astype(np.int32)
            if np.mean(roi_interest[y1:y2, x1:x2]) > 0.90:
                subindex.append(j)
        subindex = np.array(subindex)
        if len(subindex) > 0:
            class_ids = class_ids[subindex]
            rois = rois[subindex]
            if self.only_person is "car":
                class_ids[class_ids == 2] = 1
        else:
            rois = np.array([], dtype=np.float64)
            class_ids = np.array([], dtype=np.float64)
        im = cv2.imread(im_filename)[:, :, ::-1] / 255.0
        return rois, class_ids, im

    def omit_parked_cars(self, roi_group, index):
        if len(self.parking_lot) > 0:
            # if len(roi_group) == 1:
            #     roi_group = np.expand_dims(roi_group, axis=0)
            overlaps = count_utils.compute_intersection_2(self.parking_lot, roi_group, bike=False)
            kept_index = np.where(overlaps > 100)[0]
            return index[kept_index]
        else:
            return index

    def give_prediction(self, im_filename, background, roi_interest):
        if len(self.pred_stat) > 0:
            rois, class_ids, im = self._give_prediction_baseon_precalculation(im_filename, roi_interest)
            orig_rois, index = rois, []
            framed_metas = [[512, 288, 1280, 720, 0, 224]]
            framed_metas[0][3], framed_metas[0][2] = np.shape(im)[0], np.shape(im)[1]
            framed_metas[0][1] = framed_metas[0][0] * framed_metas[0][3] // framed_metas[0][2]
            framed_metas[0][-1] = framed_metas[0][0] - framed_metas[0][1]
            _feature_use = [[] for _ in range(len(rois))]
        else:
            detector_output = model_arch.get_prediction(im_filename, input_sizes[self.compound_coef],
                                                        self.preprocess_mean, self.preprocess_std,
                                                        background, self.model, self.threshold, self.nms_threshold,
                                                        self.regressBoxes, self.clipBoxes, self.only_person,
                                                        student=self.student, filter_small_box=self.filter_small_box,
                                                        x_y_threshold=self.x_y_threshold,
                                                        roi_interest=roi_interest, minus_bg_norm=self.minus_bg_norm)
            [_, rois, orig_rois, _, _, framed_metas, class_ids], im = detector_output
        std_value = tracking_utils.get_std_for_detection(im, rois)
        current_rois, current_orig_rois, current_std_value = [], [], []
        for i in self.class_index:
            _index = np.where(class_ids == i)[0]
            if self.only_person == "ped_car":
                if self.class_group[i] == "car" and len(_index) > 0:
                    _index = self.omit_parked_cars(rois[_index], _index)
            else:
                _index = np.arange(len(rois))
            current_rois.append(rois[_index])
            current_orig_rois.append(orig_rois[_index])
            current_std_value.append(std_value[_index])

        self.current_rois = current_rois
        self.current_orig_rois = current_orig_rois  # not needed
        self.current_std_value = current_std_value  # yes
        self.current_class_ids = class_ids  # yes
        self.framed_metas = framed_metas
        self.im = im

    def give_initial_or_empty_stat(self, class_index, iterr, line_group):
        """Note this initialization is for possibly multiple types of objects"""
        ci = class_index
        num_current_detection = len(self.current_rois[ci])
        if num_current_detection > 0:
            if iterr == 0:
                current_person_id = np.arange(num_current_detection + 1)[1:]
            else:
                current_person_id = np.arange(self.count[ci] + 1 + num_current_detection)[(int(self.count[ci]) + 1):]
        else:
            current_person_id = []
        new_object_index = np.arange(len(current_person_id))
        self.count[ci] += num_current_detection
        self.person_id_old[ci] = current_person_id.copy()
        self.old_rois[ci] = self.current_rois[ci].copy()
        self.old_std_value[ci] = self.current_std_value[ci].copy()
        if self.activate_kalman_filter:
            self.old_trajectory_stat[ci] = tracking_utils.get_kalman_init(self.current_rois[ci], 0)
        else:
            self.old_trajectory_stat[ci] = []
        _, self.old_percentage[ci] = count_utils.assign_bbox_to_border_single_line(self.current_rois[ci].copy(),
                                                                                   line_group, 0.2)
        count_movement = np.zeros([len(line_group) * 2 + 1])
        if iterr == 0:
            self.count_movement_group = [np.zeros(len(count_movement)) for _ in self.class_group]
            self.count_move_pedestrain_group = np.zeros([len(count_movement)])
            self.count_move_cyclists_group = np.zeros([len(count_movement)])
        direction_arrow = np.zeros([len(self.current_rois[ci])])
        ep = np.zeros([len(self.current_rois[ci])])
        return current_person_id, new_object_index, count_movement, direction_arrow, ep, []

    def give_update_stat(self, class_index):
        ci = class_index
        update_state = tc.update_count_iou(self.im, self.old_rois[ci], self.old_trajectory_stat[ci],
                                           self.old_std_value[ci],
                                           self.current_rois[ci], self.current_std_value[ci],
                                           self.std_group[ci],
                                           self.framed_metas, self.count[ci],
                                           self.iou_threshold_list[ci],
                                           self.person_id_old[ci], [],
                                           x_y_threshold=self.x_y_threshold,
                                           activate_kalman_filter=self.activate_kalman_filter)
        self.old_rois[ci], self.old_trajectory_stat[ci], self.old_std_value[ci], \
            self.current_rois[ci], current_person_id, new_object_index, self.person_id_old[ci], \
            self.count[ci], ep, _removed_index = update_state
        return current_person_id, new_object_index, ep, _removed_index

    def count_in_separate_window(self, counted_direction, line_group):
        direction_arrow_group, current_count_group = [], []
        for ci in self.num_class:
            direction_arrow, \
                count_movement, \
                percentage_new = count_utils.count_given_boundary(self.current_person_id[ci],
                                                                  self.new_object_index[ci], self.current_rois[ci],
                                                                  self.person_id_old[ci], self.old_percentage[ci],
                                                                  self.old_rois[ci],
                                                                  self.percentage_difference_group[ci],
                                                                  counted_direction, line_group)
            self.old_percentage[ci] = percentage_new
            self.count_movement_group[ci][:(2 * len(counted_direction) + 1)] += count_movement
            direction_arrow_group.append(direction_arrow)
            current_count_group.append(count_movement)
        return direction_arrow_group, current_count_group

    def save_stat_in_object(self, current_person_id_group, direction_arrow_group, new_object_index_group,
                            kalman_prediction):
        for ci in self.num_class:
            current_person_id = current_person_id_group[ci]
            direction_arrow = direction_arrow_group[ci]
            new_object_index = new_object_index_group[ci]
            current_person_id = [current_person_id.astype(np.int) if not isinstance(current_person_id, list)
                                 else current_person_id][0]
            for _single_numeric_index, _single_person_id in enumerate(current_person_id):
                if _single_numeric_index in new_object_index:
                    self.roi_group[ci]["id%d" % _single_person_id] = [self.current_rois[ci][_single_numeric_index]]
                    _std_value = self.old_std_value[ci][np.where(self.person_id_old[ci] == _single_person_id)[0]]
                    self.std_group[ci]["id%d" % _single_person_id] = [_std_value]
                    self.direction_group[ci]["id%d" % _single_person_id] = [direction_arrow[_single_numeric_index]]
                    self.roi_std_group[ci]["id%d" % _single_person_id] = np.array([0, 0])
                else:
                    _roi_0 = self.roi_group[ci]["id%d" % _single_person_id]
                    _roi_1 = [self.current_rois[ci][_single_numeric_index]]
                    self.roi_group[ci]["id%d" % _single_person_id] = np.concatenate([_roi_0, _roi_1], axis=0)
                    _std_value = self.old_std_value[ci][np.where(self.person_id_old[ci] == _single_person_id)[0]][0]
                    self.std_group[ci]["id%d" % _single_person_id].append(_std_value)
                    self.direction_group[ci]["id%d" % _single_person_id].append(direction_arrow[_single_numeric_index])
                    self.roi_std_group[ci]["%d" % _single_person_id] = tracking_utils.calc_roi_std(
                        self.roi_group[ci]["id%d" % _single_person_id])
                if self.activate_kalman_filter:
                    self.speed_group[ci]["id%d" % _single_person_id] = self.old_trajectory_stat[ci][
                        np.where(self.person_id_old[ci] == _single_person_id)[0][0]].get_speed()
                else:
                    self.speed_group[ci]["id%d" % _single_person_id] = tracking_utils.calc_speed_based_roi_loc(
                        self.roi_group[ci]["id%d" % _single_person_id])
            #             print("already appeared person id", self.person_id_old[ci])
            #             print("current person id", current_person_id)
            for _single_numeric_index, _single_person_id in enumerate(self.person_id_old[ci]):
                _index = np.where(current_person_id == _single_person_id)[0]
                if _index in new_object_index:
                    self.person_id_group[ci]["id%d" % _single_person_id] = [1]
                    _old_percentage = np.expand_dims(self.old_percentage[ci][_single_numeric_index], axis=0)
                    self.percentage_difference_group[ci][
                        "id%d" % _single_person_id] = _old_percentage
                    # so this percentage difference group is not really percentage
                    # difference, it's actually the percentage group
                else:
                    if len(_index) > 0:
                        _old_percentage_0 = self.percentage_difference_group[ci]["id%d" % _single_person_id]
                        _old_percentage_1 = [self.old_percentage[ci][_single_numeric_index]]
                        self.percentage_difference_group[ci]["id%d" % _single_person_id] = np.concatenate([
                            _old_percentage_0, _old_percentage_1], axis=0)
                    if _single_person_id in current_person_id:
                        if kalman_prediction[ci][_index[0]] == 0:
                            self.person_id_group[ci]["id%d" % _single_person_id].append(1)
                    else:
                        self.person_id_group[ci]["id%d" % _single_person_id].append(0)

    def let_arrow_keep_going(self, current_person_id_group, direction_arrow_group):
        for ci in self.num_class:
            current_person_id = current_person_id_group[ci]
            direction_arrow = direction_arrow_group[ci]
            already_counted_boxes = [_it for _it, v in enumerate(self.old_percentage[ci]) if 100 in v]
            if len(already_counted_boxes) > 0:
                for _single_count_box in already_counted_boxes:
                    _id = self.person_id_old[ci][_single_count_box]
                    if self.direction_group[ci]["id%d" % _id][-2] != 0:
                        _num_index = np.where(current_person_id == _id)[0]
                        if len(_num_index) > 0:
                            direction_arrow[_num_index] = self.direction_group[ci]["id%d" % _id][-2]
                            self.direction_group[ci]["id%d" % _id][-1] = direction_arrow[_num_index][0]
                direction_arrow_group[ci] = direction_arrow

        return direction_arrow_group

    def _remove_object_key(self, ci, _key):
        del self.roi_group[ci][_key]
        del self.std_group[ci][_key]
        del self.direction_group[ci][_key]
        del self.percentage_difference_group[ci][_key]
        del self.person_id_group[ci][_key]

    def _cancel_object(self, ci, _kept_id):
        update_state_after_remove = tracking_utils.cancel_object(self.old_rois[ci],
                                                                 self.old_trajectory_stat[ci],
                                                                 self.old_std_value[ci], self.person_id_old[ci],
                                                                 self.old_percentage[ci], _kept_id,
                                                                 counting=True,
                                                                 activate_kalman_filter=self.activate_kalman_filter)
        self.old_rois[ci], self.old_trajectory_stat[ci], self.old_feature_embedding[ci], \
            self.old_anchor_index[ci], self.old_std_value[ci], self.person_id_old[ci], \
            self.old_percentage[ci] = update_state_after_remove

    def remove_disappeared_objects(self, kf_removed_index):
        _disappear_person_id_group = []
        for ci in self.num_class:
            _dis_single_class = []
            for _single_iterr, _single_id in enumerate(self.person_id_old[ci]):
                if np.mean(self.person_id_group[ci]["id%d" % _single_id][
                           -5:]) == 0:  # for the cityflow need to use std # for aicity no std
                    if np.max(self.roi_std_group[ci][
                                  "id%d" % _single_id]) >= 0:  # 20:  # for the cityflow 20, for ai-city 0
                        _dis_single_class.append(_single_id)
                        # print("removing %s" % self.class_group[ci], _single_id)
            if len(kf_removed_index[ci]) > 0:
                # print("removing %s because of kalman filter" % self.class_group[ci], kf_removed_index[ci])
                _dis_single_class = np.concatenate([_dis_single_class, kf_removed_index[ci]], axis=0)
                _dis_single_class = np.unique(_dis_single_class)
                if self.class_group[ci] == "car":
                    if len(self.parking_lot) > 0:
                        #  -----------Here, calculate the overlapping with the parking lot-------------#
                        _roi = self.roi_group[ci]
                        _parked_car = []
                        for _id in _roi.keys():
                            if len(_roi[_id]) <= 1:
                                continue
                            overlaps = tracking_utils.compute_overlaps(self.parking_lot, _roi[_id])
                            if np.max(overlaps) > 0.62:
                                _parked_car.append(int(_id.split("id")[1]))
                                print("car %s is parked in the defined parking lot" % _parked_car)
                            if len(_parked_car) > 0:
                                _dis_single_class = np.unique(np.concatenate([_dis_single_class, _parked_car], axis=0))
            if len(_dis_single_class) > 0:
                _removed_person_id = _dis_single_class
                _removed_person_id_index = [np.where(self.person_id_old[ci] == v)[0] for v in _removed_person_id]
                _kept_id = np.delete(np.arange(len(self.person_id_old[ci])), np.array(_removed_person_id_index))
                # print("remove object", _kept_id)
                self._cancel_object(ci, _kept_id)
                for _single_remove in _removed_person_id:
                    self._remove_object_key(ci, "id%d" % _single_remove)
            if len(self.old_rois[ci]) > self.max_num_objects:
                self._cancel_object(ci, self.max_num_objects)
            for _single_key in list(self.roi_group[ci].keys()):
                if int(_single_key.strip().split("id")[1]) not in self.person_id_old[ci]:
                    # print("this %s has been removed from the pool set" % self.class_group[ci], _single_key)
                    self._remove_object_key(ci, _single_key)
                else:
                    if len(self.roi_group[ci][_single_key]) > 100:
                        self.roi_group[ci][_single_key] = self.roi_group[ci][_single_key][-100:]
                        self.std_group[ci][_single_key] = self.std_group[ci][_single_key][-100:]
                        self.person_id_group[ci][_single_key] = self.person_id_group[ci][_single_key][-100:]
                        self.direction_group[ci][_single_key] = self.direction_group[ci][_single_key][-100:]
                        self.percentage_difference_group[ci][
                            _single_key] = self.percentage_difference_group[ci][_single_key][-100:]

    def run(self, im_filenames, save_json_filename, save_video, save_video_filename, show,
            use_precalculated_detection=False, predefine_line=[]):
        camera_name = im_filenames[0].strip().split('/')[-3]
        date = im_filenames[0].strip().split('/')[-2]
        if "aic2020" in im_filenames[0]:
            line_group, counted_direction = count_utils.give_aicity_linegroup(im_filenames[0])
            cam = im_filenames[0].strip().split('/')[-2]
            roi_interest = np.load(im_filenames[0].split("Dataset_A_Frame")[0] + "ROIs/%s.npy" % cam)
            print("The roi interest region", np.shape(roi_interest))
        elif "/jovyan/" in im_filenames[0]:
            line_group, counted_direction = count_utils.give_line_group_antwerp(im_filenames[0], self.only_person)
            roi_interest = np.load('/home/jovyan/bo/dataset/%s/annotations/rois.npy' % camera_name)
            self.parking_lot = np.load('/home/jovyan/bo/dataset/%s/annotations/parking_lot.npy' % camera_name)
            print("....I am doing experiment on camera %s and date %s" % (camera_name, date))
        else:
            line_group = predefine_line
            counted_direction = np.arange(len(line_group))
            roi_interest = []
        print("counting use multiple boundaries", line_group)
        if self.subtract_bg:
            if "aic2020" in im_filenames[0]:
                path_middle = "bo/normal_data/aic2020/AIC20_track1/Dataset_A_Frame/annotations/bg_%s.npy" % cam
                if 'tmp' in im_filenames[0]:
                    bg = np.load("/tmp/" + path_middle)
                else:
                    bg = np.load("/project_scratch/" + path_middle)
            else:
                bg = np.load('/home/jovyan/bo/dataset/%s/annotations/bg_%s.npy' % (camera_name, date))
        else:
            bg = [0.0, 0.0, 0.0]
        if save_video:
            video_writer = vt.get_video_writer(np.shape(cv2.imread(im_filenames[0])),
                                               save_video_filename, [])
            stat_folder = save_video_filename.strip().split('.avi')[0] + '/'
            if not os.path.exists(stat_folder):
                os.makedirs(stat_folder)
        else:
            if save_video_filename:
                stat_folder = save_video_filename.strip().split('.avi')[0] + '/'
                if not os.path.exists(stat_folder):
                    os.makedirs(stat_folder)
            else:
                stat_folder = None
        save_box_stat = [v for v in range(len(im_filenames) + 1)[500:3000] if len(im_filenames) % v == 0]
        if len(save_box_stat) == 0:
            save_box_stat = len(im_filenames)
        else:
            save_box_stat = int(save_box_stat[-1])
        print("saving staticis every %d" % save_box_stat)
        count_stat_for_saving = []
        if use_precalculated_detection:
            self._load_precalculated_stat(date)

        iterr = 0
        while iterr < len(im_filenames) - 1:
            for iterr, single_imname in enumerate(tqdm(im_filenames)):
                now = time.ctime(os.path.getmtime(single_imname))
                if iterr % save_box_stat == 0:
                    print("empty video statistics", iterr)
                    video_stat = []
                self.give_prediction(single_imname, bg, roi_interest)
                _current_person_id_g, _new_object_index_g, _direction_arrow_group, _ep = [], [], [], []
                _current_count_group = []
                _kf_remove_index = []
                for ci in self.num_class:
                    if iterr == 0 or len(self.old_rois[ci]) == 0:
                        _track_count_state = self.give_initial_or_empty_stat(ci, iterr, line_group)
                    else:
                        _track_count_state = self.give_update_stat(ci)
                    _current_person_id_g.append(_track_count_state[0])
                    _new_object_index_g.append(_track_count_state[1])
                    _ep.append(_track_count_state[-2])
                    _kf_remove_index.append(_track_count_state[-1])
                    if iterr == 0:
                        _current_count_group.append(_track_count_state[2])
                        self.count_movement_group[ci] += _track_count_state[2]
                        _direction_arrow_group.append(_track_count_state[3])

                # current_person_id, new_object_index, count_movement, direction_arrow, ep, remove_index
                self.current_person_id = _current_person_id_g
                self.new_object_index = _new_object_index_g
                if iterr != 0:
                    _direction_arrow_group, _current_count_group = self.count_in_separate_window(counted_direction,
                                                                                                 line_group)
                self.save_stat_in_object(self.current_person_id, _direction_arrow_group, self.new_object_index,
                                         _ep)
                _direction_arrow_group = self.let_arrow_keep_going(self.current_person_id, _direction_arrow_group)
                for ci in self.num_class:
                    self.direction_string_group[ci] = count_utils.assign_direction_num_to_string(
                        self.direction_group[ci],
                        self.direction_string_group[ci], counted_direction)

                _statistics = {"frame": single_imname.strip().split('/')[-1],
                               "time": now}
                _result_per_frame = {"frame": single_imname.strip().split('/')[-1],
                                     "time": now}

                for ci in self.num_class:
                    _cls_name = self.class_group[ci]
                    if _cls_name == "person" and np.sum(_current_count_group[ci]) != 0:
                        if type(self.bikespeed) is list or len(np.shape(self.bikespeed)) == 3:
                            _count_person, _count_bike, \
                                self.bike_id_group, \
                                self.pedestrian_id_group = count_utils.filter_out_bike_by_iou(
                                    self.direction_string_group[ci], self.roi_group[ci],
                                    _current_count_group[ci], self.bike_id_group, self.pedestrian_id_group,
                                    self.bikespeed)
                        else:
                            _count_person, _count_bike, \
                                self.bike_id_group, \
                                self.pedestrian_id_group = count_utils.filter_out_cyclist_from_person(
                                    self.direction_string_group[ci], self.speed_group[ci], _current_count_group[ci],
                                    self.bike_id_group, self.pedestrian_id_group, self.bikespeed)
                        self.count_move_pedestrain_group += _count_person
                        self.count_move_cyclists_group += _count_bike
                        _statistics.update({"count_move_pedestrian": _count_person.copy(),
                                            "count_move_bike": _count_bike.copy()})
                        _result_per_frame.update(
                            {"count_move_pedestrian": _count_person.copy(),
                             "count_move_bike": _count_bike.copy(),
                             "aggregate_movement_pedestrian": self.count_move_pedestrain_group.copy(),
                             "aggregate_movement_cyclists": self.count_move_cyclists_group.copy()})
                    elif _cls_name == "car" and np.sum(_current_count_group[ci]) != 0:
                        _keys = self.direction_string_group[ci].keys()
                        for _per_key in _keys:
                            self.car_id_group[_per_key] = self.direction_string_group[ci][_per_key]

                    _statistics.update({"rois_%s" % _cls_name: self.current_rois[ci],
                                        "current_person_id_%s" % _cls_name: self.current_person_id[ci],
                                        "current_new_index_%s" % _cls_name: self.new_object_index[ci],
                                        "count_%s" % _cls_name: self.count[ci],
                                        "count_move_%s" % _cls_name: _current_count_group[ci],
                                        "ep_%s" % _cls_name: _ep[ci],
                                        "direction_arrow_%s" % _cls_name: _direction_arrow_group[
                                            ci]})  # this is correct!
                    _result_per_frame.update({"current_person_id_%s" % _cls_name: self.current_person_id[ci],
                                              "current_new_index_%s" % _cls_name: self.new_object_index[ci],
                                              "count_%s" % _cls_name: self.count[ci],
                                              "count_move_%s" % _cls_name: _current_count_group[ci],
                                              "ep_%s" % _cls_name: _ep[ci],
                                              "direction_arrow_%s" % _cls_name: _direction_arrow_group[ci],
                                              "current_count_%s" % _cls_name: len(self.current_rois[ci]),
                                              "aggregate_count_%s" % _cls_name: self.count[ci].copy(),
                                              "aggregate_movement_%s" % _cls_name: self.count_movement_group[
                                                  ci].copy()})
                video_stat.append(_statistics)
                count_stat_for_saving.append(_result_per_frame)
                self.remove_disappeared_objects(_kf_remove_index)

                _direc_count_stat = [[line_group, self.count_movement_group[ci],
                                      _direction_arrow_group[ci]] for ci in self.num_class]
                if save_video_filename is not None or show is True:
                    _im_annotate = vt.put_multiple_stat_on_im(self.im, self.current_rois,
                                                              self.current_person_id, self.new_object_index,
                                                              self.count, _ep,
                                                              class_group=self.class_group,
                                                              show=show, show_direction_count=True,
                                                              direction_count_stat=_direc_count_stat,
                                                              im_index='_'.join(self.class_group) + "%d" % iterr,
                                                              resize=self.resize)
                    if save_video:
                        video_writer.write((_im_annotate * 255.0).astype('uint8'))
                if iterr % (save_box_stat - 1) == 0:
                    if iterr != 0 and stat_folder:
                        video_stat.append([self.pedestrian_id_group, self.bike_id_group, self.car_id_group])
                        pickle.dump(video_stat,
                                    open(stat_folder + "/%d" % iterr, 'wb'))
            if save_video:
                video_writer.release()
            if save_json_filename:
                count_stat_for_saving.append([self.pedestrian_id_group, self.bike_id_group, self.car_id_group])
                pickle.dump(count_stat_for_saving, open(save_json_filename, 'wb'))

#         except:
#             print("oops!", sys.exc_info()[0], "occurred.")
#             if save_video:
#                 video_writer.release()()
#             if save_json_filename:
#                 pickle.dump(count_stat_for_saving, open(save_json_filename, 'wb'))
