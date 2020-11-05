import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import efficientdet.count_utils as count_utils
import efficientdet.utils as eff_utils
import efficientdet.kalmanfilter2 as kalman_scale


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def xyah_to_tlbr(xyah_input):
    xyah = xyah_input.copy()
    xyah[2:][xyah[2:] < 0] = 0.1
    xyah[2] = xyah[2] * xyah[3]
    xyah[:2] -= xyah[2:] / 2
    xyah[2:] = xyah[:2] + xyah[2:]
    return xyah

def tlbr_to_xyah(obs):
    if isinstance(obs, list):
        obs = np.array(obs)
    xyah_obs = obs.copy()
    xyah_obs[:, 2] -= xyah_obs[:, 0]
    xyah_obs[:, 3] -= xyah_obs[:, 1]
    xyah_obs[:, :2] += xyah_obs[:, 2:] / 2
    xyah_obs[:, 2] /= xyah_obs[:, 3]
    return xyah_obs


def kalman_init(observations, kalman_filter):
    """This function is used to initialize the trajectory
    observations: [N, 4], a list of bounding boxes predictions in the first step, or in the middle
    Return:
    each element in traject is a list, which shows the [state, and error in the estimation]
    """
    obs_xyah = tlbr_to_xyah(observations)
    traject = [kalman_filter.initiate(v) for v in obs_xyah]
    return traject


def kalman_init_scale(observations):
    traject = [kalman_scale.KalmanBoxTracker(v) for v in observations]
    return traject
    
    
def kalman_traject(kalman_filter, observation, mean, cov):
    """Give trajectory of an object with kalman filter
    Args:
        kalman_filter: the kalman class state for all the past scenes
        observation: if observation is not [], then it means this object has been detected, 
        so the predictions will be corrected based on the observation. Otherwise, I am 
        only doing the prediction step 
        mean: the state vector from previous step [8]
        cov: the covaraince matrix from previous step [8x8]        
    """
    if len(observation) == 0:
        mean, cov = kalman_filter.predict(mean, cov)
        pred_box = xyah_to_tlbr(mean[:4])
    else:
        mean, cov = kalman_filter.predict(mean, cov)
        obs_xyah = tlbr_to_xyah([observation])
        mean, cov = kalman_filter.update(mean, cov, obs_xyah[0])
        pred_box = xyah_to_tlbr(mean[:4])
    return pred_box, mean, cov


def kalman_traject_scale(kalman_traject, observation):
    if len(observation) == 0:
        pos = kalman_traject.predict()
    else:
        pos = kalman_traject.predict()
        kalman_traject.update(observation)
    return pos, kalman_traject


def invert_invert_affine(roi_orig_scale, framed_metas=[(1536, 1152, 960, 720, 0, 384)]):
    new_w, new_h, old_w, old_h, padding_w, padding_h = framed_metas[0]
    roi_input_scale = roi_orig_scale.copy()
    roi_input_scale[[0, 2]] = roi_input_scale[[0, 2]] * (new_w / old_w)
    roi_input_scale[[1, 3]] = roi_input_scale[[1, 3]] * (new_h / old_h)
    return roi_input_scale

    
def bucketize(tensor, bucket_boundaries):
    result = torch.zeros_like(tensor, dtype=torch.int32)
    for boundary in bucket_boundaries:
        result += ((tensor > boundary).int())
    result = [v-1 for v in result]
    return result


def get_num_feature(imsize, num_anchors):
    num_block = 5
    num_feat = torch.zeros([num_block+1], requires_grad=False)
    for i in range(num_block):
        _feat = (imsize // 2**(3+i))**2 * num_anchors
        num_feat[i+1] = _feat
    num_feat = torch.cumsum(num_feat, dim=0)
    return num_feat


def get_csh(rois_xy):
    rois = rois_xy.copy()
    rois[:, 2:] -= rois[:, :2]   # hw
    rois[:, :2] += rois[:, 2:]/2.0  # center
    return rois

def get_xy(roi_csh):
    rois = roi_csh.copy()
    rois[:, 2:][rois[:, 2:] == 0] = 0.1
    rois[:, :2] -= rois[:, 2:] / 2  #xy    
    rois[:, 2:] += rois[:, :2]  #xy
    rois[:, 2:] += 1
    return rois.astype(np.int)


def get_featuremap_wrt_rois(feature_maps, rois, imsize, show_figure=False):
    """Extracts the feature maps with respect to the rois
    Args:
        feature_maps: a list of feature map, each feature map is with shape [batch_size, f_ch, fh, fw]
        rois: coordinates of the detected boxes: [N, 4]
        imsize: the input size
        show_figure: bool, 
    """
    fh = [np.shape(v)[-1] for v in feature_maps]
    rois_centerhw = get_csh(rois)
    scale = [imsize / v for v in fh]
    num_features = len(feature_maps)
    aggregate_feature_group = [[] for _ in range(len(rois))]
    roi_multi_levels = []
    for i in range(num_features):
        roi_ = get_xy(rois_centerhw / scale[i])  # N, 4
        for j, single_roi in enumerate(roi_):
            _feat = feature_maps[i][0, :, single_roi[1]:single_roi[3], single_roi[0]:single_roi[2]]
            aggregate_feature_group[j].append(_feat)
        roi_multi_levels.append(roi_)
    if show_figure is True:
        feature_maps_npy = [np.sum(v[0], axis=0) for v in feature_maps]
        for i in range(len(rois)):
            fig = plt.figure(figsize=(12, 7))
            for j in range(5):
                _roi = get_xy(rois_centerhw[i:(i+1)] / scale[j])[0]
                ax = fig.add_subplot(1, 5, j + 1)
                _feat = feature_maps_npy[j]
                _feat = (_feat - np.min(_feat)) / (np.max(_feat) - np.min(_feat))
                ax.imshow(_feat)
                ax.plot([_roi[0], _roi[2]], [_roi[1], _roi[1]], color='r')
                ax.plot([_roi[0], _roi[2]], [_roi[3], _roi[3]], color='r')
                ax.plot([_roi[0], _roi[0]], [_roi[1], _roi[3]], color='r')
                ax.plot([_roi[2], _roi[2]], [_roi[1], _roi[3]], color='r')
        return aggregate_feature_group, roi_multi_levels
    else:
        return aggregate_feature_group
    

def calc_euc_dist_roi(roi_0, roi_1):
    """This function calculates the euclidean distance between bounding boxes
    roi_0: [x1, y1, x2, y2] where x1,y1 are the coordinates for the left top 
    and x2, y2 are the coordinates for the right bottom
    roi_1: the same
    I will just calculate the distance between the center coordinate of two rois
    """
    x0_0, y0_0, x0_1, y0_1 = roi_0
    x1_0, y1_0, x1_1, y1_1 = roi_1
    roi_0_center = [x0_0 + (x0_1 - x0_0) / 2, y0_0 + (y0_1 - y0_0) / 2]
    roi_1_center = [x1_0 + (x1_1 - x1_0) / 2, y1_0 + (y1_1 - y1_0) / 2]
    dist = (roi_0_center[0] - roi_1_center[0]) ** 2 + (roi_0_center[1] - roi_1_center[1]) ** 2
    return np.sqrt(dist)


def calc_euc_dist_roi_group(roi0_group, roi1_group):
    dist = torch.zeros([len(roi0_group), len(roi1_group)])
    for i, single_roi0 in enumerate(roi0_group):
        for j, single_roi1 in enumerate(roi1_group):
            d = calc_euc_dist_roi(single_roi0, single_roi1)
            dist[i, j] = d
    return dist


def calc_roi_std(q):
    roi = q.copy()
    roi[:, 0] += (roi[:, 2] - roi[:, 0])/2
    roi[:, 1] += (roi[:, 3] - roi[:, 1])/2
    std = np.std(roi[:, :2], axis=0)
    return std



def calc_feature_similarity(old_feature_embedding, old_anchor_index, 
                            new_feature_embedding, new_anchor_index, dist_option):
    num_new, num_old = len(new_anchor_index), len(old_anchor_index)
    similarity_new_old = np.zeros([num_new, num_old])
    for i in range(num_new):
        _per_new_box_sim = []
        for j in range(num_old):
            if new_anchor_index[i] == old_anchor_index[j]:
                anchor_group = [new_anchor_index[i]]
            else:
                anchor_group = [old_anchor_index[j], new_anchor_index[i]]
            _old_feat = [old_feature_embedding[j][v] for v in anchor_group]
            _new_feat = [new_feature_embedding[i][v] for v in anchor_group]
            _sim = calc_eucdist2(_old_feat, _new_feat, dist_option)
            similarity_new_old[i,j] = np.mean(_sim)
    return similarity_new_old


def calc_eucdist2(q0, q1, dist_metric="euc"):
    """Caculates the euclidean distance between feature maps 
    """
    if len(q0) > 10:
        q0 = [q0]
    if len(q1) > 10:
        q1 = [q1]
    dist = np.zeros([len(q0)])
    iterr = 0
    for s_q0, s_q1 in zip(q0, q1):
        s_q0 = torch.from_numpy(s_q0)
        s_q1 = torch.from_numpy(s_q1)
        if s_q1.size() != s_q0.size():
            s_q0 = F.interpolate(s_q0.unsqueeze(0), s_q1.size()[1:]).squeeze(0)
        if dist_metric is "euc":
            diff = torch.sqrt((s_q0 - s_q1)**2).sum().div(np.prod(s_q1.size()))
            sim = 1/(1+diff)
        elif dist_metric is "cos":
            s_q0 = s_q0.reshape([1, -1])
            s_q1 = s_q1.reshape([1, -1])
            sim = nn.CosineSimilarity(dim=1)(s_q0, s_q1)
        del s_q0, s_q1
        dist[iterr] = sim.detach().cpu().numpy()
        iterr+=1
    return dist
        

def get_std_for_detection(image, rois):
    """Calculates the standard deviation for per detected bounding boxes
    image: [imh, imw, ch]
    rois: [N, 4], x1, y1, x2, y2, maybe float
    """
    if isinstance(rois, list):
        rois = np.array(rois)
    im_cut = [image[v[1]:v[3], v[0]:v[2]] for v in rois.astype(np.int)]
    std_value = [np.std(v)*100 for v in im_cut]
    return np.array(std_value)


def get_conf_std(single_std_group, predict_std, conf_crit=3):  #50 before
    """Get the confidence interval for the standard deviation
    single_std_group: [n]
    predict_std: single_value
    Note: for Antwerpen dataset, conf_crit = 15
    for mot17 dataset, conf_crit = 20
    """
#    single_std_group = single_std_group[single_std_group!=0]
#    print("---object has appeared %d times" % len(single_std_group))
    if len(single_std_group) == 1:
        if abs(single_std_group[0] - predict_std) > 4:
            crit = "cancel"
        else:
            crit = "keep"
    else:
        _conf = conf_crit * np.std(single_std_group) / np.sqrt(len(single_std_group))
        _conf = [1.0 if _conf < 1.0 else _conf][0]  # before it's 2.0 Aug-10
        lower = np.mean(single_std_group) - _conf
        upper = np.mean(single_std_group) + _conf
        if predict_std+1 < lower or predict_std-1 > upper:
            crit = "cancel"
        else:
            crit = "keep"
#         print("avg %.2f" % np.mean(single_std_group),
#               "std %.4f" % (np.std(single_std_group)/np.sqrt(len(single_std_group))),
#               "upper %.2f" % upper, "lower %.2f" % lower, "actual %.2f" % predict_std)

    return crit


def update_state(old_trajectory_stat, old_anchor_index, old_feature_embedding, person_id_old, old_rois, old_std_value,
                 _roi_index, _new_roi, _new_anchor_index, _new_feature_map, _new_std_value, current_person_id, count,
                 kalman_filter, use_feature_maps=True, activate_kalman_filter=True):
    person_id_old = np.concatenate([person_id_old, [count+1]], axis=0)
    old_rois = np.concatenate([old_rois, [_new_roi]], axis=0)
    old_std_value = np.concatenate([old_std_value, [_new_std_value]], axis=0)
    if activate_kalman_filter:
        _new_traject = get_kalman_init([_new_roi], kalman_filter)
        old_trajectory_stat.append(_new_traject[0])
    if use_feature_maps:
        old_feature_embedding.append(_new_feature_map)
        old_anchor_index = np.concatenate([old_anchor_index, [_new_anchor_index]], axis=0)
    current_person_id[_roi_index] = count + 1
    stat = [old_trajectory_stat, old_anchor_index, old_feature_embedding, person_id_old, old_rois, 
           old_std_value, current_person_id, count + 1]
    return stat


def get_kalman_init(rois, kalman_filter):
    if kalman_filter == 0:
        _new_traject = kalman_init_scale(rois)
    else:
        _new_traject = kalman_init(rois, kalman_filter)
    return _new_traject


def get_kalman_traject(single_old_trajectory_stat, _new_roi, kalman_filter):
    if isinstance(single_old_trajectory_stat, tuple):
        _v0, _v1 = single_old_trajectory_stat[0].copy(), single_old_trajectory_stat[1].copy()
        pos, _mean, _cov = kalman_traject(kalman_filter, _new_roi, _v0, _v1) 
        single_old_trajectory_stat = tuple([_mean, _cov])
    else:
        pos, single_old_trajectory_stat = kalman_traject_scale(single_old_trajectory_stat, 
                                                               _new_roi)
        pos = pos[0]
    return pos, single_old_trajectory_stat
        

def restore_state(old_trajectory_stat, old_anchor_index, old_feature_embedding, person_id_old, old_rois, old_std_value,
                  _old_roi_index, _roi_index, _new_roi, _new_anchor_index, _new_feature_map, _new_std_value, 
                  current_person_id, kalman_filter, use_feature_maps=True, activate_kalman_filter=True):
    if activate_kalman_filter:
        pos, single_old_trajectory_stat = get_kalman_traject(old_trajectory_stat[_old_roi_index], 
                                                             _new_roi, kalman_filter)
        old_trajectory_stat[_old_roi_index] = single_old_trajectory_stat
    old_rois[_old_roi_index] = _new_roi
    current_person_id[_roi_index] = person_id_old[_old_roi_index]
    old_std_value[_old_roi_index] = _new_std_value
    if use_feature_maps:
        old_feature_embedding[_old_roi_index] = _new_feature_map
        old_anchor_index[_old_roi_index] = _new_anchor_index
    return old_trajectory_stat, old_anchor_index, old_feature_embedding, old_rois, old_std_value, current_person_id


def calc_speed_based_roi_loc(roi_group):
    if len(roi_group) > 1:
        center = roi_group[:, :2] + (roi_group[:, 2:] - roi_group[:, :2])/2
        diff = center[-1] - center[0]
        speed = np.sqrt(diff[0] ** 2 + diff[1] ** 2) / len(center)
    else:
        speed = 0
    return speed


def cancel_object(old_rois, old_trajectory_stat, old_feature_embedding, 
                  old_anchor_index, old_std_value, person_id_old, 
                  old_percentage, _kept_id, counting, use_feature_maps=True, 
                  activate_kalman_filter=True):
    if type(_kept_id) is int:
        _kept_id = np.arange(len(old_rois))[-_kept_id:]
    old_rois = old_rois[_kept_id]
    person_id_old = person_id_old[_kept_id]
    old_std_value = old_std_value[_kept_id]
    if activate_kalman_filter:
        old_trajectory_stat = [old_trajectory_stat[v] for v in _kept_id]
    
    if use_feature_maps:
        old_feature_embedding = [old_feature_embedding[v] for v in _kept_id]
        old_anchor_index = old_anchor_index[_kept_id]
    if counting is True:
        old_percentage = old_percentage[_kept_id]
    return old_rois, old_trajectory_stat, old_feature_embedding, \
        old_anchor_index, old_std_value, person_id_old, old_percentage


def recover_missing_route(im, object_index_nooverlap, old_rois, old_trajectory_stat, old_feature_embedding, 
                          old_anchor_index, std_group, person_id_old, new_features, 
                          framed_metas, kalman_filter, dist_option, x_y_threshold, use_feature_maps=True):
    """Predicts the current location of an object that is not detected by the detector
    Args:
        im: input image, [imh, imw, ch], [0, 1]
        object_index_nooverlap: numeric index for the old rois, person_id_old[old_object_index] is the actual person id
        old_rois: [N, 4]
        old_trajectory_stat: a list of [N] elements, each element contains the previous location and velocity
        old_feature_embedding: a list of [N] elements, each element contains 5 feature maps for per object
        old_anchor_index: [N], an array
        std_group: [num_images, N]
    """
#     print("%d people lost their track" % len(object_index_nooverlap))
    input_size = framed_metas[0][0]
    imh, imw = framed_metas[0][3], framed_metas[0][2]
    _miss_detect_roi = []
    _person_id_for_miss_detection = []
    _removed_index = []
    _jumping_object = []
    for single in object_index_nooverlap:
#         print("---person %d lost its track" % person_id_old[single])
        _old_box = old_rois[single]
        _new_pos, _single_old_trajectory_stat = get_kalman_traject(old_trajectory_stat[single], 
                                                                   [], kalman_filter)
        _new_pos[_new_pos < 0] = 0
        _pos_back_input_scale = invert_invert_affine(_new_pos, framed_metas)
        if int(_new_pos[2] - _new_pos[0]) == 0 or int(_new_pos[3] - _new_pos[1]) == 0 or \
            int(_new_pos[0]) > imw or int(_new_pos[1]) > imh or \
            int(_pos_back_input_scale[2] - _pos_back_input_scale[0]) == 0 or \
            int(_pos_back_input_scale[3] - _pos_back_input_scale[1]) == 0 or \
            int(_new_pos[2]) > x_y_threshold[0] or \
            int(_new_pos[3]) < x_y_threshold[1] or \
            int(_pos_back_input_scale[0] + 10) >= input_size or \
            int(_pos_back_input_scale[1] + 10) >= input_size :
#             print(": the predicted boxes are outside of the scene, \
#                     so omit the prediction and remove its statistics", _new_pos, _pos_back_input_scale)
            _removed_index.append(person_id_old[single])
        else:            
            _std_value = get_std_for_detection(im, [_new_pos])[0]
            _crit = get_conf_std(std_group["id%d" % person_id_old[single]], _std_value, 3) # before it's 2, for CAM16 On aicity, I am using 20
    #             print("---after checking the standard deviation----")
            if _crit is "cancel":
    #                 print("the standard deviation is far away from its past", _std_value, std_group["id%d" % person_id_old[single]])
    #                 print(": the predicted box is far away from the actual location, \
    #                     so omit the prediction and remove statistics")
    #                _jumping_object.append(single)
                pass
            else:        
                if use_feature_maps:
                    _aggregate_feature = get_featuremap_wrt_rois(new_features, 
                                                                 np.array([_pos_back_input_scale]), 
                                                                 input_size, show_figure=False)
                    _dist = calc_eucdist2(old_feature_embedding[single], _aggregate_feature[0], dist_option)
                    _value_use = old_anchor_index[single]
                    del _aggregate_feature
    #                 print(": similarity between predicted and actual features %.2f" % _dist[old_anchor_index[single]])
                else:
                    _value_use = 1
                if _value_use <= 0:
    #                   print(": similarity is lower than threshold, so omit the prediction and remove statistics")
                    pass
    #                    _removed_index.append(single)  # this is not correct!
                else:
    #                     print(": this person is still in the scene but detector fails to detect")
                    old_trajectory_stat[single] = _single_old_trajectory_stat
                    _miss_detect_roi.append(_new_pos)  # the miss detections, these miss detections are used to update
                    old_rois[single] = _new_pos  # this is the estimated location of this object
                    _person_id_for_miss_detection.append(person_id_old[single])  

    return _miss_detect_roi, _person_id_for_miss_detection, _removed_index, \
        _jumping_object, old_rois, old_trajectory_stat
