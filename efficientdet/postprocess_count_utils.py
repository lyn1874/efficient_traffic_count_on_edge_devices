import numpy as np


def assign_count_index(id_group, class_name, movement_string):
    id_use = id_group.copy()
    keys_group = [v for v in id_use.keys()]
    move_value = np.array([id_use[key][0] for key in keys_group])
    if not movement_string:
        movement_string = ["up", "down", "left", "right"]
    for _direc_index, single_direc in enumerate(movement_string):
        _subindex = np.where(move_value == single_direc)[0]
        for iterr, single_index in enumerate(_subindex):
            _temp = [class_name, _direc_index + 1, single_direc, iterr + 1]
            id_use[keys_group[single_index]] = _temp
    return id_use


def concatenate_id(id_group, class_name, movement_string):
    tot_id = {}
    for iterr, single_id in enumerate(id_group):
        _id = assign_count_index(single_id, class_name[iterr], movement_string)
        tot_id.update(_id)
    return tot_id


def rearrange_stat(stat_original, only_person, class_group, algo, movement_string, return_concat_id=False):
    stat = stat_original.copy()
    if not movement_string:
        movement_string = ["static", "up", "down", "left", "right"]
    if only_person is not "car":
        tot_id_ped_bike = concatenate_id(stat[-1][:-1], ["pedestrian", "bike"], movement_string[1:])
        id_ped_bike_numeric = np.array([int(v.split("id")[1]) for v in tot_id_ped_bike.keys()])
    if only_person is not "person":
        tot_id_car = concatenate_id(stat[-1][-1:], ["car"], movement_string[1:])
        id_car_numeric = np.array([int(v.split("id")[1]) for v in tot_id_car.keys()])
    if only_person is "person":
        tot_id = [tot_id_ped_bike]
        num_id = [id_ped_bike_numeric]
    elif only_person is "car":
        tot_id = [tot_id_car]
        num_id = [id_car_numeric]
    elif only_person is "ped_car":
        tot_id = [tot_id_ped_bike, tot_id_car]
        num_id = [id_ped_bike_numeric, id_car_numeric]
    if return_concat_id:
        return tot_id, num_id
    count_stat_frameindex = []
    for iterr, single_stat in enumerate(stat[:-1]):
        for cls_index, single_class in enumerate(class_group):
            if "current_person_id_%s" % single_class in single_stat.keys():
                q = single_stat["current_person_id_%s" % single_class]
                direc_arrow = single_stat["direction_arrow_%s" % single_class]
                count_id_move = np.zeros([len(q)])
                identity = []
                if len(q) > 0:
                    for _qiter, _q in enumerate(q):
                        if algo is "angle":
                            if _q in num_id[cls_index]:
                                _va = tot_id[cls_index]["id%d" % _q]
                                count_id_move[_qiter] = _va[-1]
                                single_stat["direction_arrow_%s" % single_class][_qiter] = _va[1]
                                identity.append(_va[0])
                                count_stat_frameindex.append(
                                    [single_stat["frame"], _q, movement_string[_va[1]], _va[0]])
                        elif algo is "boundary":
                            if _q in num_id[cls_index] and direc_arrow[_qiter] != 0.0:
                                _va = tot_id[cls_index]["id%d" % _q]
                                count_id_move[_qiter] = _va[-1]
                                single_stat["direction_arrow_%s" % single_class][_qiter] = _va[1]
                                identity.append(_va[0])
                                count_stat_frameindex.append(
                                    [single_stat["frame"], _q, movement_string[_va[1]], _va[0]])

            else:
                count_id_move = []
                identity = []
            single_stat.update({"count_id_%s" % single_class: count_id_move,
                                "identity_%s" % single_class: identity})
    return stat, count_stat_frameindex


def give_count(_stat_re, only_person, movement_string):
    count_num = np.zeros([len(_stat_re) - 1, 3, len(movement_string)])
    time_use = []
    for i, s_s in enumerate(_stat_re[:-1]):
        time_use.append(s_s['time'].strip().split(' ')[3])
    if only_person != "car":
        cls_g = ["person"]
        special_name = ["pedestrian", "bike"]
        for i, s_s in enumerate(_stat_re[:-1]):
            for cls_v in cls_g:
                iden = np.array(s_s["identity_%s" % cls_v])
                direction_arrow = s_s["direction_arrow_%s" % cls_v]
                count_id = s_s["count_id_%s" % cls_v]
                direct_nonzero = np.where(direction_arrow > 0)[0]
                direc_subset = direction_arrow[direct_nonzero]
                count_id = count_id[direct_nonzero]
                for cls_index, _sname in enumerate(special_name):
                    count_num[i, cls_index] = count_num[i - 1, cls_index].copy()
                    if len(iden) > 0:
                        _index = np.where(iden == _sname)[0]
                        if len(_index) > 0:
                            for _s_direc in np.unique(direc_subset[_index]):
                                count_num[i, cls_index, int(_s_direc)] = np.max(
                                    count_id[_index][direc_subset[_index] == _s_direc])
    if only_person != "person":
        for i, s_s in enumerate(_stat_re[:-1]):
            iden = np.array(s_s["identity_car"])
            direction_arrow = s_s["direction_arrow_car"]
            count_id = s_s["count_id_car"]
            count_num[i, 2, :] = count_num[i - 1, 2, :].copy()
            if len(count_id) > 0:
                direct_nonzero = np.where(direction_arrow > 0)[0]
                direc_subset = direction_arrow[direct_nonzero]
                count_id = count_id[direct_nonzero]
                if len(iden) > 0:
                    _index = np.where(iden == "car")[0]
                    for _s_direc in np.unique(direc_subset[_index]):
                        count_num[i, 2, int(_s_direc)] = np.max(count_id[_index][direc_subset[_index] == _s_direc])

    return count_num, time_use
