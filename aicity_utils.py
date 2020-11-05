# understand AIC dataset
import os
import numpy as np
import cv2


def save_frame_from_video(video_path, path2write):
    if not os.path.exists(path2write):
        os.makedirs(path2write)
    cap = cv2.VideoCapture(video_path)
    i = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(path2write + '/frame_%08d.jpg' % i, frame)
        i += 1 
    cap.release()
    cv2.destroyAllWindows()
    print("There are %d frames with fps %d" % (i, fps))
    return i
    
    
def load_fps_frame_number(path):
    content = open(path, 'r').read().rstrip('\n').split('\n')
    if 'track1_vid_stats' in path:
        content = content[1:]
    content = [v.split('\t') for v in content]
    return content


def save_frames(txt_path, video_path):
    """video_path: /project_scratch/bo/....../Dataset_A/"""
    frame_dir = '/'.join(video_path.strip().split('/Dataset')[:-1]) + "/Dataset_A_Frame/"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    content = load_fps_frame_number(txt_path)
    tot_frame = 0
    for single_content in content:
        _video = video_path + single_content[0]
        print("--------------------camera %s----------------------" % single_content[0])
        print("The actual number of frames and fps is ", single_content[2], single_content[1])
        _subdir = frame_dir + single_content[0].split('.mp4')[0]
        _single_num_frame = save_frame_from_video(_video, _subdir)
        print("----------------------------------------------------")
        tot_frame += _single_num_frame
    print("There are %d frames in total" % tot_frame)
    

def create_mask_for_per_camera():
    path = "/project_scratch/bo/normal_data/aic2020/AIC20_track1/ROIs/"
    shapepath = '/'.join(path.split("ROIs")[:-1]) + "screen_shot_with_roi_and_movement/"
    print("path to read roi", path)
    print("path to read shape", shapepath)
    path_group = [v for v in os.listdir(shapepath) if 'dawn' in v or 'rain' in v or 'snow' in v]
    print(path_group)
    for i in range(21)[-1:]:
        with open(path + "cam_%d.txt" % i, 'r') as f:
            c = f.readlines()
            c = [v.split('\n')[0].split(',') for v in c]
            c = np.array([[int(v[0]), int(v[1])] for v in c]).astype('int32')
        f.close()
        shape = np.shape(cv2.imread(shapepath + "cam_%d.jpg" % i))
        mask = np.zeros(shape).astype('uint8')
        pts = c
        pts = pts.reshape((-1,1,2))
        mask = cv2.fillPoly(mask, [pts], color=(255, 255, 255))
        mask = np.sum(mask, axis=-1, keepdims=-1)
        mask = (mask != 0).astype('int32')
        np.save(path + "cam_%d" % i, mask)
        print("Finish camera %d....." % i)
        
    
def give_filename_id_from_json(tr_json):
    filename_from_json = [v['file_name'] for v in tr_json['images']]
    h_w = [[int(v['height']), int(v['width'])] for v in tr_json["images"]]
    fileid_from_json = [v['id'] for v in tr_json['images']]
    fileid_from_annotation = np.array([v['image_id'] for v in tr_json['annotations']])
    bbox_from_annotation = [v['bbox'] for v in tr_json['annotations']]
    label_from_annotation = [v['category_id'] - 1 for v in tr_json["annotations"]]
    return filename_from_json, fileid_from_json, \
        fileid_from_annotation, bbox_from_annotation, label_from_annotation, h_w


def prepare_labeled_validation_data():
    """Creats the training and test data
    The labeled images are acquired from HCMUs team
    """
    path_mom = "/project_scratch/bo/normal_data/aic2020/"
    tr_json = json.load(open(path_mom + 'annotations_2/instances_train_car.json', 'rb'))
    val_json = json.load(open(path_mom + 'annotations_2/instances_val_car.json', 'rb'))

    tr_filename, tr_fileid, tr_annot_fileid, \
        tr_box, tr_lab, tr_h_w = give_filename_id_from_json(tr_json)
    val_filename, val_fileid, val_annot_fileid, \
        val_box, val_lab, val_h_w = give_filename_id_from_json(val_json)

    tot_filename = np.concatenate([tr_filename, val_filename], axis=0)
    tot_fileid = np.concatenate([tr_fileid, val_fileid], axis=0)
    tot_annot_id = np.concatenate([tr_annot_fileid, val_annot_fileid], axis=0)
    tot_box = np.concatenate([tr_box, val_box], axis=0)
    tot_lab = np.concatenate([tr_lab, val_lab], axis=0)
    tot_h_w = np.concatenate([tr_h_w, val_h_w], axis=0)
    print(len(tot_filename), len(tot_fileid), len(tot_annot_id), len(tot_box), len(tot_lab))
    
    cam_string_group = np.array([v.strip().split('_frame')[0] for v in tot_filename])
    tot_frame = 0
    filename_list_path = "/project/bo/normal_data/aic2020/AIC20_track1/train_filename_list/"
    if not os.path.exists(filename_list_path):
        os.makedirs(filename_list_path)

    use_string0 = ["1_dawn", "1_rain", "2", "2_rain", "3", "3_rain"]
    use_string1 = ["%d" % i for i in range(21)[10:]]
    use_string = np.concatenate([use_string0, use_string1], axis=0)

    cam_use = ["cam_%s" % s for s in use_string]
    for single_cam in cam_use:
        cam_subfile = np.array(sorted(os.listdir(path_mom + 'AIC20_track1/Dataset_A_Frame/%s/' % single_cam)))
        subset = np.where(cam_string_group == single_cam)[0]
        fileid_sub = tot_fileid[subset]
        bbox_subset, lab_subset = [], []
        for i in fileid_sub:
            _l = np.where(tot_annot_id == i)[0]
            bbox_subset.append([list(tot_box[j]) for j in _l])
            lab_subset.append([int(v) for v in tot_lab[_l]])
        test_index = [int(v.strip().split('.')[0].split("frame_")[1]) // 90 for v in tot_filename[subset]]
        pred_stat = {}
        filename_subset = [str(v) for v in tot_filename[subset]]
        sort_index = np.argsort(filename_subset)
        print("--first filename", filename_subset[sort_index[0]])
        print("--last filename", filename_subset[sort_index[-1]])
        pred_stat["filename"] = [filename_subset[i] for i in sort_index]
        pred_stat["boxes"] = [bbox_subset[i] for i in sort_index]
        pred_stat["score"] = [np.ones(len(v)).astype('float32') for v in pred_stat["boxes"]]
        pred_stat["label"] = [lab_subset[i] for i in sort_index]
        pred_stat["heightwidth"] = [[int(v[0]), int(v[1])] for v in tot_h_w[subset]]
        tot_frame += len(subset)
        json_file = path_mom + "AIC20_track1/Dataset_A_Frame/annotations/%s_manual_label" % single_cam
        pickle.dump(pred_stat, open(json_file, 'wb'))
        cam_subfile = cam_subfile[np.delete(np.arange(len(cam_subfile)), test_index)]

        with open(filename_list_path + "%s.txt" % single_cam, 'w') as f:
            [f.writelines(v+'\n') for v in cam_subfile]
        f.close()
        print("Finish %s..................." % single_cam, len(pred_stat["filename"]), len(cam_subfile))
    print("Total ", tot_frame)
    
    
def main():
    txt_file = '/project_scratch/bo/normal_data/aic2020/AIC20_track1/Dataset_A/track1_vid_stats.txt'
    video_path = '/project_scratch/bo/normal_data/aic2020/AIC20_track1/Dataset_A/'
    save_frames(txt_file, video_path)