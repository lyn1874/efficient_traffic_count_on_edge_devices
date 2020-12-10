import os
import json
import pickle
import numpy as np
import cv2

# The information for the category
# id: label, it needs to be an int, starts from 1
# name, str, it needs to be a name

# These are the information for the images
# id: the index for the images
# filename: the filenames, end with jpg
# height: the height of the images
# width: the width of the images

# These are the information for the annotations
# image_id: the index for the images
# id: the index for the annotations, for example, if there is only one annotation in per image, then id = image_id
# category_id: the label, it has to be int,
# is_crowd: if it's 0, then it means it only shows one annotation in an image
# area: the bounding box size
# bbox: [x_top_left, y_top_left, width, height]
# segmentation: contains the x and y coordinates for the vertices of the polygon around every object instance for the segmentation masks, i don't have this value

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_item_id = 0
image_id = 0
annotation_id = 0

category_set = dict()
image_set = set()
    


def addCatItem(name, item_id):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id = item_id
#    category_item_id += 1

    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id


def addImgItem(file_name, size):
    global image_id
    if file_name == None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] == None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] == None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    if "cam" in file_name.split('/')[1]:
        bg_name = str(file_name.split('/')[1].split('_frame')[0])
    else:
        bg_name = str(file_name.split('/')[0])
    image_item['bg_name'] = bg_name

    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox, score, identity=None):
    """Args: bbox: [left_top_x, left_top_y, width, height]"""
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    #left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    #right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    #right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])
    area = bbox[2] * bbox[3]
        
    
    annotation_item['segmentation'].append(seg)
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_item['score'] = score
    if identity:
        annotation_item['identity'] = identity
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def create_dataset(im_folder, annotation_folder, camera_name, sequence_name, 
                   current_image_id, current_category_id, imsize, only_person, skip, skip_label=1,
                   num_seq = [0, -1], park_area=[]):
    """This function creates the ai flanders data in coco fashion
    im_folder: the folder that saves all the images
    annotation_folder: the predicted labels using the teacher model
    imsize: it differs between sequence to sequence, either [480, 640] or [720,1280]
    """
    anno_data = pickle.load(open(annotation_folder, 'rb'))
    if num_seq[-1] != -1:
        boxes = anno_data["boxes"][num_seq[0]:num_seq[-1]]
        label = anno_data["label"][num_seq[0]:num_seq[-1]]
        score = anno_data["score"][num_seq[0]:num_seq[-1]]
    else:
        boxes, label, score = anno_data["boxes"], anno_data["label"], anno_data["score"]
    print(len(boxes), len(label), len(score))
    if im_folder:
        all_im = []
        for v in sorted(os.listdir(im_folder)):
            if 'Sequence_' in v:
                _im = [v + '/' + q for iterr, q in enumerate(
                    sorted(os.listdir(im_folder + v))) if iterr % skip == 0 and '.jpg' in q]
                all_im.append(_im)
        all_im = [v for j in all_im for v in j]
        if num_seq[-1] != -1:
            all_im = np.array(all_im)[num_seq[0]:num_seq[-1]]
        else:
            all_im = np.array(all_im)
        print(im_folder, all_im[0]) 
        imsize = np.shape(cv2.imread(im_folder + all_im[0]))[:-1]
        height, width = imsize
        if only_person == "person":
            class_label = [0]
        elif only_person == "all":
            class_label = [0, 1, 3]
        elif only_person == "car":
            class_label = [2, 7]
        elif only_person == "car_truck":
            class_label = [0, 1]
        elif only_person == "ped_car":
            class_label = [0, 1]
        if only_person == "all":
            gt_label_group = np.array(["person", "car", "bike"])
            object_name_group = ["person", "car", " ", "bike"]
        elif only_person == "ped_car":
            gt_label_group = np.array(["person", "car"])
            object_name_group = ["person", "car"]
        else:
            object_name_group = ["car", "truck"]
            gt_label_group = np.array(["car", "truck"])
        
        height_width_group = [[height, width] for i in range(len(all_im))]
    else:
        object_name_group = ["car", "truck"]
        all_im = anno_data["filename"][num_seq[0]:num_seq[-1]]
        height_width_group = anno_data["heightwidth"]
        print(len(all_im), len(height_width_group))
        class_label = [0, 1]
        gt_label_group = np.array(["car", "truck"])
        imsize = [height_width_group[0][0], height_width_group[0][1], 3]
    i = 0
    num_save_image = 0
    print("There are %d images" % len(all_im))
    for iterr, single_im_name in enumerate(all_im):
        if iterr % skip_label != 0:
            continue
        size = dict()
        size['width'] = height_width_group[iterr][1]
        size['height'] = height_width_group[iterr][0]
        size['depth'] = 3
        single_box = boxes[iterr]
        single_label = label[iterr].astype(np.int32) # - 1
        single_score = score[iterr]
        if sequence_name + '/' + single_im_name not in image_set:
            current_image_id = addImgItem(sequence_name + '/' + single_im_name, size)
        else:
            raise Exception("duplicate images", sequence_name, single_im_name)
        if 3 in single_label:
            print(single_label)
        if np.shape(single_label)[0] > 0:
            for anno_index in range(np.shape(single_label)[0]):
                if single_label[anno_index] in class_label:
                    i += 1
                    object_name = object_name_group[single_label[anno_index]]
                    if object_name == "car" and np.sum(park_area) > 0:
                        _x_y = single_box[anno_index].copy()
                        _x_y[2] += _x_y[0]
                        _x_y[3] += _x_y[1]
                        if compute_intersection(park_area, _x_y) > 0.90:
                            continue
                    _pred_prob = int((single_score[anno_index]*100))
                    if object_name not in category_set:
                        print(gt_label_group, object_name)
                        item_id = int(np.where(gt_label_group == object_name)[0][0])+1
                        current_category_id = addCatItem(object_name, item_id)
                    else:
                        current_category_id = category_set[object_name]
                    bbox = []
                    for _box_index in range(4):
                        bbox.append(int(single_box[anno_index][_box_index]))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox, _pred_prob)
        num_save_image += 1
    for iterr, v in enumerate(gt_label_group):
        if v not in category_set:
            current_category_id = addCatItem(v, iterr+1)
    print("camera %s, sequence %s, there are %d frames are saved and %d frames include person" % (camera_name, 
                                                                                                  sequence_name, 
                                                                                                  num_save_image, 
                                                                                                  i))
    return current_image_id, current_category_id, imsize, all_im


def calc_background_image(im_folder, im_size, skip):
    if type(im_folder) == str:
        print(im_folder)
        im_tot = len(os.listdir(im_folder))
        if '.jpg' in os.listdir(im_folder)[0]:
            if skip == 0:
                all_file = [im_folder + v for v in os.listdir(im_folder)]
            else:
                all_file = [im_folder + v for v in os.listdir(im_folder)[::skip]]
        else:
            all_file = sorted([im_folder + v for v in os.listdir(im_folder) if 'Sequence' in v])
            all_im = []
            for single_file in all_file:
                _im = [single_file + '/' + v for v in sorted(os.listdir(single_file))[::skip]]
                all_im.append(_im)
            all_file = [v for j in all_im for v in j]                
    else:
        all_file = im_folder[::skip]
    print(all_file[0])
    print(all_file[-1])
    print("calculating bg from", np.shape(all_file), "images")
    bg = np.zeros([im_size[0], im_size[1], 3])
    for single_file in all_file:
        _im = cv2.imread(single_file)[:, :, ::-1]/255.0
        if np.shape(_im)[0] != im_size[0]:
            _im = cv2.resize(_im, dsize=tuple(im_size))
        bg += _im
    bg = bg/len(all_file)
    print("There are %d images from folder" % len(all_file))
    return bg


def create_antwerpen(camera_name, sequence, only_person, tr_or_tt, 
                     compound_coef, skip, skip_label, num_seq, park_area=[], save_bg=False):
    data_folder = '/home/jovyan/bo/dataset/%s/' % camera_name
    annotation_folder = data_folder + 'annotations/'
    if not os.path.exists(annotation_folder):
        os.makedirs(annotation_folder)
    current_image_id = None
    current_category_id = None
    imsize = np.shape(cv2.imread(data_folder+"/%s/%010d.jpg" % (sequence[0], 1)))[:-1]
    print("Image Size", imsize)
    for single_sequence in sequence:
        im_folder = data_folder + "%s/" % single_sequence
        gt_file = '/home/jovyan/bo/exp_data/teacher_result/%s/d%d_%s' % (camera_name, 
                                                                                compound_coef, single_sequence)
        print("The loaded prediction file is.................", gt_file)
        current_image_id, current_category_id, imsize, _ = create_dataset(im_folder, gt_file, 
                                                               camera_name, single_sequence,
                                                               current_image_id, current_category_id, 
                                                               imsize, only_person, skip, skip_label, num_seq, park_area)
        if not os.path.isfile(annotation_folder + 'bg_%s.npy' % single_sequence) and save_bg == True:
            bg = calc_background_image(im_folder, imsize, skip=60)
            np.save(annotation_folder + 'bg_%s.npy' % single_sequence, bg)
            cv2.imwrite(annotation_folder + 'bg_%s.jpg' % single_sequence, (bg * 255.0).astype('uint8')[:,:,::-1])
        
    json_folder = annotation_folder + "instances_%s.json" % tr_or_tt
    json.dump(coco, open(json_folder, 'w'))
    
    