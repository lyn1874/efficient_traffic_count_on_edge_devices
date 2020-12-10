import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, subtract_bg, set='train2017', imh_imw = [480, 640], 
                 transform=None):
        self.root_dir = root_dir
        self.imh_imw = imh_imw
        self.set_name = set
        self.transform = transform
        self.subtract_bg = subtract_bg
        
        if '/mot' not in self.root_dir:       
            self.bg_path = self.root_dir + '/annotations/'
            if self.subtract_bg:
                self.load_bg()

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))  
        self.image_ids = self.coco.getImgIds()
        self.load_classes()
    
    def load_bg(self):
        bg_npy = sorted([v for v in os.listdir(self.bg_path) if 'bg_' in v and '.npy' in v])
        self.bg_filename_group = bg_npy
        self.bg_group = [np.load(self.bg_path + '/' + v) for v in bg_npy]
        if self.subtract_bg is False:
            self.bg_group = [np.zeros(np.shape(v)) for v in self.bg_group]
        print("----subtracking background...............")
        [print("background max %.2f" % np.max(v), " min %.2f" % np.min(v)) for v in self.bg_group]
        self.bg_filename_group = np.array([v.strip().split('bg_')[1].strip().split('.npy')[0] for v
                                  in self.bg_filename_group])
        print(self.bg_filename_group)
        
    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)
        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, image_info['file_name'])
        if self.subtract_bg == True:
            _ind = np.where(self.bg_filename_group == image_info['bg_name'])[0]
            bg_use = self.bg_group[_ind[0]]
        else:
            bg_use = [0.0, 0.0, 0.0]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # here it transfers it to rgb
        img = img.astype(np.float32) / 255.0
        img = img - bg_use  # rgb [-1, 1]
        return img
    
    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)
        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations
    
    
    def patch_annotation_on_empty_im(self, img, ca_id):
        imh, imw = self.imh_imw
        _rand_ = np.random.choice(len(self.people), 1)[0]
        rand_person = self.people[_rand_]
        wid, hei = np.shape(rand_person)[:-1]
        x1n, y1n = [np.random.randint(0, imh-wid, [1])[0], np.random.randint(0, imw-hei, [1])[0]]
        x2n, y2n = [x1n + wid, y1n + hei]
        img[x1n:x2n, y1n:y2n] = rand_person
        annotations = np.reshape([y1n, x1n, y2n, x2n, ca_id], [1, 5]).astype('float64')
        return img, annotations        

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample
    

class Augmenter_H(object):
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[::-1, :, :]
            rows, cols, channels = image.shape

            x1 = annots[:, 1].copy()
            x2 = annots[:, 3].copy()

            x_tmp = x1.copy()

            annots[:, 1] = rows - x2
            annots[:, 3] = rows - x_tmp
            sample = {'img': image, 'annot': annots}

        return sample
        


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], background_im=[0.0, 0.0, 0.0]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])
        self.background_im = background_im

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = image.astype(np.float32)
        if np.sum(self.background_im) != 0:
            image = image - self.background_im

        return {'img': ((image - self.mean) / self.std), 'annot': annots}
    
    
    