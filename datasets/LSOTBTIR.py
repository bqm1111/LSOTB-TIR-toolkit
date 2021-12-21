import io
import os
import glob
from itertools import chain
import numpy as np
import six
from got10k.experiments import otb
from got10k.datasets import got10k
import xml.etree.ElementTree as ET
from collections import defaultdict
import cv2


class LSOTBTIR(object):
    __lsotb_tir_val_seqs = ['car_D_005', 'bus_V_003', 'boy_S_002', 'person_H_012', 'head_H_001',
                            'crowd_S_001', 'woman_H_001', 'person_S_003', 'person_D_020',
                            'person_D_022', 'hog_H_002', 'street_S_002', 'person_S_016',
                            'person_S_006', 'car_V_009', 'car_D_007', 'person_S_001',
                            'person_S_015', 'helicopter_H_002', 'person_H_003', 'person_S_018',
                            'person_S_004', 'motobiker_V_001', 'person_H_009', 'car_S_001',
                            'bird_H_001', 'person_D_009', 'dog_D_002', 'drone_D_001', 'boat_D_001',
                            'boy_S_001', 'car_V_006', 'person_S_012', 'person_V_008', 'car_D_004',
                            'bus_V_004', 'person_D_003', 'person_S_002', 'person_H_006',
                            'person_S_017', 'car_S_002', 'hog_H_003', 'person_D_021', 'person_D_006',
                            'car_V_010', 'person_H_004', 'fox_H_001', 'street_S_003', 'hog_H_004',
                            'car_V_011', 'person_S_009', 'person_H_001', 'person_S_005', 'car_V_001',
                            'person_S_014', 'car_D_002', 'person_S_019', 'deer_H_001', 'bat_H_001',
                            'car_V_003', 'person_S_010', 'car_V_007', 'bird_H_003', 'person_S_007',
                            'face_S_001', 'coyote_S_001', 'car_S_003', 'car_D_009', 'person_D_011',
                            'head_S_001', 'cat_H_002', 'bus_S_004', 'bird_H_002', 'truck_S_001',
                            'face_H_001', 'street_S_005', 'person_D_016', 'couple_S_001',
                            'hog_D_001', 'person_D_015', 'helicopter_S_001', 'person_D_019',
                            'person_S_008', 'bus_V_002', 'person_D_014', 'badger_H_001',
                            'person_H_008', 'car_V_008', 'dog_D_001', 'hog_S_001', 'person_V_007',
                            'dog_H_001', 'airplane_H_002', 'street_S_001', 'car_V_014',
                            'bus_V_005', 'person_S_011', 'person_H_014', 'airplane_H_001',
                            'cat_H_001', 'street_S_004', 'bus_V_001', 'person_H_002', 'car_V_013',
                            'motobiker_D_001', 'leopard_H_001', 'person_D_023', 'crowd_S_002',
                            'hog_H_001', 'car_V_004', 'cat_D_001', 'helicopter_H_001',
                            'pickup_S_001', 'person_H_011', 'person_D_004', 'cow_H_001',
                            'boat_H_001', 'person_V_002', 'person_S_013', 'person_H_013']

    def __init__(self, root_dir, subset='train'):
        super(LSOTBTIR, self).__init__()

        self.root_dir = root_dir
        self.subset = subset
        if subset == 'train':
            anno_parent_dir = os.path.join(root_dir, subset, "Annotations")
            seq_parent_dir = os.path.join(root_dir, subset, "TrainingData")
            self.anno_files = []
            self.seq_dirs = []
            self.seq_names = []
            for dir in os.listdir(anno_parent_dir):
                anno_dir = os.path.join(anno_parent_dir, dir)
                seq_dir = os.path.join(seq_parent_dir, dir)

                for d in os.listdir(anno_dir):
                    self.anno_files.append(os.path.join(anno_dir, d))
                    self.seq_dirs.append(os.path.join(seq_dir, d))
                    self.seq_names.append(os.path.join(dir, d))
        elif subset == 'val':
            self.seq_names = self.__lsotb_tir_val_seqs
            self.seq_dirs = [os.path.join(root_dir, subset, s, "img") for s in self.seq_names]
            self.anno_files = [os.path.join(root_dir, subset, s, "anno") for s in self.seq_names]

        else:
            print("Unknown dataset")

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception(f"Sequence {index} not found")
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(os.path.join(self.seq_dirs[index], "*.jpg")))
        img_size, anno_box = self.parse_data(self.anno_files[index])

        return img_files, anno_box, img_size

    def __len__(self):
        return len(self.seq_names)

    @staticmethod
    def parse_data(folder):
        def read_meta_data(filename):
            tree = ET.parse(filename)
            root = tree.getroot()
            bbox = []
            sz = []
            for child in root:
                if child.tag == "size":
                    for data in child:
                        sz.append(data.text)
                if child.tag == "object":
                    for c in child:
                        if c.tag == "bndbox":
                            for coordinate in c:
                                bbox.append((int)(coordinate.text))
            return sz, bbox

        anno_box = []
        for filename in sorted(os.listdir(folder)):
            image_sz, bbox = read_meta_data(os.path.join(folder, filename))
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
            anno_box.append(bbox)
        return image_sz, anno_box


if __name__ == '__main__':
    data = LSOTBTIR(root_dir="/home/martin/Pictures/datasets/LSOTB-TIR/", subset='train')
    img_files, anno, img_sz = data[105]
    for box, file in zip(anno, img_files):
        img = cv2.imread(file)
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        cv2.imshow("img", img)

        if cv2.waitKey(5) == ord('q'):
            break

    print(img_sz)
