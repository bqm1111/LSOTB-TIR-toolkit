import io
import json
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
    __lsotb_tir_val_seqs = ['airplane_H_001', 'airplane_H_002', 'badger_H_001', 'bat_H_001', 'bird_H_001', 'bird_H_002',
                            'bird_H_003', 'boat_D_001', 'boat_H_001', 'boy_S_001', 'boy_S_002', 'bus_S_004',
                            'bus_V_001',
                            'bus_V_002', 'bus_V_003', 'bus_V_004', 'bus_V_005', 'car_D_002', 'car_D_004', 'car_D_005',
                            'car_D_007', 'car_D_009', 'car_S_001', 'car_S_002', 'car_S_003', 'car_V_001', 'car_V_003',
                            'car_V_004', 'car_V_006', 'car_V_007', 'car_V_008', 'car_V_009', 'car_V_010', 'car_V_011',
                            'car_V_013', 'car_V_014', 'cat_D_001', 'cat_H_001', 'cat_H_002', 'couple_S_001',
                            'cow_H_001', 'coyote_S_001', 'crowd_S_001', 'crowd_S_002', 'deer_H_001', 'dog_D_001',
                            'dog_D_002', 'dog_H_001', 'drone_D_001', 'face_H_001', 'face_S_001', 'fox_H_001',
                            'head_H_001', 'head_S_001', 'helicopter_H_001', 'helicopter_H_002', 'helicopter_S_001',
                            'hog_D_001', 'hog_H_001', 'hog_H_002', 'hog_H_003', 'hog_H_004', 'hog_S_001',
                            'leopard_H_001', 'motobiker_D_001', 'motobiker_V_001', 'person_D_003', 'person_D_004',
                            'person_D_006', 'person_D_009', 'person_D_011', 'person_D_014', 'person_D_015',
                            'person_D_016', 'person_D_019', 'person_D_020', 'person_D_021', 'person_D_022',
                            'person_D_023', 'person_H_001', 'person_H_002', 'person_H_003', 'person_H_004',
                            'person_H_006', 'person_H_008', 'person_H_009', 'person_H_011', 'person_H_012',
                            'person_H_013', 'person_H_014', 'person_S_001', 'person_S_002', 'person_S_003',
                            'person_S_004', 'person_S_005', 'person_S_006', 'person_S_007', 'person_S_008',
                            'person_S_009', 'person_S_010', 'person_S_011', 'person_S_012', 'person_S_013',
                            'person_S_014', 'person_S_015', 'person_S_016', 'person_S_017', 'person_S_018',
                            'person_S_019', 'person_V_002', 'person_V_007', 'person_V_008', 'pickup_S_001',
                            'street_S_001', 'street_S_002', 'street_S_003', 'street_S_004', 'street_S_005',
                            'truck_S_001', 'woman_H_001']

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
        for folder in self.anno_files:
            self._gen_info(folder)
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
    def _gen_info(folder):
        if not os.path.exists(os.path.join(folder, "info.json")):
            return

        anno_file = sorted(list(glob.glob(os.path.join(folder, "*.xml"))))
        tree = ET.parse(anno_file[0])
        root = tree.getroot()
        info = defaultdict(int)

        info["num_obj"] = len(root.findall('object'))
        print(f"writing num object to file {os.path.join(folder, 'info.json')}")

        # with open(os.path.join(folder, "info.json"), "w") as f:
        #     f.write(json.dumps(info))

    def add_folder(self, file):
        filename = os.path.join("~/", file)
        img = cv2.imread(filename)
        cv2.imshow("window", img)
        cv2.waitKey()

    @staticmethod
    def parse_data(folder):
        def read_meta_data(filename):
            tree = ET.parse(filename)
            root = tree.getroot()
            bbox = []
            sz = []
            if filename.split("/")[-3] == "woman_H_001":
                for child in root:
                    if child.tag == "size":
                        for data in child:
                            sz.append(data.text)
                    if child.tag == "object":
                        for c in child:
                            if c.tag == "name" and c.text == "dog":
                                break
                            if c.tag == "bndbox":
                                for coordinate in c:
                                    bbox.append((int)(coordinate.text))

            else:
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
        for filename in sorted(glob.glob(os.path.join(folder, "*.xml"))):
            image_sz, bbox = read_meta_data(os.path.join(folder, filename))
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
            anno_box.append(bbox)
        anno_box = np.asarray(anno_box)
        return image_sz, anno_box

def write_meta_data(filename):
    info_tree = ET.parse(filename)
    root = info_tree.getroot()
    for obj in root.findall('object'):


def read_meta_data(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    print(root.find("size").find("width").text)
    num_obj = 0
    info = defaultdict(int)
    print(f"num object = {len(root.findall('object'))}")
    for obj in root.findall('object'):
        num_obj = num_obj + 1
        print("Name:", obj.find('name').text)
    info["num_obj"] = num_obj
    with open("info.json", "w") as f:
        f.write(json.dumps(info))


if __name__ == '__main__':
    # data = LSOTBTIR(root_dir="/home/martin/Pictures/datasets/LSOTB-TIR/", subset='val')
    # img_files, anno, img_sz = data["woman_H_001"]
    #
    # for box, file in zip(anno, img_files):
    #     img = cv2.imread(file)
    #     pt1 = (int(box[0]), int(box[1]))
    #     pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
    #     cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    #     cv2.imshow("img", img)
    #
    #     if cv2.waitKey(5) == ord('q'):
    #         break

    img_file = "/home/martin/Pictures/datasets/LSOTB-TIR/val/woman_H_001/anno/00000001.xml"
    read_meta_data(img_file)
    folder = "/home/martin/Pictures/datasets/LSOTB-TIR/val"
