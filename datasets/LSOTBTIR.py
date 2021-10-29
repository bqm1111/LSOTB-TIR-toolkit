import io
import os
import glob
from itertools import chain
import numpy as np
import six
from got10k.experiments import otb

class LSOTBTIR(object):
    __lsotb_tir_seqs = ['car_D_005', 'bus_V_003', 'boy_S_002', 'person_H_012', 'head_H_001',
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

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.anno_files = sorted(list(chain.from_iterable(glob.glob(
            os.path.join(root_dir, s, 'groundtruth*.txt')) for s in self.__lsotb_tir_seqs)))
        self.anno_files = self._filter_files(self.anno_files)
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)
        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], 'img/*.jpg')))

        seq_name = self.seq_names[index]
        with open(self.anno_files[index], 'r') as f:
            anno = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))

        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4

        return img_files, anno

    def _filter_files(self, filenames):
        result = []
        for filename in filenames:
            with open(filename, 'r') as f:
                if f.read().strip() == '':
                    print('Warning: %s is empty.' % filename)
                else:
                    result.append(filename)
        return result


if __name__ == '__main__':
    data = LSOTBTIR(root_dir="/home/martin/Pictures/datasets/LSOTB-TIR/val/sequences")
    img_files, anno = data["truck_S_001"]
    print(anno)
