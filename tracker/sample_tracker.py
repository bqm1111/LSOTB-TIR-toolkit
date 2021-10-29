from __future__ import absolute_import
from utils.viz import show_frame
import numpy as np
import time
from PIL import Image


class Tracker(object):
    def __init__(self, name):
        self.name = name

    def init(self, img, box):
        raise NotImplementedError()

    def update(self, img):
        raise NotImplementedError()

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            if not image.mode == 'RGB':
                image = image.convert('RGB')

            start_time = time.time()
            if f == 0:
                self.init(image, box)
            else:
                boxes[f, :] = self.update(image)
            times[f] = time.time() - start_time

            if visualize:
                show_frame(image, boxes[f, :])

        return boxes, times
