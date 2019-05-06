# encoding: utf-8

"""
@author: lichuang
@license: (C) Copyright 2010, CFCA
@file: preprosess_images.py
@time: 2018/5/8 18:
@desc: regularize images, binaries, turn into black background

"""
import os
import sys

import imageio
import numpy as np

dir_to_process = '/home/deeplearning/work/Deeplearning/dataset/writingID/offline/firmas/'
dir_processed = '/home/deeplearning/work/Deeplearning/dataset/writingID/offline/firmas_binarized/'


def _normalize_images(images_dir, processed_dir, reverse):
    """binaries, turn into black background """
    for root, dirs, files in os.walk(images_dir):
        for name in files:
            new_path = os.path.join(processed_dir, os.path.split(root)[-1])
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            if name.lower().endswith('.jpg'):
                image = imageio.imread(os.path.join(root, name))
                image[np.where(image < 230)] = 0
                image[np.where(image >= 230)] = 255
                if reverse:
                    image = 255 - image
                imageio.imwrite(os.path.join(new_path, name), image)
    print('all images processed!')


def main(argv=None):
    if argv is None:
        argv = sys.argv
    _normalize_images(dir_to_process, dir_processed, False)


if __name__ == "__main__":
    sys.exit(main())
