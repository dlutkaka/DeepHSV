"""
@file: dataset_bhsig260.py
@time: 2018/6/20 15:03
@desc:Create the paris list of BHSig260 Database

"""

import copy
import os
import sys

import imageio
import numpy as np

num_genuine = 24
num_forged = 30


# 生成数组l的全部组合（长度k）
def combine(l, k):
    answers = []
    one = [0] * k

    def next_c(li=0, ni=0):
        if ni == k:
            answers.append(copy.copy(one))
            return
        for lj in range(li, len(l)):
            one[ni] = l[lj]
            next_c(lj + 1, ni + 1)

    next_c()
    return answers


# 生成两个数组间的全部组合
def combine_2list(list1, list2):
    answers = []
    for i1 in list1:
        for i2 in list2:
            answers.append([i1, i2])
    return answers


def generate_list(data_dir, train_size, filename_pre, listfile_name):
    root_dir = os.path.basename(data_dir)
    signers_list = os.listdir(data_dir)
    list_file_train = open(listfile_name + '_train.txt', 'w')
    list_file_test = open(listfile_name + '_val.txt', 'w')

    train_indexs = np.arange(0, len(signers_list), 1)
    np.random.shuffle(train_indexs)
    train_indexs = train_indexs[:train_size]

    for i, signer in enumerate(signers_list):
        list_file = list_file_train if i in train_indexs else list_file_test
        genuine_genuine_suf = combine(list(range(1, num_genuine + 1)), 2)
        for item in genuine_genuine_suf:
            genuine0 = "%s/%s/%s-%d-G-%02d%s" % (root_dir, signer, filename_pre, int(signer), item[0], '.jpg')
            genuine1 = "%s/%s/%s-%d-G-%02d%s" % (root_dir, signer, filename_pre, int(signer), item[1], '.jpg')
            line = genuine0 + ' ' + genuine1 + ' 1\n'
            list_file.write(line)

        genuine_forged_suf = combine_2list(list(range(1, num_genuine + 1)), list(range(1, num_forged + 1)))
        for item in genuine_forged_suf:
            genuine = "%s/%s/%s-%d-G-%02d%s" % (root_dir, signer, filename_pre, int(signer), item[0], '.jpg')
            forged = "%s/%s/%s-%d-F-%02d%s" % (root_dir, signer, filename_pre, int(signer), item[1], '.jpg')
            line = genuine + ' ' + forged + ' 0\n'
            list_file.write(line)

    list_file_train.close()
    list_file_test.close()


def rename(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if not file.endswith('.jpg'):
                continue
            new_filename = file.replace('-S-00', '-S-')
            new_filename = new_filename.replace('-S-0', '-S-')
            os.rename(os.path.join(root, file), os.path.join(root, new_filename))


def tif_to_jpg(tif_dir, jpg_dir):
    for root, dirs, files in os.walk(tif_dir):
        to_dir = root.replace(tif_dir, jpg_dir)
        if not os.path.exists(to_dir):
            os.mkdir(to_dir)
        for file in files:
            if not file.endswith('.tif'):
                continue
            image = imageio.imread(os.path.join(root, file))
            jpg_file = file.replace('.tif', '.jpg')
            imageio.imwrite(os.path.join(to_dir, jpg_file), image)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    rename('/home/deeplearning/work/Deeplearning/dataset/writingID/offline/BHSig260_jpgs/')
    # generate_list('/home/deeplearning/work/Deeplearning/dataset/writingID/offline/BHSig260_jpgs/Hindi', 100, 'H-S',
    #               '../experiments/data_list/bhsig260_Hindi')
    # generate_list('/home/deeplearning/work/Deeplearning/dataset/writingID/offline/BHSig260_jpgs/Bengali', 50,
    #               'B-S',
    #               '../experiments/data_list/bhsig260_Bengali')


if __name__ == "__main__":
    sys.exit(main())
