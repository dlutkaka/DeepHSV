"""
@file: model.py
@time: 2018/4/17 15:03
@desc: Generate the list of data pairs

"""

import copy
import os
import sys

import numpy as np

image_dir = '/home/deeplearning/work/Deeplearning/dataset/writingID/offline/firmas/'
list_filename_train = '../experiments/data_list/firmas_pairs_c_train.txt'
list_filename_test = '../experiments/data_list/firmas_pairs_c_val.txt'
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


def main(argv=None):
    if argv is None:
        argv = sys.argv

    signers_list = os.listdir(image_dir)
    list_file_train = open(list_filename_train, 'w')
    list_file_test = open(list_filename_test, 'w')

    for signer in signers_list:
        list_file = list_file_train if int(signer) <= 3500 else list_file_test
        genuine_genuine_suf = combine(list(range(1, num_genuine + 1)), 2)
        for item in genuine_genuine_suf:
            genuine0 = signer + '/c-' + signer + "-%02d" % (item[0]) + '.jpg'
            genuine1 = signer + '/c-' + signer + "-%02d" % (item[1]) + '.jpg'
            line = genuine0 + ' ' + genuine1 + ' 1\n'
            list_file.write(line)

        genuine_forged_suf = combine_2list(list(range(1, num_genuine + 1)), list(range(1, num_forged + 1)))
        for item in genuine_forged_suf:
            genuine = signer + '/c-' + signer + "-%02d" % (item[0]) + '.jpg'
            forged = signer + '/cf-' + signer + "-%02d" % (item[1]) + '.jpg'
            line = genuine + ' ' + forged + ' 0\n'
            list_file.write(line)

    """随机伪造情况，每个writer 和其他writer组合"""
    random_forged_nums = 2880000
    # random_forged_val_nums = 2880000 * 0.15

    writers = np.arange(1, 4001, 1)
    writers = np.split(writers, 2)
    writers_part1 = writers[0]
    writers_part2 = writers[1]
    genuine_forged_suf = combine_2list(writers_part1, writers_part2)
    np.random.shuffle(genuine_forged_suf)
    i = 0
    for item in genuine_forged_suf:
        if i > random_forged_nums:
            break
        i += 1
        list_file = list_file_train if i % 6 != 0 else list_file_test
        genuine = '%03d' % item[0] + '/c-' + '%03d' % item[0] + "-09" + '.jpg'
        forged = '%03d' % item[1] + '/c-' + '%03d' % item[1] + "-09" + '.jpg'
        line = genuine + ' ' + forged + ' 2\n'
        list_file.write(line)

    list_file_train.close()
    list_file_test.close()


if __name__ == "__main__":
    sys.exit(main())
