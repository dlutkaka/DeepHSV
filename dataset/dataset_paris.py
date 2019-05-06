"""
@file: dataset_paris.py
@time: 2018/7/31 15:03
@desc:Create the input data pipeline using `tf.data`

"""

import numpy as np
import tensorflow as tf

image_width = None
image_height = None
images_dir = None
channels = 1


def _read_image(filename, is_augment):
    image_string = tf.read_file(tf.string_join([images_dir, filename]))
    image_decoded = tf.image.decode_png(image_string, channels=channels)

    true_constant = tf.constant(1, dtype=tf.int32, name="true_constant")
    image_decoded = tf.cond(tf.equal(true_constant, is_augment),
                            lambda: tf.image.flip_left_right(image_decoded),
                            lambda: image_decoded)
    image_resized = tf.image.resize_images(image_decoded, [image_width, image_height])
    return image_resized


def _parse_function(item):
    is_aug = tf.string_to_number(item[3], out_type=tf.int32)
    image0 = _read_image(item[0], is_aug)
    image1 = _read_image(item[1], is_aug)

    image = tf.concat([image0, image1], 2)

    return image, tf.string_to_number(item[2])


def _input_fn(params, is_training, is_augment=False, pos_repeating=1, only_label=None):
    """Train input function.

    Args:
        listfile_path: listfile  has 3 item per line
        params: contains hyperparameters of the model (ex: data_dir, image's width and height.)
    """
    listfile_path = params.signature_train_list if is_training else params.signature_val_list
    data = []
    shuffle_neg = []
    size_per_signer = params.positive_size + params.negative_size
    file = open(listfile_path)
    for i, line in enumerate(file.readlines()):
        items = line.split(' ')
        file0 = items[0]
        file1 = items[1]
        label = int(items[2])
        if (only_label is not None and label != only_label) or label == 2:
            continue

        repeating = 1
        if is_training and pos_repeating > 0 and i % size_per_signer == 0:
            """the number of positive/negative pairs is 276/996, 
            so we need to expand positive pairs, or reduce the negative pairs"""
            shuffle_neg = np.arange(params.positive_size, size_per_signer)
            np.random.shuffle(shuffle_neg)
            shuffle_neg = shuffle_neg[:params.positive_size * pos_repeating]
        if is_training and pos_repeating > 0:
            """expand positive pairs"""
            if label == 2:
                repeating = 1 if (i % params.negative_size) > params.positive_size * pos_repeating else 0
                repeating = 0
            elif label == 0:
                """reduce negative pairs """
                repeating = 1 if i % size_per_signer in shuffle_neg else 0
            elif label == 1:
                repeating = pos_repeating

        for j in range(repeating):
            """file0, file1, label, is_augment"""
            data.append((file0, file1, label, 0))
            if is_augment and is_training:
                data.append((file0, file1, label, 1))
                # data.append((file1, file0, label))
    file.close()
    np.random.shuffle(data)
    print("examples of data:  -> %d" % len(data))

    dataset = tf.data.Dataset.from_tensor_slices(np.array(data))
    dataset = dataset.map(_parse_function, num_parallel_calls=params.num_parallel_calls)
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(
        params.batch_size * params.num_gpus))
    dataset = dataset.prefetch(10)
    return dataset


def input_fn(params, is_training, repeating=1, is_augment=False, only_label=None):
    global image_width, image_height, images_dir, channels
    image_width = params.image_width
    image_height = params.image_height
    images_dir = params.images_dir
    channels = params.channels
    return _input_fn(params, is_training, pos_repeating=repeating, is_augment=is_augment, only_label=only_label)

