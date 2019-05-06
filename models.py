# encoding: utf-8

"""
@file: models.py
@time: 2018/4/17 15:03
@desc: 4 models: Siamese, SiameseInception, 2ChannelsCNN, 2ChannelsSoftmax

"""

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib

import net.inception_v3 as inception_v3
import utils


def _embedding_alexnet(is_training, images, params):
    with tf.variable_scope('Siamese', 'CFCASiamese', [images], reuse=tf.AUTO_REUSE):
        with arg_scope(
                [layers.conv2d], activation_fn=tf.nn.relu):
            net = layers.conv2d(
                images, 96, [11, 11], 4, padding='VALID', scope='conv1')
            # net = layers.batch_norm(net, decay=0.9, epsilon=1e-06, is_training=is_training)
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = layers.conv2d(net, 256, [5, 5], scope='conv2')
            # net = layers.batch_norm(net, decay=0.9, epsilon=1e-06, is_training=is_training)
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = layers_lib.dropout(
                net, keep_prob=0.7, is_training=is_training)
            net = layers.conv2d(net, 384, [3, 3], scope='conv3')
            net = layers.conv2d(net, 256, [3, 3], scope='conv4')
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool5')
            net = layers_lib.dropout(
                net, keep_prob=0.7, is_training=is_training)
            net = layers_lib.flatten(net, scope='flatten1')
            net = layers_lib.fully_connected(net, 1024, scope='fc1',
                                             weights_regularizer=layers.l2_regularizer(0.0005))
            net = layers_lib.dropout(
                net, keep_prob=0.5, is_training=is_training)
            net = layers_lib.fully_connected(net, params.embedding_size, scope='fc2',
                                             weights_regularizer=layers.l2_regularizer(0.0005))
            return net


def _embedding_inception(is_training, images, params):
    logits, endpoints = inception_v3.inception_v3(
        images, num_classes=params.embedding_size, is_training=is_training,
        dropout_keep_prob=params.keep_prob, reuse=tf.AUTO_REUSE, scope='InceptionV3')
    return logits


def _embedding_2logits(is_training, embeddings, labels):
    """embeddings to 2 logits and losss"""
    logits = layers_lib.fully_connected(
        embeddings, 2, scope='fc3', reuse=tf.AUTO_REUSE)
    logits_array = tf.split(logits, 2, 1)
    logits_diff = tf.subtract(logits_array[0], logits_array[1])

    if labels is not None:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.int64)))
        return loss, logits_diff
    else:
        return None, logits_diff


def _calculate_eucd2(embedding1, embedding2):
    eucd2 = tf.pow(tf.subtract(embedding1, embedding2), 2)
    eucd2 = tf.reduce_sum(eucd2, 1)
    eucd = tf.sqrt(eucd2 + 1e-6, name="eucd")
    return tf.reshape(eucd2, [-1, 1]), tf.reshape(eucd, [-1, 1])


def _loss_siamese(images, labels, params, is_training, embedding_func):
    """<SigNet: Convolutional Siamese Network for Writer
        Independent Offline Signature Verification>"""
    images = tf.split(images, 2, axis=3)
    images0 = tf.reshape(
        images[0], [-1, params.image_width, params.image_height, 1])
    images1 = tf.reshape(
        images[1], [-1, params.image_width, params.image_height, 1])

    """When using Siamese, The Complex network such as Inception will 
        cause overfitting even in first epoch"""
    embeddings0 = embedding_func(is_training, images0, params)
    embeddings1 = embedding_func(is_training, images1, params)

    eucd2, eucd = _calculate_eucd2(embeddings0, embeddings1)
    if labels is not None:
        labels_t = tf.reshape(labels, [-1, 1])
        labels_f = tf.reshape(tf.subtract(
            1.0, labels, name="1-yi"), [-1, 1])  # labels_ = !labels;

        c = tf.constant(int(params.margin), dtype=tf.float32, name="C")
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(
            tf.subtract(c, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")

        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss, eucd
    else:
        return None, eucd


def _loss_siamese_alexnet(images, labels, params, is_training):
    return _loss_siamese(images, labels, params, is_training, _embedding_alexnet)


def _loss_siamese_inception(images, labels, params, is_training):
    return _loss_siamese(images, labels, params, is_training, _embedding_inception)


def _loss_inception_2logits(images, labels, params, is_training):
    images = tf.split(images, 2, axis=3)
    images0 = tf.reshape(
        images[0], [-1, params.image_width, params.image_height, 1])
    images1 = tf.reshape(
        images[1], [-1, params.image_width, params.image_height, 1])
    embeddings0 = _embedding_inception(is_training, images0, params)
    embeddings1 = _embedding_inception(is_training, images1, params)
    embeddings = tf.concat([embeddings0, embeddings1], axis=1)
    return _embedding_2logits(is_training, embeddings, labels)


def _loss_2channels_softmax_alex(images, labels, params, is_training):
    # params.embedding_size = 2
    embeddings = _embedding_alexnet(is_training, images, params)
    logits = layers_lib.fully_connected(
        embeddings, 2, scope='fc3', reuse=tf.AUTO_REUSE)
    # logits = embeddings
    logits_array = tf.split(logits, 2, 1)
    logits_diff = tf.subtract(logits_array[0], logits_array[1])

    if labels is not None:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.int64)))
        return loss, logits_diff
    else:
        return None, logits_diff


def _loss_2channels_softmax(images, labels, params, is_training):
    logits, endpoints = inception_v3.inception_v3(
        images, num_classes=2, is_training=is_training,
        dropout_keep_prob=params.keep_prob, reuse=tf.AUTO_REUSE, scope='InceptionV3')
    logits_array = tf.split(logits, 2, 1)
    logits_diff = tf.subtract(logits_array[0], logits_array[1])

    if labels is not None:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.int64)))
        return loss, logits_diff
    else:
        return None, logits_diff


def _loss_2channels(images, labels, params, is_training):
    """<Learning to Compare Image Patches via Convolutional Neural Networks>"""

    logits, endpoints = inception_v3.inception_v3(
        images, num_classes=1, is_training=is_training,
        dropout_keep_prob=params.keep_prob, reuse=tf.AUTO_REUSE, scope='InceptionV3')

    if labels is not None:
        """ convert y from {0,1} to {-1,1}"""
        labels = tf.multiply(labels, 2.0)
        labels = tf.subtract(labels, 1.0)
        labels = tf.reshape(labels, [-1, 1])
        loss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(labels, logits)))
        return tf.reduce_mean(loss), tf.subtract(1.0, logits)
    else:
        return None, tf.subtract(1.0, logits)


def _normlize_distance(distance):
    """normalization of distance"""
    max_val = tf.reduce_max(distance)
    min_val = tf.reduce_min(distance)
    distance_norm = tf.div(tf.subtract(distance, min_val),
                           tf.subtract(max_val, min_val))
    return distance_norm


models = {"Siamese": _loss_siamese_alexnet,
          "SiameseInception": _loss_siamese_inception,
          "Inception_2logits": _loss_inception_2logits,
          "2ChannelsAlexnet": _loss_2channels_softmax_alex,
          "2ChannelsCNN": _loss_2channels,
          "2ChannelsSoftmax": _loss_2channels_softmax}


def model_fn_signature(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: input batch of images
        labels:True or not
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL }
        params: contains hyper parameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    loss_function = models[params.model]

    losses_all_tower = []
    distance_all_tower = []
    images_all_tower = tf.split(features, params.num_gpus, axis=0)
    labels_all_tower = None
    if labels is not None:
        labels = tf.reshape(labels, [-1])
        labels_all_tower = tf.split(labels, params.num_gpus, axis=0)

    for i in range(params.num_gpus):
        worker_device = '/{}:{}'.format('gpu', i)
        images_tower = images_all_tower[i]

        device_setter = utils.local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                params.num_gpus, tf.contrib.training.byte_size_load_fn))
        with tf.device(device_setter):
            if labels_all_tower is not None:
                loss, distance = loss_function(
                    images_tower, labels_all_tower[i], params, is_training)
                losses_all_tower.append(loss)
            else:
                _, distance = loss_function(
                    images_tower, None, params, is_training)
            distance_all_tower.append(distance)

    consolidation_device = '/cpu:0'
    with tf.device(consolidation_device):
        distance = tf.concat(distance_all_tower, 0)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'distance': distance}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.reduce_mean(losses_all_tower, name='loss_mean')
        labels = tf.reshape(labels, [-1, 1])
        labels_reversal = tf.reshape(tf.subtract(
            1.0, labels), [-1, 1])  # labels_ = !labels;
        positive_distance = tf.reduce_mean(tf.multiply(labels, distance))
        negative_distance = tf.reduce_mean(
            tf.multiply(labels_reversal, distance))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('positive_distance', positive_distance)
        tf.summary.scalar('negative_distance', negative_distance)

        distance_norm = _normlize_distance(distance)
        metric_ops = tf.metrics.auc(labels_reversal, distance_norm)
        tf.summary.scalar('auc', metric_ops[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            sec_at_spe_metric = tf.metrics.sensitivity_at_specificity(
                labels_reversal, distance_norm, 0.90)
            eval_metric_ops = {'evaluation_auc': metric_ops,
                               'sec_at_spe': sec_at_spe_metric}

            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

        else:

            logging_hook = tf.train.LoggingTensorHook({"positive_distance": positive_distance,
                                                       "negative_distance": negative_distance,
                                                       "auc": metric_ops[1]}, every_n_iter=100)

            # optimizer = tf.train.RMSPropOptimizer(params.learning_rate)
            optimizer = tf.train.AdamOptimizer(params.learning_rate)
            global_step = tf.train.get_global_step()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss, global_step=global_step, colocate_gradients_with_ops=True)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
