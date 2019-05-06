"""
@file: model.py
@time: 2018/4/17 15:03
@desc:Train and evaluate the model

"""

import argparse
import os

import tensorflow as tf

import utils
from dataset.dataset_paris import input_fn
from models import model_fn_signature as model_fn
from utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test')
parser.add_argument('--mode', default='evaluate')

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = 'dataset/params.json'
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    config = tf.estimator.RunConfig(tf_random_seed=229,
                                    model_dir=args.model_dir,
                                    save_checkpoints_steps=params.save_checkpoints_steps,
                                    save_summary_steps=params.save_summary_steps,
                                    keep_checkpoint_max=params.keep_checkpoint_max)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    if args.mode.lower() == 'train':
        """ model:{"Siamese", "SiameseInception", "2ChannelsCNN", "2ChannelsSoftmax" """
        tf.logging.info("Starting training model : {} ".format(params.model))
        estimator.train(lambda: input_fn(params, is_training=True, repeating=1, is_augment=True))
        # estimator.train(lambda: input_fn(params, is_training=True, repeating=1, is_augment=False))
        res = estimator.evaluate(lambda: input_fn(params, is_training=False, is_augment=False))

    elif args.mode.lower() == 'predict':
        res = estimator.predict(lambda: input_fn(params, is_training=False, is_augment=False, only_label=0))
        distance_negative = [x['distance'] for x in res]
        res = estimator.predict(lambda: input_fn(params, is_training=False, is_augment=False, only_label=1))
        distance_positive = [x['distance'] for x in res]
        utils.compute_eer(distance_positive=distance_positive, distance_negative=distance_negative)
        utils.visualize(distance_positive=distance_positive, distance_negative=distance_negative)

    else:
        tf.logging.info("Evaluation on test set.")
        res = estimator.evaluate(lambda: input_fn(params, is_training=False, is_augment=False))

        """evaluate from first checkpoint to last"""
        # checkpoint_file = open(args.model_dir + '/checkpoint', 'r')
        # checkpoint_lines = list(checkpoint_file.readlines())
        # checkpoint_file.close()
        # for i in range(1, len(checkpoint_lines)):
        #     checkpoint = checkpoint_lines[i].split('\"')[-2]
        #
        #     res = estimator.evaluate(
        #         lambda: input_fn(params, False, False),
        #         steps=100,
        #         checkpoint_path=args.model_dir + '/' + checkpoint)
        #     for key in res:
        #         print("{}: {}".format(key, res[key]))
