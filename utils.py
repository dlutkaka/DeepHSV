"""
@file: model.py
@time: 2018/4/17 15:03
@desc:General utility functions

"""

import json
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import six
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import device_setter


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def visualize(distance_positive, distance_negative):
    kwargs = dict(histtype='stepfilled', alpha=0.5, normed=True, bins=40)
    plt.hist(distance_positive, **kwargs)
    plt.hist(distance_negative, **kwargs)

    plt.title('visualize distance')
    plt.show()


def compute_eer(distance_positive, distance_negative):
    all_true = len(distance_negative)
    all_false = len(distance_positive)
    distance_positive = np.column_stack((np.array(distance_positive), np.zeros(len(distance_positive))))
    distance_negative = np.column_stack((np.array(distance_negative), np.ones(len(distance_negative))))
    distance = np.vstack((distance_positive, distance_negative))
    distance = distance[distance[:, 0].argsort(), :]  # sort by first column
    # np.savetxt('distribution_siamese.txt', distance)
    distance = np.matrix(distance)

    min_dis = sys.maxsize
    min_th = sys.maxsize
    eer = sys.maxsize
    fa = all_false
    miss = 0

    for i in range(0, all_true + all_false):
        if distance[i, 1] == 1:
            miss += 1
        else:
            fa -= 1

        fa_rate = float(fa) / all_false
        miss_rate = float(miss) / all_true

        if abs(fa_rate - miss_rate) < min_dis:
            min_dis = abs(fa_rate - miss_rate)
            eer = max(fa_rate, miss_rate)
            min_th = distance[i, 0]

    print('eer:', eer, ' threshold:', min_th)
    return [eer, min_th]


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops is None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser


def reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.

    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are is large enough.

    Args:
      input_tensor: input tensor of size [batch_size, height, width, channels].
      kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

    Returns:
      a tensor with the kernel size.

    TODO(jrru): Make this function work with unknown shapes. Theoretically, this
    can be done with the code below. Problems are two-fold: (1) If the shape was
    known, it will be lost. (2) inception.tf.contrib.slim.ops._two_element_tuple
    cannot
    handle tensors that define the kernel size.
        shape = tf.shape(input_tensor)
        return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                          tf.minimum(shape[2], kernel_size[1])])

    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [
            min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
        ]
    return kernel_size_out
