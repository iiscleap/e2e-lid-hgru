import numpy as np
import tensorflow as tf

from config import *
from utils.data_utils import *

train_data_list = np.genfromtxt(path_conf["train_list"], dtype='str')

dev_data = np.load(path_conf["dev_data"])
dev_labels = np.load(path_conf["dev_labels"])

dev_data, dev_labels = split_data_batches(dev_data, dev_labels, train_conf["val_batch_size"])


def gen_batch_val():
    for d_i, d_batch in enumerate(dev_data):
        dev_labels_batch = dev_labels[d_i]
        dev_data_batch = split_data(d_batch, train_conf["window"], train_conf["hop"])
        yield dev_data_batch, dev_labels_batch


def gen_batch_train():
    while True:
        data_sample = train_data_list[np.random.choice(train_data_list.shape[0], train_conf["train_batch_size"], replace=False)]
        data, labels = prep_batch(data_sample, label_dict, train_conf["sample_len"])
        data = split_data(data, train_conf["window"], train_conf["hop"])
        yield data, labels


dataset_train = tf.data.Dataset.from_generator(gen_batch_train, output_types=(tf.float64, tf.int32),
                                               output_shapes=(tf.TensorShape([None, FEA_DIMENSION, train_conf["window"]]), tf.TensorShape([None])))
dataset_train = dataset_train.prefetch(1)

dataset_val = tf.data.Dataset.from_generator(gen_batch_val, output_types=(tf.float64, tf.int32),
                                             output_shapes=(tf.TensorShape([None, FEA_DIMENSION, train_conf["window"]]), tf.TensorShape([None])))
dataset_val = dataset_val.prefetch(1)
