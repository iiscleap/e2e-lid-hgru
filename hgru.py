import tensorflow as tf
import datetime
import logging
import os.path
import numpy as np

from gen_data import dataset_train, dataset_val
from config import *
from utils.nn_utils import gru_layer, attention, dense_layer, bi_gru_layer

logging.basicConfig(filename=log_conf["base_dir"] + log_conf["file_name"],level=logging.DEBUG)
logging.info(str(datetime.datetime.now()))

summary_writer = tf.summary.FileWriter(summary_conf["checkpoint_dir"])


def model(x_in, y_in):
    x = tf.transpose(x_in, perm=[0, 2, 1])
    y_one_hot = tf.one_hot(y_in, train_conf["num_classes"])

    t_batch_size = tf.shape(y_in)[0]

    gru_10msec_out = gru_layer(x, (model_conf["gru_layer1_units"]), name = 'GRU_layer_10msec')
    gru_10msec_out = tf.reshape(gru_10msec_out, shape=[-1, 10, model_conf["gru_layer1_units"]])
    gru_100msec_out = gru_layer(gru_10msec_out, (model_conf["gru_layer2_units"]), name = 'GRU_layer_100msec')
    gru_100msec_out = tf.reshape(gru_100msec_out, shape=[t_batch_size, -1, model_conf["gru_layer2_units"]])
    gru_1sec_out = bi_gru_layer(gru_100msec_out, model_conf["bigru_units"], name = 'GRU_layer_1sec')

    attn_out, attn_weights = attention(gru_1sec_out, model_conf["attention_size"], name = 'attn_layer')
    attn_weights = tf.identity(attn_weights, name="attn_weights")
    attn_out = tf.identity(attn_out, name="embedding")

    dense_out = dense_layer(attn_out, model_conf["dense_dim"], name = "dense_layer")
    logits = dense_layer(dense_out, train_conf["num_classes"], name = "logits")
    predictions = tf.nn.softmax(logits, name='predictions')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot),
                          name='loss')
    return loss, predictions, attn_weights


def main():
    iterator = tf.data.Iterator.from_structure(dataset_train.output_types,
                                               dataset_train.output_shapes)
    next_batch = iterator.get_next()
    training_init_op = iterator.make_initializer(dataset_train)
    validation_init_op = iterator.make_initializer(dataset_val)

    x_in = next_batch[0]
    y_in = next_batch[1]
    global_step = tf.Variable(0, name="global_step", trainable=False)

    hgru_loss, predictions, attn_weights = model(x_in, y_in)
    optimizer = tf.train.AdamOptimizer(learning_rate=model_conf["learning_rate"])
    grads_and_vars = optimizer.compute_gradients(hgru_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    correct_pred = tf.equal(tf.arg_max(predictions, 1), y_in)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        max_val_acc = 0
        summary = tf.Summary()
        for _ in range(train_conf["num_epochs"]):
            sess.run(training_init_op)
            for batch_count in range(train_conf["num_batches"]):
                _, step, train_loss, train_acc = sess.run([train_op, global_step,
                                                       hgru_loss, accuracy])
                time_str = datetime.datetime.now().isoformat()
                summary.value.add(tag='Train accuracy mini batch', simple_value=float(train_acc))
                summary.value.add(tag='Train loss mini batch', simple_value=float(train_loss))
                logging.info("{}: step {}, mini batch loss {:g}, "
                             "mini batch acc {:g}, ".format(time_str, step, train_loss, train_acc))
            sess.run(validation_init_op)
            val_acc = []
            val_loss = []
            while True:
                try:
                    batch_val_acc, batch_val_loss = sess.run([accuracy, hgru_loss])
                    val_acc.append(batch_val_acc)
                    val_loss.append(batch_val_loss)
                except tf.errors.OutOfRangeError:
                    val_acc = np.mean(val_acc)
                    val_loss = np.mean(val_loss)
                    logging.info('Validation loss, accuracy: ' + str(val_loss) + ' ' + str(val_acc))
                    summary.value.add(tag='Validation accuracy', simple_value=val_acc)
                    summary.value.add(tag='Validation loss', simple_value=val_loss)
                    summary_writer.flush()
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        best_model_path = os.path.join(summary_conf["model_directory"],
                                                       summary_conf["model_name"] + str(val_acc))
                        saver.save(sess, best_model_path)
                    break


if __name__ == "__main__":
    main()