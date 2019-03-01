import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split
import datetime
import logging
import sys
from gen_data import next_batch
from gen_data import training_init_op
from gen_data import validation_init_op
from gen_data import num_dev_batches

num_classes = 6
attention_size = 124
hidden_units_1 = 256
hidden_units_2 = 256
hidden_units_3 = 256
hidden_units_4 = 256
hidden_units_5 = 256
num_batches = 100000

logging.basicConfig(filename='/home/bharatp/lid_phase2/RATS/logs/hgru_big_2dense_6class_with_mul.log',level=logging.DEBUG)
logging.info(str(datetime.datetime.now()))

def attention(H, W_attn, b_attn, c_attn, hidden_dimension):  # H shape (batch_size, seq_length, hidden_dimension)
    u = tf.tanh(tf.tensordot(H, W_attn, axes=1) + b_attn)  # shape (batch_size, seq_length, attention_size)
    uc = tf.tensordot(u, c_attn, axes=1)  # shape (batch_size, seq_length)
    alphas = tf.nn.softmax(uc)  # shape same as uc
    output = tf.reduce_sum(H * tf.expand_dims(alphas, -1), 1)
    return output, alphas


summary_dir = '/home/bharatp/lid_phase2/RATS/models/board'  # Path(saver_conf["tensorboard"]) / "train_{}".format(self.task_index)
# summary_dir.mkdir(exist_ok=True)
summary_writer = tf.summary.FileWriter(str(summary_dir))

x_in = next_batch[0]
y = next_batch[1]
y_one_hot = tf.one_hot(y, num_classes)
global_step = tf.Variable(0, name="global_step", trainable=False)
x = tf.transpose(x_in, perm=[0, 2, 1])
keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keepprob')
t_batch_size = tf.shape(y)[0]
learning_rate = tf.placeholder(tf.float32, name='lr')

with tf.variable_scope('first_layer'):
    cell1 = tf.contrib.rnn.GRUCell(hidden_units_1)
    lstm_outputs1, state1 = tf.nn.dynamic_rnn(cell1, x, dtype=tf.float32)
    lstm_outputs1 = lstm_outputs1[:, -1, :]
    out1 = tf.reshape(lstm_outputs1, shape=[-1, 10, hidden_units_1])

# with tf.variable_scope('second_layer'):
#    cell2 = tf.contrib.rnn.GRUCell(hidden_units_2)
#    lstm_outputs2, state2 = tf.nn.dynamic_rnn(cell2, lstm_outputs1, dtype = tf.float32)
#    lstm_outputs2 = lstm_outputs2[:,-1,:]
#    out2 = tf.reshape(lstm_outputs2, shape=[-1,10,hidden_units_2])

with tf.variable_scope('third_layer'):
    cell3 = tf.contrib.rnn.GRUCell(hidden_units_3)
    lstm_outputs3, state3 = tf.nn.dynamic_rnn(cell3, out1, dtype=tf.float32)
    lstm_outputs3 = lstm_outputs3[:, -1, :]
    out3 = tf.reshape(lstm_outputs3, shape=[t_batch_size, -1, hidden_units_3], name='embedding_0')

with tf.variable_scope('fourth_layer'):
    cell4_fw = tf.contrib.rnn.GRUCell(hidden_units_4)
    cell4_bw = tf.contrib.rnn.GRUCell(hidden_units_4)
    lstm_outputs4, states4 = tf.nn.bidirectional_dynamic_rnn(cell4_fw, cell4_bw, out3, dtype=tf.float32)
    lstm_outputs4 = tf.concat(lstm_outputs4, 2)
    w_attn = tf.Variable(tf.truncated_normal([2 * hidden_units_4, attention_size], stddev=0.1))
    b_attn = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    c_attn = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    lstm_out, alphas = attention(lstm_outputs4, w_attn, b_attn, c_attn, 2 * hidden_units_4)
    alphas = tf.identity(alphas, name="attn_weights_1")
    lstm_out = tf.identity(lstm_out, name="embedding_main")

with tf.variable_scope('dense_layer'):
    dense_dim = 256
    training = tf.placeholder(tf.bool)
    w_dense = tf.Variable(tf.truncated_normal([2 * hidden_units_4, dense_dim], stddev=0.1))
    b_dense = tf.Variable(tf.random_normal([dense_dim]))
    h_dense = tf.add(tf.matmul(lstm_out, w_dense), b_dense)
    h_dense = tf.nn.relu(h_dense, name='embedding_dense')

    w_dense_1 = tf.Variable(tf.truncated_normal([dense_dim, dense_dim], stddev=0.1))
    b_dense_1 = tf.Variable(tf.random_normal([dense_dim]))
    h_dense_1 = tf.add(tf.matmul(h_dense, w_dense_1), b_dense_1)
    h_dense_1 = tf.nn.relu(h_dense_1, name='embedding_dense_1')

with tf.variable_scope('out_layer'):
    lambda_l2_reg = 0.0
    w_out = tf.Variable(tf.truncated_normal([dense_dim, 6], stddev=0.1))
    b_out = tf.Variable(tf.random_normal([6]))
    logits = tf.add(tf.matmul(h_dense_1, w_out), b_out, name='logit')

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
predictions = tf.nn.softmax(logits, name='predictions')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot), name='loss')
# l2_loss = lambda_l2_reg * sum(tf.nn.l2_loss(tf_var)
#             for tf_var in tf.trainable_variables()
#             if not ("Bias" in tf_var.name))
# loss = tf.add(loss, l2_loss, name='loss_with_l2')

optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

correct_pred = tf.equal(tf.arg_max(predictions, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print("Training started.")
logging.info('Training started.')
logging.info(str(datetime.datetime.now()))
lr = 1e-5
no_epochs = 300
with tf.Session(config=config) as sess:
    sess.run(init)
    # saver.restore(sess, trained_model_path.rsplit('.',1)[0])
    max_val_acc = 0
    summary = tf.Summary()
    sess.run(training_init_op)
    for i in range(num_batches):
        _, __, step, train_loss, train_acc = sess.run([train_op, extra_update_ops, global_step,
                                                       loss, accuracy])
        current_step = tf.train.global_step(sess, global_step)
        time_str = datetime.datetime.now().isoformat()
        summary.value.add(tag='train accuray mini batch', simple_value=float(train_acc))
        summary.value.add(tag='train loss mini batch', simple_value=float(train_loss))
        if (i+1) % 50 == 0:
            logging.info(
                "{}: step {}, mini batch loss {:g}, mini batch acc {:g}".format(time_str, step, train_loss, train_acc))
            print(
                "{}: step {}, mini batch loss {:g}, mini batch acc {:g}".format(time_str, step, train_loss, train_acc))
        if (i+1) % 250 == 0:
            sess.run(validation_init_op)
            val_acc = 0
            val_loss = 0.0
            for d_i in range(num_dev_batches):
                batch_val_acc, batch_val_loss, _ = sess.run([accuracy, loss, extra_update_ops])
                val_acc += batch_val_acc
                val_loss += batch_val_loss
            val_loss = val_loss / (d_i + 1)
            val_acc = val_acc / (d_i + 1)
            print("Validation loss, accuracy: " + str(val_loss) + ' ' + str(val_acc))
            logging.info('Validation loss, accuracy:' + str(val_loss) + ' ' + str(val_acc))
            summary.value.add(tag='Validation accuray', simple_value=float(val_acc))
            summary.value.add(tag='Validation loss', simple_value=float(val_loss))
            summary_writer.add_summary(summary, i)
            summary_writer.flush()
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                best_model_path = '/home/bharatp/lid_phase2/RATS/models/checkpoints/hgru_big_2dense_ntwk_with_mul_' + str(
                    val_acc)
                saver.save(sess, best_model_path)
            sess.run(training_init_op)




