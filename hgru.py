import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
import datetime
import logging

logging.basicConfig(filename='/home/bharatp/lid_phase2/RATS/logs/hgru.log',level=logging.DEBUG)
logging.info(str(datetime.datetime.now()))

train_data_1 = np.load('/home/data1/LRE2017/bharat/RATS/bnf_train_data/rats_bnf_train_data_30s_one_sample_per_file_25000.npy')
train_labels_1 = np.load('/home/data1/LRE2017/bharat/RATS/bnf_train_data/rats_bnf_train_labels_30s_one_sample_per_file_25000.npy')

train_data_2 = np.load('/home/data1/LRE2017/bharat/RATS/bnf_train_data/rats_bnf_train_data_30s_one_sample_per_file_50000.npy')
train_labels_2 = np.load('/home/data1/LRE2017/bharat/RATS/bnf_train_data/rats_bnf_train_labels_30s_one_sample_per_file_50000.npy')

#train_data = np.load('/home/data1/LRE2017/bharat/RATS/bnf_train_data/rats_bnf_train_data_30s_one_sample_per_file_last.npy')
#train_labels = np.load('/home/data1/LRE2017/bharat/RATS/bnf_train_data/rats_bnf_train_labels_30s_one_sample_per_file_last.npy')
dev_data = np.load('/home/data1/LRE2017/bharat/RATS/bnf_train_data/rats_bnf_dev_1_data_30s.npy')
dev_labels = np.load('/home/data1/LRE2017/bharat/RATS/bnf_train_data/rats_bnf_dev_1_labels_30s.npy')

logging.info('Data loaded.')
logging.info(str(datetime.datetime.now()))
mini_batch_size = 32
window = 20
hop = 10
hidden_units_1 = 128
hidden_units_2 = 128
hidden_units_3 = 128
hidden_units_4 = 128
hidden_units_5 = 128

def hot_encode(a, n_classes):
    b = np.zeros((a.size, n_classes))
    b[np.arange(a.size),a] = 1
    return b

def split_data(data, window, hop): #Assume len(data)-10 is divisible by 300
    seq_len = data.shape[2]
    l = int((seq_len-window)/hop)+1 #Ex: (310-20)/10 + 1 =30
    j = 0
    data_splits = []
    for i in range(l):
       x = data[:,:,j:j+window]
       data_splits.append(x)
       j += hop
    a = []
    for i in range(len(data)):
       d = [y[i] for y in data_splits]
       a.append(np.array(d))
    return np.vstack(a)

def adjust_data(data, size):
    if size==10:
        d = data[:,:,0:910]
    else:
        l = random.randint(0,data.shape[2]-10)
        chunk = data[:,:,l:l+10]
        d = np.append(data, chunk, axis=2)
    return d

def adjust_data_(data, size):
        l = random.randint(0,data.shape[2]-size)
        d = data[:,:,l:l+size]
        return d

def batch_iter(data, labels, batch_size, num_epochs, size):
    data_size = len(data)
    for e in range(num_epochs):
        for b in range(data_size/batch_size):
            l = random.sample(range(data_size), batch_size)
            batch = data[l]
            batch = adjust_data(batch, size)
            batch = split_data(batch, window, hop)
            batch_labels = labels[l]
            yield batch, batch_labels

def batch_iter_(data, labels, batch_size, num_epochs, size):
    data_size = len(data)
    for e in range(num_epochs):
        for b in range(data_size/batch_size):
            l = random.sample(range(data_size), batch_size)
            batch = data[l]
            batch = adjust_data_(batch, size)
            batch = split_data(batch, window, hop)
            batch_labels = labels[l]
            yield batch, batch_labels

train_labels_1 = hot_encode(train_labels_1, 5)
train_labels_2 = hot_encode(train_labels_2, 5)
dev_labels = hot_encode(dev_labels, 5)

def split_data_batches(data, labels, size, mini_batch_size_):
   data = adjust_data(data, size)
   d_len = len(data)
   batches = d_len/mini_batch_size_
   data = data[0:(batches*mini_batch_size_)]
   labels = labels[0:(batches*mini_batch_size_)]
   data = np.array(np.split(data, batches))
   labels = np.array(np.split(labels, batches))
   return data, labels
mini_batch_size_ = 71
dev_data, dev_labels = split_data_batches(dev_data, dev_labels, 30, mini_batch_size_)

attention_size = 124

def attention(H, W_attn, b_attn, c_attn, hidden_dimension):#H shape (batch_size, seq_length, hidden_dimension)
   u = tf.tanh(tf.tensordot(H, W_attn, axes=1) + b_attn) #shape (batch_size, seq_length, attention_size)
   uc = tf.tensordot(u, c_attn, axes=1) #shape (batch_size, seq_length)
   alphas = tf.nn.softmax(uc) #shape same as uc
   output = tf.reduce_sum(H * tf.expand_dims(alphas, -1), 1)
   return output, alphas

summary_dir = '/home/bharatp/lid_phase2/RATS/models/board'#Path(saver_conf["tensorboard"]) / "train_{}".format(self.task_index)
#summary_dir.mkdir(exist_ok=True)
summary_writer = tf.summary.FileWriter(str(summary_dir))

x_in = tf.placeholder("float32", [None, 80, None], name='datainput')
y = tf.placeholder("float", [None, 5], name='labelsinput')
y_1s = tf.placeholder("float", [None, 5], name='labelsinput_1s')
global_step = tf.Variable(0, name="global_step", trainable=False)
x = tf.transpose(x_in, perm=[0,2,1])
keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keepprob')
t_batch_size = tf.placeholder(tf.int32, name='batchsize')
learning_rate = tf.placeholder(tf.float32, name='lr')

with tf.variable_scope('first_layer'):
    cell1 = tf.contrib.rnn.GRUCell(hidden_units_1)
    lstm_outputs1, state1 = tf.nn.dynamic_rnn(cell1,x,dtype = tf.float32)
    lstm_outputs1 = lstm_outputs1[:,-1,:]
    out1 = tf.reshape(lstm_outputs1, shape=[-1,10,hidden_units_1])

#with tf.variable_scope('second_layer'):
#    cell2 = tf.contrib.rnn.GRUCell(hidden_units_2)
#    lstm_outputs2, state2 = tf.nn.dynamic_rnn(cell2, lstm_outputs1, dtype = tf.float32)
#    lstm_outputs2 = lstm_outputs2[:,-1,:]
#    out2 = tf.reshape(lstm_outputs2, shape=[-1,10,hidden_units_2])

with tf.variable_scope('third_layer'):
    cell3 = tf.contrib.rnn.GRUCell(hidden_units_3)
    lstm_outputs3, state3 = tf.nn.dynamic_rnn(cell3, out1, dtype = tf.float32)
    lstm_outputs3 = lstm_outputs3[:,-1,:]
    out3 = tf.reshape(lstm_outputs3, shape=[t_batch_size,-1,hidden_units_3], name='embedding_0')

with tf.variable_scope('fourth_layer'):
    cell4_fw = tf.contrib.rnn.GRUCell(hidden_units_4)
    cell4_bw = tf.contrib.rnn.GRUCell(hidden_units_4) 
    lstm_outputs4, states4 = tf.nn.bidirectional_dynamic_rnn(cell4_fw, cell4_bw, out3, dtype = tf.float32)
    lstm_outputs4 = tf.concat(lstm_outputs4, 2)
    w_attn = tf.Variable(tf.truncated_normal([2*hidden_units_4, attention_size], stddev = 0.1))
    b_attn = tf.Variable(tf.random_normal([attention_size], stddev = 0.1))
    c_attn = tf.Variable(tf.random_normal([attention_size], stddev = 0.1))
    lstm_out, alphas = attention(lstm_outputs4, w_attn, b_attn, c_attn, 2*hidden_units_4)
    alphas = tf.identity(alphas, name="attn_weights_1")
    lstm_out = tf.identity(lstm_out, name="embedding_main")

with tf.variable_scope('dense_layer'):
    dense_dim = 256
    training = tf.placeholder(tf.bool)
    w_dense = tf.Variable(tf.truncated_normal([2*hidden_units_4, dense_dim], stddev = 0.1))
    b_dense = tf.Variable(tf.random_normal([dense_dim]))
    h_dense = tf.add(tf.matmul(lstm_out, w_dense), b_dense)
    h_dense = tf.nn.relu(h_dense, name='embedding_dense')

with tf.variable_scope('out_layer'):
    lambda_l2_reg = 0.0
    w_out = tf.Variable(tf.truncated_normal([dense_dim, 5], stddev = 0.1))
    b_out = tf.Variable(tf.random_normal([5]))
    logits = tf.add(tf.matmul(h_dense, w_out), b_out, name='logit')

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
global_step = tf.Variable(0, name="global_step", trainable=False)
predictions = tf.nn.softmax(logits, name='predictions')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y), name='loss')
#l2_loss = lambda_l2_reg * sum(tf.nn.l2_loss(tf_var)
#             for tf_var in tf.trainable_variables()
#             if not ("Bias" in tf_var.name))
#loss = tf.add(loss, l2_loss, name='loss_with_l2')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

correct_pred = tf.equal(tf.arg_max(predictions, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32), name='accuracy')
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print ("Training started.")
logging.info('Training started.')
logging.info(str(datetime.datetime.now()))
lr=1e-4
no_epochs = 100
with tf.Session(config=config) as sess:
    sess.run(init)
    max_val_acc = 0
    batches_1 = batch_iter(train_data_1, train_labels_1, mini_batch_size, no_epochs, 30)
    batches_2 = batch_iter(train_data_2, train_labels_2, mini_batch_size, no_epochs, 30)
    steps = no_epochs*2*len(train_data_1)/mini_batch_size 
    summary = tf.Summary()
    for i in range(steps):
        r = random.random()
        if r<=0.5:
           batch = next(batches_1)
        else:
           batch = next(batches_2)
        _,__, step, train_loss, train_acc = sess.run([train_op, extra_update_ops, global_step, 
                                                      loss, accuracy], 
                                                      feed_dict={x_in:batch[0], y:batch[1], 
                                                      training:True, 
                                                      t_batch_size:mini_batch_size, learning_rate:lr})
        current_step = tf.train.global_step(sess, global_step)
        time_str = datetime.datetime.now().isoformat()
        summary.value.add(tag='train accuray mini batch', simple_value=float(train_acc))
        summary.value.add(tag='train loss mini batch', simple_value=float(train_loss))
        if i % 50 == 0:
           logging.info("{}: step {}, mini batch loss {:g}, mini batch acc {:g}".format(time_str, step, train_loss, train_acc))
           print ("{}: step {}, mini batch loss {:g}, mini batch acc {:g}".format(time_str, step, train_loss, train_acc))
        if i % 250 == 0:
            val_acc = 0
            val_loss= 0.0
            l30s = len(dev_data)

            for d_i, d_batch in enumerate(dev_data):
                d_labels = dev_labels[d_i]
                d_batch = split_data(d_batch, window, hop)
                batch_val_acc, batch_val_loss,_ = sess.run([accuracy, loss, extra_update_ops], feed_dict={x_in: d_batch,
                       y: d_labels, training:False, t_batch_size:mini_batch_size_})
                val_acc += batch_val_acc
                val_loss += batch_val_loss
            val_loss = val_loss/(d_i+1)
            val_acc = val_acc/(d_i+1)
            print "Validation loss, accuracy: " +  str(val_loss) +  ' ' +str(val_acc)
            logging.info('Validation loss, accuracy:'+str(val_loss)+' '+str(val_acc))
            summary.value.add(tag='Validation accuray', simple_value=float(val_acc))
            summary.value.add(tag='Validation loss', simple_value=float(val_loss))
            summary_writer.add_summary(summary, i)
            summary_writer.flush()
            if val_acc > max_val_acc:
               max_val_acc = val_acc
               best_model_path = '/home/bharatp/lid_phase2/RATS/models/checkpoints/hgru_' + str(val_acc)
               saver.save(sess, best_model_path)
                                            


