import numpy as np
import random
import tensorflow as tf

window = 20
hop = 10
data_list = '/Users/bharat/Desktop/RATS_sample/rats_sample.lst'
sample_len = 3010
batch_size = 64
batch_size_val = 71
data_npy = np.genfromtxt(data_list, dtype='str')
label_dict = {'alv':0, 'fas':1, 'prs':2, 'pus':3, 'urd':4, 'mul':5}

dev_data = ''
dev_labels = ''

def random_chunk(d):
   l = d.shape[1]
   if l > sample_len:
     rd_idx = random.sample(range(l-sample_len),1)[0]
     sample = d[:,rd_idx:(rd_idx+sample_len)]
   else:
     sample = d
   return sample[np.newaxis,:]

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

def prep_batch(data_sample):
    data = []
    labels = []
    for sample in data_sample:
        label = label_dict[sample.rsplit('/', 3)[1]]
        try:
            sample = np.load(sample)
        except Exception:
            continue
        l = sample.shape[1]
        if l >= sample_len:
            sample = random_chunk(sample)
            data.append(sample)
            labels.append(label)
    data = np.vstack(data)
    labels = np.hstack(labels)
    return data, labels

def adjust_data(data):
    l = random.randint(0,data.shape[2]-10)
    chunk = data[:,:,l:l+10]
    d = np.append(data, chunk, axis=2)
    return d

def split_data_batches(data, labels, mini_batch_size_):
   data = adjust_data(data)
   d_len = len(data)
   batches = d_len/mini_batch_size_
   data = data[0:(batches*mini_batch_size_)]
   labels = labels[0:(batches*mini_batch_size_)]
   data = np.array(np.split(data, batches))
   labels = np.array(np.split(labels, batches))
   return data, labels

mini_batch_size_ = 71
dev_data, dev_labels = split_data_batches(dev_data, dev_labels, batch_size_val)
num_dev_batches = len(dev_data)

def gen_batch_val():
    for d_i, d_batch in enumerate(dev_data):
        dev_labels_batch = dev_labels[d_i]
        dev_data_batch = split_data(d_batch, window, hop)
        yield dev_data_batch, dev_labels_batch

def gen_batch_train():
    while True:
        data_sample = data_npy[np.random.choice(data_npy.shape[0], batch_size, replace=False)]
        data, labels = prep_batch(data_sample)
        #print(data.shape, labels.shape)
        data = split_data(data, window, hop)
        yield data, labels

dataset_train = tf.data.Dataset.from_generator(gen_batch_train, output_types=(tf.float64, tf.int32),
                                               output_shapes=(tf.TensorShape([None, 80, 20]), tf.TensorShape([None])))
dataset_train = dataset_train.prefetch(2)

dataset_val = tf.data.Dataset.from_generator(gen_batch_val, output_types=(tf.float64, tf.int32),
                                             output_shapes=(tf.TensorShape([None, 80, 20]), tf.TensorShape([None])))
dataset_val = dataset_val.prefetch(2)

iterator = tf.data.Iterator.from_structure(dataset_train.output_types,
                                           dataset_train.output_shapes)
next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(dataset_train)
validation_init_op = iterator.make_initializer(dataset_val)

'''
iterator_train = dataset_train.make_one_shot_iterator()
batch = iterator_train.get_next()
with tf.Session() as sess:
    print('From iterator:')
    a = sess.run(batch[0])
    print(a.shape)
    #print (a[0].shape, a[1].shape)
'''