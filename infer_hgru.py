import numpy as np
import tensorflow as tf
import sys
import datetime

test_pred_file_path = sys.argv[1]
trained_model_path = sys.argv[2] + '.meta'

print(datetime.datetime.now())

label_map_path = '/home/data/rats_lid/docs/fileset_info.tab'
label_map = np.loadtxt(label_map_path, dtype='str', skiprows=1, usecols=(0,3))
label_dict = {'alv':0, 'fas':1, 'prs':2, 'pus':3, 'urd':4, 'mul':5}

def get_label(fname):
   label = label_map[np.where(label_map == fname)[0]][0,1]
   if label in label_dict:
      return label_dict[label]
   else:
      return label_dict['mul']

def slice_patch(a, l):
    a = a.T
    nd0 = a.shape[0] - l + 1
    m,n = a.shape
    s0,s1 = a.strides
    b = np.lib.stride_tricks.as_strided(a, shape=(nd0,l,n), strides=(s0,s0,s1))
    c = np.swapaxes(b,1,2)
    return c

def prep_data_for_infer(a):
         l = a.shape[1]
         s = l/1010
         a = a[:,0:(1010*s)]
         b = np.split(a,s,axis=1)
         a = np.stack(b)
         return a

def prep_data_for_infer_old(a):
     #a2 = a
     l = a.shape[1]
     if l >= 1000:
         s = l/1000
         a = a[:,0:(1000*s)]
         b = np.split(a,s,axis=1)
         a = np.stack(b)
         #a = slice_patch(a,1000)
     else:
         rep = int(1000/l) + 1
         a = np.concatenate([a]*rep, axis=1)
         a = a[:,0:1000]
         a = np.reshape(a,(1,a.shape[0],a.shape[1]))
     return a

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

def adjust_data_new(data, size):
    if size==10:
        d = data[:,:,0:910]
    else:
        l = random.randint(0,data.shape[2]-10)
        chunk = data[:,:,l:l+10]
        d = np.append(data, chunk, axis=2)
    return d

def adjust_data_1s(a):
   l = a.shape[1]
   #if l > 1010:
   #   a = prep_data_for_infer(a)
   #   return a
   if (l-10)%100 == 0:
      a = np.reshape(a, (1,a.shape[0],a.shape[1]))
      return np.vstack([a])
   else:
      x = (((l)/100) + 1)*100 + 10
      a = np.concatenate([a]*2, axis=1)
      if a.shape[1] < x:
         a = np.concatenate([a]*2, axis=1)
      a = a[:,0:x]
      a = np.reshape(a,(1,a.shape[0],a.shape[1]))
      return a

def adjust_data_3s(a):
   l = a.shape[1]
   if (l-10)%300 == 0:
      a = np.reshape(a, (1,a.shape[0],a.shape[1]))
      return np.vstack([a])
   else:
      x = (((l)/300) + 1)*300 + 10
      a = np.concatenate([a]*2, axis=1)
      if a.shape[1] < x:
         a = np.concatenate([a]*3, axis=1)
      a = a[:,0:x]
      a = np.reshape(a,(1,a.shape[0],a.shape[1]))
      return a

def adjust_data_simple(a):
   l = a.shape[1]
   if (l-20)%10 == 0:
      a = np.reshape(a, (1,a.shape[0],a.shape[1]))
      return np.vstack([a])
   else:
      x = (((l)/10) + 1)*10 + 20
      a = np.concatenate([a]*2, axis=1)
      if a.shape[1] < x:
         a = np.concatenate([a]*2, axis=1)
      a = a[:,0:x]
      a = np.reshape(a,(1,a.shape[0],a.shape[1]))
      return a

def prep_batch_labels(labels, seq_len):
    n = labels.shape[0]
    labels = np.repeat(labels, seq_len, axis=0)
    labels = np.array(np.split(labels, n, axis=0))
    return labels

#config = tf.ConfigProto()
config = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
#config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(trained_model_path)
    saver.restore(sess, trained_model_path.rsplit('.',1)[0])
    graph = tf.get_default_graph()
    input_x = graph.get_tensor_by_name('datainput:0')
    b_size = graph.get_tensor_by_name('batchsize:0')
    pred_op = graph.get_operation_by_name('predictions')
    test_pred_file = open(test_pred_file_path,'w')
    bnf_dev_lst = '/home/bharatp/ieee_taslp/lists/RATS/dev_1_bnf_all_shuffled.lst'
    test_acc = 0
    with open(bnf_dev_lst) as f:
       for i,l in enumerate(f):
           if (i+1)%500 == 0:
               print 'processed ' + str(i+1) + ' files.'
               print test_acc*1.0/i
           l = l.rstrip()
           a = np.load(l)
           fname, channel = l.rsplit('/', 1)[-1].split('.')[0].rsplit('_', 1)
           true_label = get_label(fname)
           if a.shape[1] == 0:
              prediction = [0.2]*5
              test_pred_file.write(fname + '\t' + str(prediction) + '\n')
              continue
           print a.shape
           a = adjust_data_1s(a)
           print a.shape
           a = split_data(a, 20, 10)
           prediction = sess.run(pred_op.outputs, feed_dict = {input_x:a, b_size:1})
           #print prediction
           pred_label = np.argmax(prediction[0])
           #print pred_label
           prediction = prediction[0].tolist()
           if pred_label == true_label:
                test_acc += 1
           test_pred_file.write(fname + '\t' + str(prediction) + '\n')
    test_pred_file.close()
    print ('Test acc:', test_acc*1.0/i)
print(datetime.datetime.now())
