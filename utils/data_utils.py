import numpy as np
import random


def random_chunk(d, sample_len):
    l = d.shape[1]
    if l > sample_len:
        rd_idx = random.sample(range(l-sample_len),1)[0]
        sample = d[:,rd_idx:(rd_idx+sample_len)]
    else:
        sample = d
    return sample[np.newaxis,:]


def split_data(data, window, hop):
    seq_len = data.shape[2]
    l = int((seq_len-window)/hop)+1
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


def prep_batch(data_sample, label_dict, sample_len):
    data = []
    labels = []
    for sample in data_sample:
        label = label_dict[sample.rsplit('/', 3)[1]]
        try:
            sample = np.load(sample)
        except Exception:
            continue
        s_len = sample.shape[1]
        if s_len >= sample_len:
            sample = random_chunk(sample, sample_len)
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


def split_data_batches(data, labels, batch_size):
    data = adjust_data(data)
    d_len = len(data)
    batches = d_len/batch_size
    data = data[0:(batches*batch_size)]
    labels = labels[0:(batches*batch_size)]
    data = np.array(np.split(data, batches))
    labels = np.array(np.split(labels, batches))
    return data, labels
