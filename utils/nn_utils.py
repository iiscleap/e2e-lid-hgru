import tensorflow as tf


def attention(attn_input, attn_size, name= ''):  # attn_in shape (batch_size, seq_length, hidden_dimension)
    with tf.variable_scope(name):
        w_attn = tf.Variable(tf.truncated_normal([int(attn_input.get_shape()[1]), attn_size], stddev=0.1))
        b_attn = tf.Variable(tf.random_normal([attn_size], stddev=0.1))
        c_attn = tf.Variable(tf.random_normal([attn_size], stddev=0.1))
        u = tf.tanh(tf.tensordot(attn_input, w_attn, axes=1) + b_attn)  # shape (batch_size, seq_length, attention_size)
        uc = tf.tensordot(u, c_attn, axes=1)  # shape (batch_size, seq_length)
        alphas = tf.nn.softmax(uc)  # shape same as uc
        output = tf.reduce_sum(attn_input * tf.expand_dims(alphas, -1), 1)
    return output, alphas


def gru_layer(gru_input, cell_size, name=''):
    with tf.variable_scope(name):
        gru_cell_stack = [tf.contrib.rnn.GRUCell(size) for size in cell_size]
        gru_cells = tf.nn.rnn_cell.MultiRNNCell(gru_cell_stack)
        gru_outputs, gru_states = tf.nn.dynamic_rnn(gru_cells, gru_input, dtype=tf.float32)
        # Using output of last timestep
        gru_output = gru_outputs[:, -1, :]
    return gru_output


def bi_gru_layer(gru_input, cell_size, name=''):
    with tf.variable_scope(name):
        cell_fw = tf.contrib.rnn.GRUCell(cell_size)
        cell_bw = tf.contrib.rnn.GRUCell(cell_size)
        gru_outputs, gru_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, gru_input, dtype=tf.float32)
    return tf.concat(gru_outputs, 2)


def dense_layer(dense_input, num_units, name = ''):
    with tf.variable_scope(name):
        w_dense = tf.Variable(tf.truncated_normal([int(dense_input.get_shape()[1]), num_units], stddev=0.1))
        b_dense = tf.Variable(tf.random_normal([num_units]))
        h_dense = tf.nn.relu(tf.add(tf.matmul(dense_input, w_dense), b_dense), name = name)
    return h_dense

