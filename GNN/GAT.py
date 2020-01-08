
# coding: utf-8

'''


Modified on 18/8/4.
Copyright 2018. All rights reserved.
License from MIT.

'''

import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

def relu(x, alpha=0., max_value=None):
    '''
    ReLU.
    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

# 作用：计算注意力系数+加权求和
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    # 实际只用到了输入（seq），输出size（out_sz），activation（激活函数）
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # conv1d三个参数：input输入，filters卷积核数目，kernel_size卷积核大小
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        # 8个过滤器，卷积核宽度为1，即最后输出为（1，node_size，8）

        # simplest self-attention possible
        # self-attention机制：向量空间到常数空间的映射
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        # coefs = tf.nn.softmax(relu(logits))# + bias_mat)
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
        # coefs为注意力系数，即每次卷积时，用来进行加权求和的系数

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)
        # ret为线性组合后每个节点的输出特征

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return tf.nn.elu(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

class BaseGAttN:
    '''
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    def training(loss, lr, l2_coef):
        # weight decay
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        # training op
        train_op = opt.minimize(loss+lossL2)
        
        return train_op
    '''
    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)

class GAT(BaseGAttN):
    def __init__(self, x, class_size, weighted_adj, node_size, hid_units, n_heads, nonlinearity, residual, **kwargs):
        super(GAT, self).__init__(**kwargs)
        self.inputs = x
        self.nb_classes = class_size
        self.bias_mat = weighted_adj
        self.nb_nodes = node_size
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = nonlinearity
        self.residual = residual


    def inference(self):
        # 传入的实参为：gcn_out,64,0,hid_units,n_heads,nonlinearity,residual
        # 猜测：输入的特征向量，分类数，偏置（无用），hid_inits, n_heads, 激活函数， residual（无用）
        # hid_inits: numbers of hidden units per each attention head in each layer（github示例程序中为[8]）
        # n_heads: additional entry for the output layer（github示例程序中为[8, 1]）多头注意力里"头"的数目
        attns = []
        for _ in range(self.n_heads[0]):
            attns.append(attn_head(self.inputs, bias_mat=self.bias_mat,
                out_sz=self.hid_units[0], activation=self.activation,residual=False))
        h_1 = tf.concat(attns, axis=-1)
        # attn_head(inputs)拼接形成h_1

        # 由于n_heads这里只有两维，github代码中对中间维度的处理在这里去掉了
        
        out = []
        for i in range(self.n_heads[-1]):    # 下标-1为最后一维，即1
            out.append(attn_head(h_1, bias_mat=self.bias_mat,
                out_sz=self.nb_classes, activation=lambda x: x, residual=False))
        logits = tf.add_n(out) / self.n_heads[-1]    # 最后一个层，不再合并attention，而是求平均
        logits = tf.nn.softmax(logits)  # 这一句是比github多出来的
        return logits





