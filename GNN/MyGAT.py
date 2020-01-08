# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
import time

from sklearn import metrics

import GAT
import os
import csv
from GAT import *

data_dir = "data/GCNdata/20191116 900data newHIN 8class"
out_dir = "mygat_result"

node_size = 947  # 节点数目
train_size = int(node_size * 0.6)  # 训练集数目，取60%
node_embedding = 16  # 节点特征向量长度（tf-idf)

# GCN
hid_units = [8]     # 第一层输出特征的维数
n_heads = [8, 1]    # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu

# train
class_size = 8  # 分类数量
meta_size = 20  # 元路径数量
epoch_num = 15000  # 迭代周期数
learning_rate = 1e-4  # alpha
grad_cut = 1
momentum = 0.9  # 动量，梯度下降用；


# 原始的梯度下降权值更新为 w = w - learning_rate * dw
# 加入动量后变成：v = mu * v - learning_rate * dw； w = w + v
# 其中，v初始化为0，mu是设定的一个超变量，最常见的设定值是0.9。
# 可以这样理解上式：如果上次的momentum(v)与这次的负梯度方向是相同的，那这次下降的幅度就会加大，从而加速收敛

def get_data():
    instance = np.load(os.path.join(data_dir, "instance.npy"))
    label = np.load(os.path.join(data_dir, "label.npy"))
    index = [i for i in range(node_size)]
    np.random.shuffle(index)
    index = index[0:train_size]
    train_data = instance[index]
    train_label = label[index]
    test_index = [i for i in range(node_size) if i not in index]
    test_data = instance[test_index]
    test_label = label[test_index]

    return train_data, train_label, test_data, test_label

# train_data, train_label, test_data, test_label = get_data()
train_data = np.load(os.path.join(data_dir, 'train_data.npy'))
train_label = np.load(os.path.join(data_dir, 'train_label.npy'))
test_data = np.load(os.path.join(data_dir, 'test_data.npy'))
test_label = np.load(os.path.join(data_dir, 'test_label.npy'))


def get_train_matrix(xdata, adj_data):
    train_data_index = train_data.tolist()
    train_data_index = list(map(int, train_data_index))
    print(type(train_data_index[0]))
    train_xdata = xdata[train_data_index, :]
    train_adj_data = list()
    for i in range(adj_data.shape[0]):
        tmp_m = adj_data[i][train_data_index, :]
        tmp_m = tmp_m[:, train_data_index]
        train_adj_data.append(tmp_m)
    train_adj_data_arr = np.array(train_adj_data)
    return train_xdata, train_adj_data_arr


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

# In[8]:

class MyGAT(object):
    def __init__(self, session,
                 meta,
                 nodes,
                 class_size,
                 embedding):
        self.meta = meta
        self.nodes = nodes
        self.class_size = class_size
        self.embedding = embedding

        self.build_placeholders()

        # 前向传播：返回为损失函数值，pair是否同类的预测概率、元路径权重、pair在GCN中得到的分类特征向量
        self.loss, self.probabilities, self.weight, self.v1 = self.forward_propagation()
        # self.l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01), tf.trainable_variables())
        # self.pred = tf.to_int32(2 * self.probabilities)  # 超过0.5即为1，不过即为0
        self.pred = tf.argmax(self.probabilities, 1)
        self.pred = tf.to_int32(self.pred)
        correct_prediction = tf.equal(self.pred, self.t)  # 逐个元素判断是否相等，相等为True，否则为False
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 布尔转浮点，算平均值
        print('Forward propagation finished.')

        # 后向传播
        self.sess = session
        # train()的时候需要这个optimizer，使用Momentum算法进行梯度下降

        # optimizer = tf.train.MomentumOptimizer(self.lr, self.mom)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients = optimizer.compute_gradients(self.loss)
        var_grad = []  # 拿到变量列表后可以做很多事情，比如修改大小，去掉某些不用于学习
        for (g, v) in gradients:
           # if 'weights_n' in v.name:
                var_grad.append((g, v))
        #     # 限定导数值域-1到1
        # capped_gradients = [(tf.clip_by_value(grad, -grad_cut, grad_cut), var) for grad, var in var_grad]
        #     # 将处理后的导数继续应用到BP算法中
        self.optimizer = optimizer.apply_gradients(var_grad)
        print("gradient:" + str(self.optimizer))
        # self.optimizer = tf.train.MomentumOptimizer(self.lr, self.mom).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.global_variables())
        print('Backward propagation finished.')

    # placeholder：占位符，也是一种常量，可理解为“形参”，需要用户传递常数值
    # 创建方式：placeholder(dtype:数据类型, shape:数据形状, name:常量名)
    def build_placeholders(self):
        self.a = tf.placeholder(tf.float32, [self.meta, self.nodes, self.nodes], 'adj')
        self.x = tf.placeholder(tf.float32, [self.nodes, self.embedding], 'nxf')
        self.t = tf.placeholder(tf.int32, [None], 'labels')
        self.p = tf.placeholder(tf.int32, [None], 'train_data')     # 训练用的instance_id序列
        self.lr = tf.placeholder(tf.float32, [], 'learning_rate')
        self.mom = tf.placeholder(tf.float32, [], 'momentum')

    def forward_propagation(self):
        # input ==> (13, 5000, 5000)
        # self.x ==> (5000, 1000)
        # self.t ==> (5000,)
        # attention ==> (1, 5000, 2084)

        # variable_scope: 一种共享变量的机制，管理传给get_variable()的变量名称的作用域
        # 获取元路径权重，根据KIES的计算方式与adj矩阵相乘求和，得到邻接矩阵的数值
        with tf.variable_scope('weights_n'):
            A = tf.reshape(self.a, [self.meta, self.nodes * self.nodes])  # 每个adj矩阵二维转一维
            A_ = tf.transpose(A, [1, 0])  # 转置
            # 创建元路径权重的变量（一个一维向量），初始由xavier_initializer创建
            W = tf.nn.sigmoid(
                tf.get_variable('W', shape=[self.meta, 1], initializer=tf.contrib.layers.xavier_initializer()))
            weighted_adj = tf.matmul(A_, W)  # 元路径权重在此！
            # weighted_adj = 0.1 * weighted_adj
            # (5000*5000, 1)
            '''
            # 尝试邻接矩阵归一化
            wei_max = tf.reduce_max(weighted_adj, axis=0)
            wei_max = tf.expand_dims(wei_max, 1)
            wei_min = tf.reduce_min(weighted_adj, axis=0)
            wei_min = tf.expand_dims(wei_min, 1)
            p_up = weighted_adj - wei_min
            p_down = wei_max - wei_min
            weighted_adj = tf.divide(p_up, p_down)
            weighted_adj = 10 * weighted_adj
            '''
            weighted_adj = tf.reshape(weighted_adj, [1, self.nodes, self.nodes])  # 一维转二维，搞回去
        '''
        # GCN的操作，输入特征矩阵和邻接矩阵，输出为(1, 事件数, 分类数)的矩阵，即分类结果
        with tf.variable_scope('spectral_gcn'):
            gcn_out = GCN(tf.expand_dims(self.x, 0), weighted_adj, [self.gcn_output1, self.gcn_output2, self.class_size]).build()
        '''

        with tf.variable_scope('attention_n'):
            gat_out = GAT(tf.expand_dims(self.x, 0), self.class_size, weighted_adj, node_size, hid_units,
                                    n_heads, nonlinearity, residual).inference()

        # 获取event instance在GCN内训练后对应的分类向量
        with tf.variable_scope('extract_n'):
            p = tf.one_hot(self.p, tf.to_int32(self.nodes))
            p = tf.matmul(p, gat_out[0])    # p.shape=>(train_size, class_size)
            '''
            # 生成one-hot向量的矩阵，即根据id确定行向量里哪个是1，其余都为0
            p1 = tf.one_hot(self.p1, tf.to_int32(self.nodes))
            # 和GCN的分类矩阵相乘，即把属于自己的那一行分类向量提出来
            p1 = tf.matmul(p1, gcn_out[0])
            p2 = tf.one_hot(self.p2, tf.to_int32(self.nodes))
            p2 = tf.matmul(p2, gcn_out[0])
        #            p3 = tf.one_hot([3000], tf.to_int32(self.nodes))
        #            p3 = tf.matmul(p3, gcn_out[0])
            '''

        # 计算事件pair的模之比，得到是否属于同一类的预测值，交叉熵求loss
        with tf.variable_scope('cosine'):
            '''
            # p ==> (batch_size, feature)
            p1_norm = tf.sqrt(tf.reduce_sum(tf.square(p1), axis=1))
            p2_norm = tf.sqrt(tf.reduce_sum(tf.square(p2), axis=1))
            #            p1_p2 = tf.reduce_sum(tf.multiply(p1, p2), axis=1)
            #            cosine = p1_p2 / (p1_norm * p2_norm)
            #            c = tf.expand_dims(cosine, -1)
            #            prob = tf.concat([-c, c], axis=1)
            c = p1_norm / p2_norm
            c = tf.expand_dims(c, -1)  # batch, 1
            c = tf.concat([c, 1 / c], axis=1)
            # c中为每个事件pair的向量模之比的[原值、倒数]，然后做下面的对数变换
            '''
            # true_prob = -tf.log(tf.reduce_max(c, axis=1) - 1 + 1e-8)
            # true_p = tf.expand_dims(true_prob, -1)
            # prob = tf.concat([-true_p, true_p], axis=1)
            '''
            p_max = tf.reduce_max(p, axis=1)
            p_max = tf.expand_dims(p_max, 1)
            p_min = tf.reduce_min(p, axis=1)
            p_min = tf.expand_dims(p_min, 1)
            p_up = p - p_min
            p_down = p_max - p_min
            prob = tf.divide(p_up, p_down)
            '''
            prob = p
            # 交叉熵函数计算loss
            # 交叉熵刻画的是两个概率分布之间的距离，或可以说它刻画的是通过概率分布q来表达概率分布p的困难程度
            # self.t代表正确答案，prob代表的是预测值，交叉熵越小，两个概率的分布约接近
            # loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.one_hot(self.t, class_size), logits=prob)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prob, labels=tf.one_hot(self.t, class_size))
            loss = tf.reduce_mean(loss)
            # loss += tf.nn.l2_loss(W)
        # 整个前向传播过程最终返回：损失函数，pair是否同类的预测值、元路径权重、pair在GCN中得到的分类特征向量
        return loss, prob, W, p

    def train(self, x, a, t, p, learning_rate, momentum=0.9):
        feed_dict = {
            self.x: x,  # xdata
            self.a: a,  # adj_data
            self.t: t,  # label
            self.p: p,  # data
            self.lr: learning_rate,
            self.mom: momentum
        }
        _, loss, acc, pred, w, v1, prob = self.sess.run(
            [self.optimizer, self.loss, self.accuracy, self.pred, self.weight, self.v1, self.probabilities],
            feed_dict=feed_dict)
        # run()函数：在定义了神经网络的前向传播和后向传播之后，直接调用，批量计算，十分好用！
        # run()的参数：理解为(输出，输入)；输出为tf.Tensor的列表，类型为numpy.ndarray；输入为placeholder的列表
        return loss, acc, pred, w, v1, prob

    def test(self, x, a, t, p):
        feed_dict = {
            self.x: x,
            self.a: a,
            self.t: t,
            self.p: p
            #   self.lr: learning_rate,
            #   self.mom: momentum
        }
        acc, pred = self.sess.run([self.accuracy, self.pred], feed_dict=feed_dict)
        return acc, pred


# F1分数为精确率和召回率的调和平均数，用于衡量模型健壮性的指标，最大值为1，最小值为0
def com_f1(pred, label):
    MI_F1 = []
    l = len(pred)
    TP = 0  # 真正类，是指真实标签为1，预测标签的值也为1
    FP = 0  # 假正类，真实标签为0，预测标签值为1
    FN = 0  # 假负类，真实标签为1，预测标签值为0
    TN = 0  # 真负类，真实标签为0，预测标签为0
    f1 = 0
    for i in range(l):
        if pred[i] == 1 and label[i] == 1:
            TP += 1
        elif pred[i] == 1:
            FP += 1
        elif label[i] == 1:
            FN += 1
        else:
            TN += 1
    if TP + FP == 0:
        pre = 0
    else:  # 精确率：用于反映模型预测实例中的精确程度，即预测为正的样本中有多少是真正的正样本
        pre = TP / (TP + FP)
    if TP + FN == 0:
        rec = 0
    else:  # 召回率：用于反映模型的敏感程度，即正确的样本中有多少被预测为正确的样本
        rec = TP / (TP + FN)
    acc = (TP + TN) / l  # 正确率：反映一个模型能预测正确的概率。反映模型的正确程度（算法顾名思义）
    if (pre + rec) != 0:
        f1 = 2 * pre * rec / (pre + rec)  # F1分数：精确率和召回率的调和平均数
    return [pre, rec, acc, f1]


if __name__ == "__main__":
    xdata = np.load(os.path.join(data_dir, "xdata_d2v.npy"))  # 特征矩阵
    adj_data = np.load(os.path.join(data_dir, "20_adj_data.npy"))  # 一串邻接矩阵
    train_xdata, train_adj_data = get_train_matrix(xdata, adj_data)
    PRAF = []  # "Pre","Rec","Acc","F1"

    with open(os.path.join(out_dir, "parameter.txt"), "w") as f:
        f.write("train_size:" + str(train_size) + '\n')
        f.write("node_size:" + str(node_size) + '\n')
        f.write("node_embedding:" + str(node_embedding) + '\n')
        f.write("class_size:" + str(class_size) + '\n')
        f.write("meta_size:" + str(meta_size) + '\n')
        f.write("hid_units:" + str(hid_units) + '\n')
        f.write("n_heads:" + str(n_heads) + '\n')
        f.write("learning_rate:" + str(learning_rate) + '\n')
        f.write("grad_cut:" + str(grad_cut) + '\n')

    with tf.Session() as sess:
        net = MyGAT(class_size=class_size, meta=meta_size, nodes=node_size,
                    session=sess, embedding=node_embedding)
        sess.run(tf.global_variables_initializer())  # 第一次run，神经网络初始化

        minloss = maxacc = 0
        max_acc = 0
        for epoch in range(epoch_num):  # 神经网络执行次数
                loss, acc, pred, w, v1, prob = net.train(xdata, adj_data, train_label, train_data, learning_rate, momentum)
                # print("prob: " + str(prob[0]))

                print("loss: {:.4f} ,acc: {:.4f}".format(loss, acc))
                #                    print(pdata[0])
                #                    print(v1)
                #                    print(v2)
                #                    print(prob)
                # if index % 3984 == 0:
                    # print(test_data[:,0].shape)
                    # print(test_data[:,1].shape)
                eva_acc, eva_pred = net.test(xdata, adj_data, test_label, test_data)
                # PRAF.append(com_f1(eva_pred, test_label))
                with open(os.path.join(out_dir, "test_acc.txt"), "a+") as f:
                    f.write(str(eva_acc))
                    f.write("\n")
                print(str(epoch) + '------------------------------------------------------Test acc: {:.4f}'.format(eva_acc))

                if acc > max_acc:
                    max_acc = acc
                    f = open(os.path.join(out_dir, "results_new.txt"), 'w')
                    f.write('batch accuracy:' + str(acc))
                    f.write('\n')
                    f.write('weight:' + str(w))
                    f.write('\n')
                    NMI = metrics.normalized_mutual_info_score(test_label, eva_pred)
                    f.write('NMI:' + str(NMI))
                    f.close()
                    print('batch accuracy:', acc)
                    print('golden label:', test_label)
                    print('pred label:', eva_pred)
                    print('weight:', w)
                    net.saver.save(sess, "mygat_result/gnn_model")
                    print('********************* Model Saved *********************')

                with open(os.path.join(out_dir, "train_acc.txt"), "a+") as f:
                    f.write(str(acc))
                    f.write("\n")
                print("epoch{:d} : , train_loss: {:.4f} ,train_acc: {:.4f}".format(epoch, loss, acc))
                '''
                with open(os.path.join(out_dir, "PRAF.csv")", "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Pre", "Rec", "Acc", "F1"])
                    for d in PRAF:
                        writer.writerow(d)
                '''
                if maxacc < acc:
                    maxacc = acc
                    minloss = loss

                #if eva_acc == 0:
                    #break
        f = open(os.path.join(out_dir, "results_final.txt"), 'w')
        f.write('batch accuracy:' + str(acc))
        f.write('\n')
        f.write('weight:' + str(w))
        f.write('\n')
        f.close()

        print("train end!")
        print("The loss is {:.4f},The acc is{:.4f}".format(minloss, maxacc))

# In[ ]:



