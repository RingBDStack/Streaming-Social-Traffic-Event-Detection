import tensorflow as tf
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.gaussian_process.kernels import RBF

_LAYER_UIDS = {}

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    # 在卷积函数内调用时，x为特征矩阵，y为[特征矩阵维度，GCN参数（512, 256）]
    # tf.layers.conv2d: 二维卷积层的函数接口
    # 参数输入: tf.layers.conv2d(x:tensor输入, filters:输出空间的维数（即卷积过滤器的数量）,kernel_size:卷积窗的高和宽)
    res = tf.layers.conv2d(x,y[1],[1 ,y[0]])
    # 返回值：[filter_height, filter_width, in_channels, out_channels]
    return res[:, :, 0, :]


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input, adj_matrix , output_dim, dropout=0.,act=tf.nn.relu, bias=False,
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout  # 一种算法，在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃

        self.act = act
        self.adj_matrix = adj_matrix
        
        self.bias = bias        # 偏置
        self.input = input
        self.output_dim = output_dim

        with tf.variable_scope(self.name + '_vars'):
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
                
    def call(self):                                      
        with tf.name_scope(self.name):
             outputs = self._call(self.input)
        return outputs

    def _call(self, inputs):
        #print(inputs)
        x = inputs
        x = tf.nn.dropout(x, self.dropout)  # 实际调用时为0.5，即每次丢一半
                                          
        # convolve
        x = tf.matmul(self.adj_matrix,x)
        print(x.shape)
        # 卷积计算主要在这一句，dot()内调用了layers.conv2d
        pre_sup = dot(tf.expand_dims(x,-1),[int(self.input.shape[-1]),int(self.output_dim)])
        output = pre_sup

        # bias
        if self.bias:
            output += self.vars['bias']
        # 最后应用激活函数，返回矩阵
        return self.act(output)

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
    
        self.vars = {}
        self.placeholders = {}


        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

    def build(self):
        raise NotImplementedError

class GCN(Model):
    def __init__(self, x, adj_matrix,output_dim, dropout = 0.5,**kwargs):
        super(GCN, self).__init__(**kwargs)
        # 注释为PPGCN.py中传入的参数
        self.input = x                  # tf.expand_dims(self.x, 0)，特征矩阵维度加1，由(n, n)变为(1, n, n)
                                        # 不加这一维就和adj_matrix维度不一致，无法卷积
        self.adj_matrix = adj_matrix    # weighted_adj，计算好的邻接矩阵
        self.dropout = dropout          # 默认值在上面
        self.output_dim = output_dim    # [self.gcn_output1, self.gcn_output2, self.class_size]
                                        # 依次为GCN的两个参数 和 分类的数目
        
    def build(self):
        if len(self.output_dim) == 0:
            return self.input

        outputs = GraphConvolution(input = self.input,
                        adj_matrix = self.adj_matrix,
                        output_dim=self.output_dim[0],
                        act=tf.nn.relu,
                        dropout=self.dropout).call()

        for i in range(1,len(self.output_dim)):
            outputs = GraphConvolution(input=outputs,
                            adj_matrix = self.adj_matrix,
                            output_dim=self.output_dim[i],
                            act=tf.nn.relu,
                            dropout=self.dropout).call()

        # GraphConvolution相当于执行了3次，输出维度的前两维为(1，事件数量)；
        # 最后一维3次分别为embedding长度、GCN参数1、GCN参数2，最后输出的第3维为分类数量
        return outputs
