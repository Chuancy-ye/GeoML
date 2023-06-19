import numpy as np
from bean import nn, autoencoder


# 建立autoencoder框架
def aebuilder(nodes):
    layers = len(nodes)
    ae = autoencoder()  # 获取autoencoder类
    for i in range(layers-1):
        # 训练时，我们令输入等于输出，所以每一个训练时的autoencoder层为[n1,n2,n1]形式的结构
        ae.add_one(nn([nodes[i], nodes[i + 1], nodes[i]]))
    return ae


# 训练autoencoder，ae为Autoencoder的训练模型，x为输入，interactions为训练迭代次数
def aetrain(ae, x, interations):
    elayers = len(ae.encoders)
    for i in range(elayers):
        # 单层Autoencoder训练
        ae.encoders[i] = nntrain(ae.encoders[i], x, x, interations)
        # 单层训练后，获取该Autoencoder层中间层的值，作为下一层的训练输入
        nntemp = nnff(ae.encoders[i], x, x)
        # feed forward
        x = nntemp.values[1]
    return ae


# 对神经网络进行训练
def nntrain(nn, x, y, iterations):
    for i in range(iterations):
        nnff(nn, x, y)
        nnbp(nn)
    return nn


# 前馈函数
def nnff(nn,x,y):
    layers = nn.layers
    numbers = x.shape[0]
    # 赋予初值
    nn.values[0] = x

    for i in range(1, layers):
        nn.values[i] = sigmod(np.dot(nn.values[i-1], nn.W[i-1])+nn.B[i-1])

    # 初始化P值(增加)
    for j in range(1, layers-1):
        nn.P[j] = nn.values[j].sum(axis = 0)/(nn.values[j].shape[0])

    # 最后一层与实际的误差
    nn.error = y - nn.values[layers-1]
    # 计算KL项
    sparsity = nn.sparse*np.log(nn.sparse/nn.P[layers-2])
    +(1-nn.sparse)*np.log((1-nn.sparse)/(1-nn.P[layers-2]))
    # 修改loss
    nn.loss = 1.0/2.0 * (nn.error**2).sum()/numbers + nn.beta * sparsity.sum()
    return nn


# 激活函数
def sigmod(x):
    return 1.0 / (1.0 + np.exp(-x))

# BP函数
def nnbp(nn):
    layers = nn.layers;
    # 初始化delta

    deltas = list();
    for i in range(layers):
        deltas.append(0)

    # 最后一层的delta为
    deltas[layers - 1] = -nn.error * nn.values[layers - 1] * (1 - nn.values[layers - 1])
    # 其他层的delta为
    for j in range(1, layers - 1)[::-1]:  # 倒过来
        # 求deltas的同时，要求spare这个惩罚项,因为在Python中，如果是向量，都默认为N*.的矩阵，所以不能用dot,只能用点乘
        # 还有我们给他重复了样本的数量次，(增加)
        pj = np.ones([nn.values[j].shape[0], 1]) * nn.P[j]
        sparsity = nn.beta * (-nn.sparse / pj + (1 - nn.sparse) / (1 - pj))
        # deltas进行了修改
        deltas[j] = (np.dot(deltas[j + 1], nn.W[j].T) + sparsity) * nn.values[j] * (1 - nn.values[j])
    # 更新W值
    for k in range(layers - 1):
        nn.W[k] -= nn.u * np.dot(nn.values[k].T, deltas[k + 1]) / (deltas[k + 1].shape[0])
        nn.B[k] -= nn.u * deltas[k + 1] / (deltas[k + 1].shape[0])
    return nn
