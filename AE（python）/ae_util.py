import numpy as np
from bean import nn, autoencoder


# 建立autoencoder框架
def aebuilder(nodes):
    layers = len(nodes)
    ae = autoencoder()  # 获取autoencoder类
    for i in range(layers - 1):
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
def nnff(nn, x, y):
    layers = nn.layers
    # 获取训练样本数量
    numbers = x.shape[0]
    # 赋予初值
    nn.values[0] = x
    for i in range(1, layers):
        nn.values[i] = sigmod(np.dot(nn.values[i - 1], nn.W[i - 1]) + nn.B[i - 1])
    # 最后一层与实际的误差
    nn.error = y - nn.values[layers - 1]
    # 代价损失
    nn.loss = 1.0 / 2.0 * (nn.error ** 2).sum() / numbers
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
        deltas[j] = np.dot(deltas[j + 1], nn.W[j].T) * nn.values[j] * (1 - nn.values[j])
    # 更新W值和B值
    for k in range(layers - 1):
        nn.W[k] -= nn.u * np.dot(nn.values[k].T, deltas[k + 1]) / (deltas[k + 1].shape[0])
        nn.B[k] -= nn.u * deltas[k + 1] / (deltas[k + 1].shape[0])
    return nn
